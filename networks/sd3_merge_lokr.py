import argparse
import math
import os
import time
from typing import Any, Dict, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library.utils import setup_logging, str_to_dtype, MemoryEfficientSafeOpen, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

import lora_sd3 as lora_sd3
from library import sai_model_spec, train_util, sd3_utils


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    回傳輸入維度分解的兩個值，第二個值大於或等於第一個值
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def rebuild_tucker(t, wa, wb):
    """
    重建 Tucker 分解的權重
    """
    return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)


def make_kron(w1, w2, scale):
    """
    使用 Kronecker 乘積重建權重
    """
    # 檢查輸入權重的有效性
    if not torch.isfinite(w1).all() or not torch.isfinite(w2).all():
        logger.warning("Input weights contain invalid values in make_kron")
        w1 = torch.nan_to_num(w1, nan=0.0, posinf=0.0, neginf=0.0)
        w2 = torch.nan_to_num(w2, nan=0.0, posinf=0.0, neginf=0.0)

    # 檢查 scale 的有效性
    if not torch.isfinite(torch.tensor(scale)):
        logger.warning(f"Invalid scale value in make_kron: {scale}, using 1.0")
        scale = 1.0

    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()

    try:
        # 檢查輸入權重的數值範圍，防止極端值
        w1_max = torch.max(torch.abs(w1))
        w2_max = torch.max(torch.abs(w2))

        if w1_max > 1000 or w2_max > 1000:
            logger.warning(f"Input weights have extreme values: w1_max={w1_max}, w2_max={w2_max}, normalizing...")
            # 正規化權重到合理範圍
            if w1_max > 1000:
                w1 = w1 * (1000.0 / w1_max)
            if w2_max > 1000:
                w2 = w2 * (1000.0 / w2_max)

        rebuild = torch.kron(w1, w2)

        # 檢查 Kronecker 乘積結果的有效性
        if not torch.isfinite(rebuild).all():
            logger.warning("Kronecker product contains invalid values, cleaning up...")
            rebuild = torch.nan_to_num(rebuild, nan=0.0, posinf=0.0, neginf=0.0)

        if scale != 1:
            rebuild = rebuild * scale

            # 檢查縮放後結果的有效性
            if not torch.isfinite(rebuild).all():
                logger.warning("Scaled result contains invalid values, cleaning up...")
                rebuild = torch.nan_to_num(rebuild, nan=0.0, posinf=0.0, neginf=0.0)

            # 檢查結果的數值範圍
            result_max = torch.max(torch.abs(rebuild))
            if result_max > 10000:
                logger.warning(f"Result has extreme values: {result_max}, clamping...")
                rebuild = torch.clamp(rebuild, -10000, 10000)

        return rebuild

    except Exception as e:
        logger.error(f"Error in make_kron: {e}")
        # 返回零張量作為 fallback
        if w1.dim() == 2 and w2.dim() == 2:
            result_shape = (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])
        else:
            # 對於卷積層，需要更複雜的形狀計算
            result_shape = w1.shape[:-1] + (w1.shape[-1] * w2.shape[-1],)

        return torch.zeros(result_shape, dtype=w1.dtype, device=w1.device)


def rebuild_lokr_weight(w1, w1a, w1b, w2, w2a, w2b, t2, alpha, rank):
    """
    重建 LoKr 權重
    Args:
        w1, w1a, w1b: 第一組權重矩陣
        w2, w2a, w2b: 第二組權重矩陣
        t2: Tucker 分解矩陣（可選）
        alpha: 縮放係數（在 LoKr 中實際上是 dim 值）
        rank: 秩
    Returns:
        torch.Tensor: 重建的權重差異
    """
    # 調試信息
    logger.debug(f"rebuild_lokr_weight called with alpha={alpha}, rank={rank}")
    if w1a is not None:
        logger.debug(f"w1a stats: min={w1a.min().item():.6f}, max={w1a.max().item():.6f}, mean={w1a.mean().item():.6f}")
    if w1b is not None:
        logger.debug(f"w1b stats: min={w1b.min().item():.6f}, max={w1b.max().item():.6f}, mean={w1b.mean().item():.6f}")
    if w2a is not None:
        logger.debug(f"w2a stats: min={w2a.min().item():.6f}, max={w2a.max().item():.6f}, mean={w2a.mean().item():.6f}")
    if w2b is not None:
        logger.debug(f"w2b stats: min={w2b.min().item():.6f}, max={w2b.max().item():.6f}, mean={w2b.mean().item():.6f}")

    # 在 LoKr 中，alpha 實際上是 dim 值，我們需要從權重矩陣中提取真正的 rank
    if w1a is not None:
        rank = w1a.shape[1]
    elif w2a is not None:
        rank = w2a.shape[1]
    else:
        # 如果沒有 w1a 或 w2a，使用 alpha 作為 rank（因為 alpha 實際上是 dim）
        rank = alpha if rank is None else rank

    # 防止除零錯誤和數值問題
    if rank <= 0:
        logger.warning(f"Invalid rank value: {rank}, using alpha as fallback")
        rank = max(1, abs(alpha)) if alpha != 0 else 1

    # 檢查 alpha 的有效性
    if not torch.isfinite(torch.tensor(alpha)):
        logger.warning(f"Invalid alpha value: {alpha}, using rank as fallback")
        alpha = float(rank)

    # 在 LoKr 中，scale 計算應該是 alpha / rank
    # 但由於 alpha 實際上是 dim 值，這個計算仍然是正確的
    scale = alpha / rank

    # 限制 scale 的範圍，防止數值不穩定
    if scale > 1000:
        logger.warning(f"Scale value {scale} is too large, clamping to 1000")
        scale = 1000.0
    elif scale < 0.001:
        logger.warning(f"Scale value {scale} is too small, clamping to 0.001")
        scale = 0.001

    # 檢查 scale 的有效性
    if not torch.isfinite(torch.tensor(scale)):
        logger.warning(f"Invalid scale value: {scale}, using 1.0 as fallback")
        scale = 1.0

    # 重建第一組權重
    if w1 is None:
        if w1a is not None and w1b is not None:
            # 檢查輸入權重的有效性
            if torch.isfinite(w1a).all() and torch.isfinite(w1b).all():
                # 預處理：限制極端值
                w1a_clamped = torch.clamp(w1a, -100000, 100000)
                w1b_clamped = torch.clamp(w1b, -100000, 100000)
                w1 = w1a_clamped @ w1b_clamped
            else:
                logger.warning("w1a or w1b contains invalid values, using zeros")
                w1 = torch.zeros(w1a.shape[0], w1b.shape[1], dtype=w1a.dtype, device=w1a.device)
        else:
            logger.error("Cannot rebuild w1: both w1a and w1b are None")
            return None

    # 重建第二組權重
    if w2 is None:
        if t2 is None:
            # 標準 LoKr 分解
            if w2a is not None and w2b is not None:
                # 檢查輸入權重的有效性
                if torch.isfinite(w2a).all() and torch.isfinite(w2b).all():
                    # 預處理：限制極端值
                    w2a_clamped = torch.clamp(w2a, -100000, 100000)
                    w2b_clamped = torch.clamp(w2b, -100000, 100000)

                    if w2b.dim() > 2:
                        # 處理卷積層
                        r, o, *k = w2b.shape
                        w2 = w2a_clamped @ w2b_clamped.view(r, -1)
                        w2 = w2.view(-1, o, *k)
                    else:
                        # 處理線性層
                        w2 = w2a_clamped @ w2b_clamped
                else:
                    logger.warning("w2a or w2b contains invalid values, using zeros")
                    if w2b.dim() > 2:
                        r, o, *k = w2b.shape
                        w2 = torch.zeros(w2a.shape[0], o, *k, dtype=w2a.dtype, device=w2a.device)
                    else:
                        w2 = torch.zeros(w2a.shape[0], w2b.shape[1], dtype=w2a.dtype, device=w2a.device)
            else:
                logger.error("Cannot rebuild w2: both w2a and w2b are None")
                return None
        else:
            # Tucker 分解
            if w2a is not None and w2b is not None and t2 is not None:
                if torch.isfinite(t2).all() and torch.isfinite(w2a).all() and torch.isfinite(w2b).all():
                    w2 = rebuild_tucker(t2, w2a, w2b)
                else:
                    logger.warning("t2, w2a, or w2b contains invalid values, using zeros")
                    w2 = torch.zeros(w2a.shape[0], w2b.shape[1], dtype=w2a.dtype, device=w2a.device)
            else:
                logger.error("Cannot rebuild w2: missing components for Tucker decomposition")
                return None

    # 檢查重建權重的有效性
    if w1 is None or w2 is None:
        logger.error("Failed to rebuild weights")
        return None

    if not torch.isfinite(w1).all() or not torch.isfinite(w2).all():
        logger.warning("Rebuilt weights contain invalid values, cleaning up...")
        w1 = torch.nan_to_num(w1, nan=0.0, posinf=0.0, neginf=0.0)
        w2 = torch.nan_to_num(w2, nan=0.0, posinf=0.0, neginf=0.0)

    # 使用 Kronecker 乘積重建最終權重
    result = make_kron(w1, w2, scale)

    # 最終檢查結果的有效性
    if result is not None and not torch.isfinite(result).all():
        logger.warning("Final result contains invalid values, cleaning up...")
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def load_state_dict(file_name, dtype):
    """
    載入狀態字典
    """
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict: Dict[str, Union[Any, torch.Tensor]], dtype, metadata, mem_eff_save=False):
    """
    保存到文件
    """
    if dtype is not None:
        logger.info(f"轉換至 {dtype}...")
        for key in tqdm(list(state_dict.keys())):
            if type(state_dict[key]) == torch.Tensor and state_dict[key].dtype.is_floating_point:
                # 檢查張量的數值有效性
                if not torch.isfinite(state_dict[key]).all():
                    logger.warning(f"Tensor {key} contains invalid values before dtype conversion, cleaning up...")
                    state_dict[key] = torch.nan_to_num(state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

                # 進行類型轉換
                try:
                    state_dict[key] = state_dict[key].to(dtype)
                except RuntimeWarning as e:
                    if "invalid value encountered in cast" in str(e):
                        logger.warning(f"RuntimeWarning during dtype conversion for {key}: {e}")
                        # 再次清理無效值並重試
                        state_dict[key] = torch.nan_to_num(state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)
                        state_dict[key] = state_dict[key].to(dtype)
                    else:
                        raise

    logger.info(f"保存至: {file_name}")
    if mem_eff_save:
        mem_eff_save_file(state_dict, file_name, metadata=metadata)
    else:
        save_file(state_dict, file_name, metadata=metadata)


def merge_to_sd3_model(
    loading_device,
    working_device,
    sd3_path: str,
    clip_l_path: str,
    clip_g_path: str,
    t5xxl_path: str,
    models,
    ratios,
    merge_dtype,
    save_dtype,
    mem_eff_load_save=False,
):
    """
    將 LoKr 模型合併至 SD3 模型
    """
    # 建立模組映射，不載入狀態字典
    lora_name_to_module_key = {}
    if sd3_path is not None:
        logger.info(f"從 SD3 模型載入鍵值: {sd3_path}")
        with safe_open(sd3_path, framework="pt", device=loading_device) as sd3_file:
            keys = list(sd3_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_sd3.LoRANetwork.LORA_PREFIX_SD3 + "_" + module_name.replace(".", "_")
                    lora_name_to_module_key[lora_name] = key

    lora_name_to_clip_l_key = {}
    if clip_l_path is not None:
        logger.info(f"從 clip_l 模型載入鍵值: {clip_l_path}")
        with safe_open(clip_l_path, framework="pt", device=loading_device) as clip_l_file:
            keys = list(clip_l_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_sd3.LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP_L + "_" + module_name.replace(".", "_")
                    lora_name_to_clip_l_key[lora_name] = key

    lora_name_to_clip_g_key = {}
    if clip_g_path is not None:
        logger.info(f"從 clip_g 模型載入鍵值: {clip_g_path}")
        with safe_open(clip_g_path, framework="pt", device=loading_device) as clip_g_file:
            keys = list(clip_g_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_sd3.LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP_G + "_" + module_name.replace(".", "_")
                    lora_name_to_clip_g_key[lora_name] = key

    lora_name_to_t5xxl_key = {}
    if t5xxl_path is not None:
        logger.info(f"從 t5xxl 模型載入鍵值: {t5xxl_path}")
        with safe_open(t5xxl_path, framework="pt", device=loading_device) as t5xxl_file:
            keys = list(t5xxl_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_sd3.LoRANetwork.LORA_PREFIX_TEXT_ENCODER_T5 + "_" + module_name.replace(".", "_")
                    lora_name_to_t5xxl_key[lora_name] = key

    # 載入基礎模型
    sd3_state_dict = {}
    clip_l_state_dict = {}
    clip_g_state_dict = {}
    t5xxl_state_dict = {}

    if mem_eff_load_save:
        if sd3_path is not None:
            with MemoryEfficientSafeOpen(sd3_path) as sd3_file:
                for key in tqdm(sd3_file.keys()):
                    tensor = sd3_file.get_tensor(key).to(loading_device)
                    # 檢查並清理基礎模型的無效值
                    if torch.isfinite(tensor).all():
                        sd3_state_dict[key] = tensor
                    else:
                        logger.warning(f"Base model tensor {key} contains invalid values, cleaning up...")
                        sd3_state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if clip_l_path is not None:
            with MemoryEfficientSafeOpen(clip_l_path) as clip_l_file:
                for key in tqdm(clip_l_file.keys()):
                    tensor = clip_l_file.get_tensor(key).to(loading_device)
                    if torch.isfinite(tensor).all():
                        clip_l_state_dict[key] = tensor
                    else:
                        logger.warning(f"CLIP-L tensor {key} contains invalid values, cleaning up...")
                        clip_l_state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if clip_g_path is not None:
            with MemoryEfficientSafeOpen(clip_g_path) as clip_g_file:
                for key in tqdm(clip_g_file.keys()):
                    tensor = clip_g_file.get_tensor(key).to(loading_device)
                    if torch.isfinite(tensor).all():
                        clip_g_state_dict[key] = tensor
                    else:
                        logger.warning(f"CLIP-G tensor {key} contains invalid values, cleaning up...")
                        clip_g_state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if t5xxl_path is not None:
            with MemoryEfficientSafeOpen(t5xxl_path) as t5xxl_file:
                for key in tqdm(t5xxl_file.keys()):
                    tensor = t5xxl_file.get_tensor(key).to(loading_device)
                    if torch.isfinite(tensor).all():
                        t5xxl_state_dict[key] = tensor
                    else:
                        logger.warning(f"T5XXL tensor {key} contains invalid values, cleaning up...")
                        t5xxl_state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        if sd3_path is not None:
            sd3_state_dict = load_file(sd3_path, device=loading_device)
            # 檢查並清理基礎模型的無效值
            for key in list(sd3_state_dict.keys()):
                if not torch.isfinite(sd3_state_dict[key]).all():
                    logger.warning(f"Base model tensor {key} contains invalid values, cleaning up...")
                    sd3_state_dict[key] = torch.nan_to_num(sd3_state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

        if clip_l_path is not None:
            clip_l_state_dict = load_file(clip_l_path, device=loading_device)
            for key in list(clip_l_state_dict.keys()):
                if not torch.isfinite(clip_l_state_dict[key]).all():
                    logger.warning(f"CLIP-L tensor {key} contains invalid values, cleaning up...")
                    clip_l_state_dict[key] = torch.nan_to_num(clip_l_state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

        if clip_g_path is not None:
            clip_g_state_dict = load_file(clip_g_path, device=loading_device)
            for key in list(clip_g_state_dict.keys()):
                if not torch.isfinite(clip_g_state_dict[key]).all():
                    logger.warning(f"CLIP-G tensor {key} contains invalid values, cleaning up...")
                    clip_g_state_dict[key] = torch.nan_to_num(clip_g_state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

        if t5xxl_path is not None:
            t5xxl_state_dict = load_file(t5xxl_path, device=loading_device)
            for key in list(t5xxl_state_dict.keys()):
                if not torch.isfinite(t5xxl_state_dict[key]).all():
                    logger.warning(f"T5XXL tensor {key} contains invalid values, cleaning up...")
                    t5xxl_state_dict[key] = torch.nan_to_num(t5xxl_state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

    # 處理每個 LoKr 模型
    for model, ratio in zip(models, ratios):
        logger.info(f"載入: {model}")
        lokr_sd, _ = load_state_dict(model, merge_dtype)  # 在 CPU 上載入

        logger.info(f"合併中...")

        # 收集所有 LoKr 模組
        lokr_modules = {}
        for key in list(lokr_sd.keys()):
            if any(suffix in key for suffix in [".lokr_w1", ".lokr_w1_a", ".lokr_w2", ".lokr_w2_a", ".alpha"]):
                # 提取模組名稱
                if ".lokr_w1" in key:
                    module_name = key[:key.find(".lokr_w1")]
                elif ".lokr_w2" in key:
                    module_name = key[:key.find(".lokr_w2")]
                elif ".alpha" in key:
                    module_name = key[:key.find(".alpha")]

                if module_name not in lokr_modules:
                    lokr_modules[module_name] = {}

                weight_type = key[len(module_name) + 1:]
                lokr_modules[module_name][weight_type] = lokr_sd[key]

        # 合併每個 LoKr 模組
        for module_name, module_weights in tqdm(lokr_modules.items()):
            # 確定目標狀態字典
            if module_name in lora_name_to_module_key:
                module_weight_key = lora_name_to_module_key[module_name]
                state_dict = sd3_state_dict
            elif module_name in lora_name_to_clip_l_key:
                module_weight_key = lora_name_to_clip_l_key[module_name]
                state_dict = clip_l_state_dict
            elif module_name in lora_name_to_clip_g_key:
                module_weight_key = lora_name_to_clip_g_key[module_name]
                state_dict = clip_g_state_dict
            elif module_name in lora_name_to_t5xxl_key:
                module_weight_key = lora_name_to_t5xxl_key[module_name]
                state_dict = t5xxl_state_dict
            else:
                logger.warning(f"找不到 LoKr 權重對應的模組: {module_name}. 跳過...")
                continue

            # 提取 LoKr 權重
            w1 = module_weights.get("lokr_w1")
            w1a = module_weights.get("lokr_w1_a")
            w1b = module_weights.get("lokr_w1_b")
            w2 = module_weights.get("lokr_w2")
            w2a = module_weights.get("lokr_w2_a")
            w2b = module_weights.get("lokr_w2_b")
            t2 = module_weights.get("lokr_t2")
            alpha = module_weights.get("alpha", 1.0)

            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()

            # 確定秩
            rank = None
            if w1a is not None:
                rank = w1a.shape[1]
            elif w2a is not None:
                rank = w2a.shape[1]
            else:
                rank = alpha

            # 重建 LoKr 權重
            try:
                lokr_weight = rebuild_lokr_weight(w1, w1a, w1b, w2, w2a, w2b, t2, alpha, rank)

                # 檢查重建是否成功
                if lokr_weight is None:
                    logger.error(f"Failed to rebuild LoKr weight for {module_name}, skipping...")
                    continue

                # 獲取原始權重
                weight = state_dict[module_weight_key]

                # 檢查權重的數值有效性
                if not torch.isfinite(weight).all():
                    logger.warning(f"Original weight contains invalid values for {module_name}, cleaning up...")
                    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

                if not torch.isfinite(lokr_weight).all():
                    logger.warning(f"LoKr weight contains invalid values for {module_name}, cleaning up...")
                    lokr_weight = torch.nan_to_num(lokr_weight, nan=0.0, posinf=0.0, neginf=0.0)

                # 移動到工作設備
                weight = weight.to(working_device, merge_dtype)
                lokr_weight = lokr_weight.to(working_device, merge_dtype)

                # 額外的數值穩定性檢查：限制極端值
                weight_max = torch.max(torch.abs(weight))
                lokr_max = torch.max(torch.abs(lokr_weight))

                if weight_max > 1000:
                    logger.warning(f"Original weight has extreme values: {weight_max}, normalizing...")
                    weight = weight * (1000.0 / weight_max)

                if lokr_max > 1000:
                    logger.warning(f"LoKr weight has extreme values: {lokr_max}, normalizing...")
                    lokr_weight = lokr_weight * (1000.0 / lokr_max)

                # 應用合併
                if len(weight.size()) == 2:
                    # 線性層
                    weight = weight + ratio * lokr_weight
                elif len(weight.size()) == 4:
                    # 卷積層
                    weight = weight + ratio * lokr_weight
                else:
                    logger.warning(f"不支援的權重形狀: {weight.shape} 於 {module_name}")
                    continue

                # 立即進行數值清理和穩定性檢查
                if not torch.isfinite(weight).all():
                    logger.warning(f"Merged weight contains invalid values for {module_name}, cleaning up...")
                    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

                # 檢查合併後的極端值並限制
                merged_max = torch.max(torch.abs(weight))
                if merged_max > 10000:
                    logger.warning(f"Merged weight has extreme values: {merged_max}, clamping...")
                    weight = torch.clamp(weight, -10000, 10000)

                # 最終的數值有效性檢查
                if not torch.isfinite(weight).all():
                    logger.error(f"Final weight still contains invalid values for {module_name}, using zeros as fallback")
                    weight = torch.zeros_like(weight)

                # 保存合併後的權重
                state_dict[module_weight_key] = weight.to(loading_device, save_dtype)

                # 清理記憶體
                del lokr_weight
                del weight

            except Exception as e:
                logger.error(f"合併 {module_name} 時發生錯誤: {e}")
                continue

    return sd3_state_dict, clip_l_state_dict, clip_g_state_dict, t5xxl_state_dict


def merge_lokr_models(models, ratios, merge_dtype, concat=False, shuffle=False):
    """
    合併多個 LoKr 模型
    """
    base_alphas = {}  # 合併模型的 alpha
    base_dims = {}

    merged_sd = {}
    base_model = None

    for model, ratio in zip(models, ratios):
        logger.info(f"載入: {model}")
        lokr_sd, lokr_metadata = load_state_dict(model, merge_dtype)

        if lokr_metadata is not None:
            if base_model is None:
                base_model = lokr_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        # 獲取 alpha 和 dim
        alphas = {}  # 當前模型的 alpha
        dims = {}  # 當前模型的 dim

        for key in lokr_sd.keys():
            if "alpha" in key:
                lokr_module_name = key[:key.rfind(".alpha")]
                # 在 LoKr 中，.alpha 欄位實際儲存的是 dim 值，不是真正的 alpha
                dim_value = float(lokr_sd[key].detach().numpy())

                # 檢查 dim 值的有效性
                if not math.isfinite(dim_value):
                    if math.isnan(dim_value):
                        logger.warning(f"NaN dim value for {lokr_module_name}, using 1.0 as fallback")
                        dim_value = 1.0
                    elif math.isinf(dim_value):
                        if dim_value > 0:
                            logger.warning(f"Positive infinity dim value for {lokr_module_name}, clamping to 10000")
                            dim_value = 10000.0
                        else:
                            logger.warning(f"Negative infinity dim value for {lokr_module_name}, clamping to 1.0")
                            dim_value = 1.0
                    else:
                        logger.warning(f"Unknown invalid dim value for {lokr_module_name}: {dim_value}, using 1.0 as fallback")
                        dim_value = 1.0

                # 限制 dim 的範圍
                if dim_value > 10000:
                    logger.warning(f"Dim value {dim_value} is too large for {lokr_module_name}, clamping to 10000")
                    dim_value = 10000.0
                elif dim_value < 0.001:
                    logger.warning(f"Dim value {dim_value} is too small for {lokr_module_name}, clamping to 0.001")
                    dim_value = 0.001

                dims[lokr_module_name] = dim_value
                if lokr_module_name not in base_dims:
                    base_dims[lokr_module_name] = dim_value

                # 在 LoKr 中，alpha 應該被忽略，因為 LoKr 不使用 alpha
                # 但為了向後相容，我們將 alpha 設為 dim 值
                alphas[lokr_module_name] = dim_value
                if lokr_module_name not in base_alphas:
                    base_alphas[lokr_module_name] = dim_value

            elif "lokr_w1" in key or "lokr_w1_a" in key:
                if "lokr_w1_a" in key:
                    lokr_module_name = key[:key.rfind(".lokr_w1_a")]
                    # 在 LoKr 中，這實際上是 factor 值，不是 dim
                    factor = lokr_sd[key].size()[1]
                    # 如果 dims 中還沒有這個模組，使用 factor 作為 dim 的預設值
                    if lokr_module_name not in dims:
                        dims[lokr_module_name] = factor
                        if lokr_module_name not in base_dims:
                            base_dims[lokr_module_name] = factor
                else:
                    lokr_module_name = key[:key.rfind(".lokr_w1")]
                    # 這是完整的權重矩陣，第一個維度是輸出維度
                    out_dim = lokr_sd[key].size()[0]
                    if lokr_module_name not in dims:
                        dims[lokr_module_name] = out_dim
                        if lokr_module_name not in base_dims:
                            base_dims[lokr_module_name] = out_dim

        # 為沒有 dim 的模組設定預設值
        for lokr_module_name in dims.keys():
            if lokr_module_name not in alphas:
                # 在 LoKr 中，alpha 應該被忽略，但為了向後相容，使用 dim 值
                alpha = dims[lokr_module_name]
                alphas[lokr_module_name] = alpha
                if lokr_module_name not in base_alphas:
                    base_alphas[lokr_module_name] = alpha

        logger.info(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # 合併
        logger.info("合併中...")
        for key in tqdm(lokr_sd.keys()):
            if "alpha" in key:
                continue

            # 確定是否需要 concat
            if "lokr_w1" in key and concat:
                concat_dim = 1 if "lokr_w1_a" in key else 0
            elif "lokr_w2" in key and concat:
                concat_dim = 0 if "lokr_w2_a" in key else 1
            else:
                concat_dim = None

            # 提取模組名稱
            if ".lokr_w1" in key:
                lokr_module_name = key[:key.find(".lokr_w1")]
            elif ".lokr_w2" in key:
                lokr_module_name = key[:key.find(".lokr_w2")]
            else:
                continue

            base_alpha = base_alphas[lokr_module_name]
            alpha = alphas[lokr_module_name]

            # 額外檢查 alpha 值的有效性
            if not math.isfinite(base_alpha) or not math.isfinite(alpha):
                logger.error(f"Invalid alpha values for {lokr_module_name}: base_alpha={base_alpha}, alpha={alpha}")
                continue

            # 防止除零和極端值
            if base_alpha == 0:
                logger.warning(f"base_alpha is 0 for {lokr_module_name}, using 1.0 as fallback")
                base_alpha = 1.0

            # 在 LoKr 中，由於 alpha 實際上是 dim 值，我們應該直接使用 ratio
            # 不需要 sqrt(alpha / base_alpha) 的計算
            scale = ratio

            # 檢查 scale 的有效性
            if not math.isfinite(scale):
                logger.warning(f"Invalid scale value for {lokr_module_name}: {scale}, using 1.0 as fallback")
                scale = 1.0

            # 限制 scale 的範圍
            if scale > 1000:
                logger.warning(f"Scale value {scale} is too large for {lokr_module_name}, clamping to 1000")
                scale = 1000.0
            elif scale < 0.001:
                logger.warning(f"Scale value {scale} is too small for {lokr_module_name}, clamping to 0.001")
                scale = 0.001

            # 為上採樣權重使用絕對值
            if "lokr_w1" in key and "lokr_w1_a" not in key:
                scale = abs(scale)

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lokr_sd[key].size() or concat_dim is not None
                ), "權重大小不匹配，維度可能不同"
                if concat_dim is not None:
                    scaled_weight = lokr_sd[key] * scale
                    # 檢查縮放後權重的有效性
                    if not torch.isfinite(scaled_weight).all():
                        logger.warning(f"Scaled weight contains invalid values for {key}, cleaning up...")
                        scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                    merged_sd[key] = torch.cat([merged_sd[key], scaled_weight], dim=concat_dim)
                else:
                    scaled_weight = lokr_sd[key] * scale
                    # 檢查縮放後權重的有效性
                    if not torch.isfinite(scaled_weight).all():
                        logger.warning(f"Scaled weight contains invalid values for {key}, cleaning up...")
                        scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                    merged_sd[key] = merged_sd[key] + scaled_weight
            else:
                scaled_weight = lokr_sd[key] * scale
                # 檢查縮放後權重的有效性
                if not torch.isfinite(scaled_weight).all():
                    logger.warning(f"Scaled weight contains invalid values for {key}, cleaning up...")
                    scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                merged_sd[key] = scaled_weight

    # 設定 alpha 到狀態字典
    for lokr_module_name, alpha in base_alphas.items():
        key = lokr_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

        if shuffle:
            # 隨機排列權重
            key_w1 = lokr_module_name + ".lokr_w1"
            key_w1a = lokr_module_name + ".lokr_w1_a"
            key_w1b = lokr_module_name + ".lokr_w1_b"

            if key_w1 in merged_sd:
                dim = merged_sd[key_w1].shape[0]
                perm = torch.randperm(dim)
                merged_sd[key_w1] = merged_sd[key_w1][perm]
            elif key_w1a in merged_sd and key_w1b in merged_sd:
                dim = merged_sd[key_w1a].shape[0]
                perm = torch.randperm(dim)
                merged_sd[key_w1a] = merged_sd[key_w1a][perm]
                merged_sd[key_w1b] = merged_sd[key_w1b][:, perm]

    logger.info("已合併模型")
    logger.info(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    # 檢查所有 dim 是否相同
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = len(dims_list) == 1
    all_same_alphas = len(alphas_list) == 1

    # 建立最小元數據
    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"
    metadata = train_util.build_minimum_network_metadata(str(False), base_model, "lycoris.kohya", dims, alphas, None)

    return merged_sd, metadata


def merge(args):
    """
    主要合併函數
    """
    if args.models is None:
        args.models = []
    if args.ratios is None:
        args.ratios = []

    assert len(args.models) == len(
        args.ratios
    ), "模型數量必須等於比例數量"

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    assert (
        args.save_to or args.clip_l_save_to or args.clip_g_save_to or args.t5xxl_save_to
    ), "必須指定 save_to 或 clip_l_save_to 或 clip_g_save_to 或 t5xxl_save_to"

    dest_dir = os.path.dirname(args.save_to or args.clip_l_save_to or args.clip_g_save_to or args.t5xxl_save_to)
    if not os.path.exists(dest_dir):
        logger.info(f"建立目錄: {dest_dir}")
        os.makedirs(dest_dir)

    if args.sd3_model is not None or args.clip_l is not None or args.clip_g is not None or args.t5xxl is not None:
        # 合併到基礎模型
        assert (args.clip_l is None and args.clip_l_save_to is None) or (
            args.clip_l is not None and args.clip_l_save_to is not None
        ), "如果指定了 clip_l，也必須指定 clip_l_save_to"

        assert (args.clip_g is None and args.clip_g_save_to is None) or (
            args.clip_g is not None and args.clip_g_save_to is not None
        ), "如果指定了 clip_g，也必須指定 clip_g_save_to"

        assert (args.t5xxl is None and args.t5xxl_save_to is None) or (
            args.t5xxl is not None and args.t5xxl_save_to is not None
        ), "如果指定了 t5xxl，也必須指定 t5xxl_save_to"

        sd3_state_dict, clip_l_state_dict, clip_g_state_dict, t5xxl_state_dict = merge_to_sd3_model(
            args.loading_device,
            args.working_device,
            args.sd3_model,
            args.clip_l,
            args.clip_g,
            args.t5xxl,
            args.models,
            args.ratios,
            merge_dtype,
            save_dtype,
            args.mem_eff_load_save,
        )

        if args.no_metadata or (sd3_state_dict is None or len(sd3_state_dict) == 0):
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.sd3_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]

            if args.model_type == "large":
                model_config = {"sd3": "5-large"}
            elif args.model_type == "medium":
                model_config = {"sd3": "5-medium"}
            else:
                model_config = None
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, False, False, False, time.time(), title=title, merged_from=merged_from, model_config=model_config
            )

        if sd3_state_dict is not None and len(sd3_state_dict) > 0:
            logger.info(f"保存 SD3 模型至: {args.save_to}")
            save_to_file(args.save_to, sd3_state_dict, save_dtype, sai_metadata, args.mem_eff_load_save)

        if clip_l_state_dict is not None and len(clip_l_state_dict) > 0:
            logger.info(f"保存 clip_l 模型至: {args.clip_l_save_to}")
            save_to_file(args.clip_l_save_to, clip_l_state_dict, save_dtype, None, args.mem_eff_load_save)

        if clip_g_state_dict is not None and len(clip_g_state_dict) > 0:
            logger.info(f"保存 clip_g 模型至: {args.clip_g_save_to}")
            save_to_file(args.clip_g_save_to, clip_g_state_dict, save_dtype, None, args.mem_eff_load_save)

        if t5xxl_state_dict is not None and len(t5xxl_state_dict) > 0:
            logger.info(f"保存 t5xxl 模型至: {args.t5xxl_save_to}")
            save_to_file(args.t5xxl_save_to, t5xxl_state_dict, save_dtype, None, args.mem_eff_load_save)

    else:
        # 只合併 LoKr 模型
        sd3_state_dict, metadata = merge_lokr_models(args.models, args.ratios, merge_dtype, args.concat, args.shuffle)

        logger.info("計算雜湊值並建立元數據...")

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(sd3_state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if args.model_type == "large":
            model_config = {"sd3": "-5-large"}
        elif args.model_type == "medium":
            model_config = {"sd3": "-5-medium"}
        else:
            model_config = None

        if not args.no_metadata:
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                sd3_state_dict, False, False, False, True, False, time.time(), title=title, merged_from=merged_from, model_config=model_config
            )
            metadata.update(sai_metadata)

        logger.info(f"保存模型至: {args.save_to}")
        save_to_file(args.save_to, sd3_state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    """
    設定參數解析器
    """
    parser = argparse.ArgumentParser(description="合併 LoKr 模型到 SD3 模型")
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help="保存精度，如果省略則與合併精度相同。支援的類型: "
        "float32, fp16, bf16, fp8 (同 fp8_e4m3fn), fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        help="合併計算精度（建議使用 float）",
    )
    parser.add_argument(
        "--sd3_model",
        type=str,
        default=None,
        help="要載入的 SD3 模型，如果省略則合併 LoKr 模型",
    )
    parser.add_argument(
        "--clip_l",
        type=str,
        default=None,
        help="clip_l 的路徑（*.sft 或 *.safetensors），應該是 float16",
    )
    parser.add_argument(
        "--clip_g",
        type=str,
        default=None,
        help="clip_g 的路徑（*.sft 或 *.safetensors），應該是 float16",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        default=None,
        help="t5xxl 的路徑（*.sft 或 *.safetensors），應該是 float16",
    )
    parser.add_argument(
        "--mem_eff_load_save",
        action="store_true",
        help="對 SD3 模型使用自訂的記憶體高效載入和保存函數",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="載入 SD3 模型的設備。LoKr 模型在 CPU 上載入",
    )
    parser.add_argument(
        "--working_device",
        type=str,
        default="cpu",
        help="工作（合併）設備。LoKr 模型的合併在 CPU 上進行",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="目標檔案名稱：safetensors 檔案",
    )
    parser.add_argument(
        "--clip_l_save_to",
        type=str,
        default=None,
        help="clip_l 的目標檔案名稱：safetensors 檔案",
    )
    parser.add_argument(
        "--clip_g_save_to",
        type=str,
        default=None,
        help="clip_g 的目標檔案名稱：safetensors 檔案",
    )
    parser.add_argument(
        "--t5xxl_save_to",
        type=str,
        default=None,
        help="t5xxl 的目標檔案名稱：safetensors 檔案",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="large",
        help="SD3 模型類型：例如 large, medium，預設為 large",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="要合併的 LoKr 模型：safetensors 檔案",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="各 LoKr 模型的比例")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="不保存 sai modelspec 元數據（會保存 LoKr 的最小 ss_metadata）",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="串接 LoKr 而非合併（輸出 LoKr 的 dim(rank) 是輸入 dim 的總和）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="隨機排列 LoKr 權重",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    merge(args)
