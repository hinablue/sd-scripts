import argparse
import math
import os
import time
from typing import Any, Dict, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library.utils import setup_logging, str_to_dtype
from library.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

import lora_flux as lora_flux
from library import sai_model_spec, train_util


def safe_isfinite_check(tensor: torch.Tensor) -> bool:
    """
    安全地检查张量是否包含有限值，支持 FP8 数据类型

    Args:
        tensor: 要检查的张量

    Returns:
        bool: 如果所有值都是有限的则返回 True，否则返回 False
    """
    # 检查是否为 FP8 数据类型
    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]:
        # FP8 类型需要先转换为更高精度进行 isfinite 检查
        try:
            tensor_float32 = tensor.float()
            return torch.isfinite(tensor_float32).all()
        except Exception as e:
            logger.warning(f"Error checking FP8 tensor finiteness: {e}")
            # 如果转换失败，尝试使用数值范围检查
            try:
                # 检查是否在 FP8 的有效范围内
                if tensor.dtype == torch.float8_e4m3fn:
                    max_val = torch.finfo(torch.float8_e4m3fn).max
                    min_val = torch.finfo(torch.float8_e4m3fn).min
                elif tensor.dtype == torch.float8_e5m2:
                    max_val = torch.finfo(torch.float8_e5m2).max
                    min_val = torch.finfo(torch.float8_e5m2).min
                else:
                    # 对于其他 FP8 类型，使用保守的范围检查
                    return True  # 假设有效

                # 检查是否在有效范围内
                return torch.all((tensor >= min_val) & (tensor <= max_val))
            except Exception:
                # 如果所有检查都失败，假设张量是有效的
                logger.warning(f"Could not validate FP8 tensor, assuming valid")
                return True
    else:
        # 对于非 FP8 类型，直接使用 isfinite
        return torch.isfinite(tensor).all()


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
    if not safe_isfinite_check(w1) or not safe_isfinite_check(w2):
        logger.warning("Input weights contain invalid values in make_kron")
        w1 = torch.nan_to_num(w1, nan=0.0, posinf=0.0, neginf=0.0)
        w2 = torch.nan_to_num(w2, nan=0.0, posinf=0.0, neginf=0.0)

    # 檢查 scale 的有效性
    if not safe_isfinite_check(torch.tensor(scale)):
        logger.warning(f"Invalid scale value in make_kron: {scale}, using 1.0")
        scale = 1.0

    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()

    try:
        rebuild = torch.kron(w1, w2)

        # 檢查 Kronecker 乘積結果的有效性
        if not safe_isfinite_check(rebuild):
            logger.warning("Kronecker product contains invalid values, cleaning up...")
            rebuild = torch.nan_to_num(rebuild, nan=0.0, posinf=0.0, neginf=0.0)

        if scale != 1:
            rebuild = rebuild * scale

            # 檢查縮放後結果的有效性
            if not safe_isfinite_check(rebuild):
                logger.warning("Scaled result contains invalid values, cleaning up...")
                rebuild = torch.nan_to_num(rebuild, nan=0.0, posinf=0.0, neginf=0.0)

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
        alpha: 縮放係數（在 LoKr 中可能為 None）
        rank: 秩
    Returns:
        torch.Tensor: 重建的權重差異
    """
    # 在 LoKr 中，優先使用從權重矩陣中提取的 rank
    if w1a is not None:
        rank = w1a.shape[1]
    elif w2a is not None:
        rank = w2a.shape[1]
    elif w1 is not None:
        # 如果沒有分解的權重，使用 w1 的輸出維度
        rank = w1.shape[0]
    elif rank is None:
        # 如果都沒有，使用預設值
        rank = 1
        logger.warning("No rank information available, using default rank=1")

    # 防止除零錯誤和數值問題
    if rank <= 0:
        logger.warning(f"Invalid rank value: {rank}, using 1 as fallback")
        rank = 1

    # 在 LoKr 中，如果 alpha 為 None，則使用 factor-based 方法
    # 此時 scale 應該基於 rank 來計算，而不是 alpha
    if alpha is not None:
        # 檢查 alpha 的有效性
        if not safe_isfinite_check(torch.tensor(alpha)):
            logger.warning(f"Invalid alpha value: {alpha}, using rank-based scale")
            alpha = None

        if alpha is not None:
            # 在 LoKr 中，scale 計算應該是 alpha / rank
            scale = alpha / rank
        else:
            # 如果 alpha 無效，使用 rank-based scale
            scale = 1.0
    else:
        # 在 LoKr 中，當 alpha 為 None 時，使用 factor-based 方法
        # 此時 scale 應該基於 rank 來計算
        scale = 1.0
        logger.info(f"Using factor-based scale for LoKr (alpha=None, rank={rank})")

    # 檢查 scale 的有效性
    if not safe_isfinite_check(torch.tensor(scale)):
        logger.warning(f"Invalid scale value: {scale}, using 1.0 as fallback")
        scale = 1.0

    # 重建第一組權重
    if w1 is None:
        if w1a is not None and w1b is not None:
            # 檢查輸入權重的有效性
            if safe_isfinite_check(w1a) and safe_isfinite_check(w1b):
                w1 = w1a @ w1b
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
                if safe_isfinite_check(w2a) and safe_isfinite_check(w2b):
                    if w2b.dim() > 2:
                        # 處理卷積層
                        r, o, *k = w2b.shape
                        w2 = w2a @ w2b.view(r, -1)
                        w2 = w2.view(-1, o, *k)
                    else:
                        # 處理線性層
                        w2 = w2a @ w2b
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
                if safe_isfinite_check(t2) and safe_isfinite_check(w2a) and safe_isfinite_check(w2b):
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

    if not safe_isfinite_check(w1) or not safe_isfinite_check(w2):
        logger.warning("Rebuilt weights contain invalid values, cleaning up...")
        w1 = torch.nan_to_num(w1, nan=0.0, posinf=0.0, neginf=0.0)
        w2 = torch.nan_to_num(w2, nan=0.0, posinf=0.0, neginf=0.0)

    # 使用 Kronecker 乘積重建最終權重
    result = make_kron(w1, w2, scale)

    # 最終檢查結果的有效性
    if result is not None and not safe_isfinite_check(result):
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
            # 檢查張量的數值有效性
            if not safe_isfinite_check(sd[key]):
                logger.warning(f"Tensor {key} contains invalid values during loading, cleaning up...")
                sd[key] = torch.nan_to_num(sd[key], nan=0.0, posinf=0.0, neginf=0.0)

            # 進行類型轉換
            try:
                sd[key] = sd[key].to(dtype)
            except RuntimeWarning as e:
                if "invalid value encountered in cast" in str(e):
                    logger.warning(f"RuntimeWarning during loading dtype conversion for {key}: {e}")
                    # 再次清理無效值並重試
                    sd[key] = torch.nan_to_num(sd[key], nan=0.0, posinf=0.0, neginf=0.0)
                    sd[key] = sd[key].to(dtype)
                else:
                    raise

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
                if not safe_isfinite_check(state_dict[key]):
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


def merge_to_flux_model(
    loading_device,
    working_device,
    flux_path: str,
    clip_l_path: str,
    t5xxl_path: str,
    models,
    ratios,
    merge_dtype,
    save_dtype,
    mem_eff_load_save=False,
):
    """
    將 LoKr 模型合併至 FLUX 模型
    """
    # 建立模組映射，不載入狀態字典
    lora_name_to_module_key = {}
    if flux_path is not None:
        logger.info(f"從 FLUX.1 模型載入鍵值: {flux_path}")
        with safe_open(flux_path, framework="pt", device=loading_device) as flux_file:
            keys = list(flux_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + module_name.replace(".", "_")
                    lora_name_to_module_key[lora_name] = key

    lora_name_to_clip_l_key = {}
    if clip_l_path is not None:
        logger.info(f"從 clip_l 模型載入鍵值: {clip_l_path}")
        with safe_open(clip_l_path, framework="pt", device=loading_device) as clip_l_file:
            keys = list(clip_l_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP + "_" + module_name.replace(".", "_")
                    lora_name_to_clip_l_key[lora_name] = key

    lora_name_to_t5xxl_key = {}
    if t5xxl_path is not None:
        logger.info(f"從 t5xxl 模型載入鍵值: {t5xxl_path}")
        with safe_open(t5xxl_path, framework="pt", device=loading_device) as t5xxl_file:
            keys = list(t5xxl_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = lora_flux.LoRANetwork.LORA_PREFIX_TEXT_ENCODER_T5 + "_" + module_name.replace(".", "_")
                    lora_name_to_t5xxl_key[lora_name] = key

    # 載入基礎模型
    flux_state_dict = {}
    clip_l_state_dict = {}
    t5xxl_state_dict = {}

    if mem_eff_load_save:
        if flux_path is not None:
            with MemoryEfficientSafeOpen(flux_path) as flux_file:
                for key in tqdm(flux_file.keys()):
                    tensor = flux_file.get_tensor(key).to(loading_device)
                    # 檢查並清理基礎模型的無效值
                    if safe_isfinite_check(tensor):
                        flux_state_dict[key] = tensor
                    else:
                        logger.warning(f"Base model tensor {key} contains invalid values, cleaning up...")
                        flux_state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if clip_l_path is not None:
            with MemoryEfficientSafeOpen(clip_l_path) as clip_l_file:
                for key in tqdm(clip_l_file.keys()):
                    tensor = clip_l_file.get_tensor(key).to(loading_device)
                    if safe_isfinite_check(tensor):
                        clip_l_state_dict[key] = tensor
                    else:
                        logger.warning(f"CLIP-L tensor {key} contains invalid values, cleaning up...")
                        clip_l_state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if t5xxl_path is not None:
            with MemoryEfficientSafeOpen(t5xxl_path) as t5xxl_file:
                for key in tqdm(t5xxl_file.keys()):
                    tensor = t5xxl_file.get_tensor(key).to(loading_device)
                    if safe_isfinite_check(tensor):
                        t5xxl_state_dict[key] = tensor
                    else:
                        logger.warning(f"T5XXL tensor {key} contains invalid values, cleaning up...")
                        t5xxl_state_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        if flux_path is not None:
            flux_state_dict = load_file(flux_path, device=loading_device)
            # 檢查並清理基礎模型的無效值
            for key in list(flux_state_dict.keys()):
                if not safe_isfinite_check(flux_state_dict[key]):
                    logger.warning(f"Base model tensor {key} contains invalid values, cleaning up...")
                    flux_state_dict[key] = torch.nan_to_num(flux_state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

        if clip_l_path is not None:
            clip_l_state_dict = load_file(clip_l_path, device=loading_device)
            for key in list(clip_l_state_dict.keys()):
                if not safe_isfinite_check(clip_l_state_dict[key]):
                    logger.warning(f"CLIP-L tensor {key} contains invalid values, cleaning up...")
                    clip_l_state_dict[key] = torch.nan_to_num(clip_l_state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

        if t5xxl_path is not None:
            t5xxl_state_dict = load_file(t5xxl_path, device=loading_device)
            for key in list(t5xxl_state_dict.keys()):
                if not safe_isfinite_check(t5xxl_state_dict[key]):
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
                state_dict = flux_state_dict
            elif module_name in lora_name_to_clip_l_key:
                module_weight_key = lora_name_to_clip_l_key[module_name]
                state_dict = clip_l_state_dict
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
            alpha = module_weights.get("alpha")

            # 在 LoKr 中，alpha 可能為空或 None，此時不應該使用 alpha
            # 而是應該從 lokr_w1 的 factor 來確定維度
            if alpha is not None:
                if isinstance(alpha, torch.Tensor):
                    alpha = alpha.item()

                # 檢查 alpha 值的有效性
                if not math.isfinite(alpha):
                    logger.warning(f"Invalid alpha value for {module_name}: {alpha}, ignoring alpha")
                    alpha = None
                else:
                    # 限制 alpha 的範圍
                    if alpha > 10000:
                        logger.warning(f"Alpha value {alpha} is too large for {module_name}, clamping to 10000")
                        alpha = 10000.0
                    elif alpha < 0.001:
                        logger.warning(f"Alpha value {alpha} is too small for {module_name}, clamping to 0.001")
                        alpha = 0.001
            else:
                logger.info(f"No alpha value found for {module_name}, using factor-based approach")

            # 確定秩 - 在 LoKr 中優先使用 w1a 的 factor，然後是 w2a
            rank = None
            if w1a is not None:
                rank = w1a.shape[1]
            elif w2a is not None:
                rank = w2a.shape[1]
            elif w1 is not None:
                # 如果沒有分解的權重，使用 w1 的輸出維度
                rank = w1.shape[0]
            elif alpha is not None:
                # 只有在沒有其他選擇時才使用 alpha
                rank = alpha
            else:
                # 如果都沒有，使用預設值
                rank = 1
                logger.warning(f"No rank information found for {module_name}, using default rank=1")

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
                if not safe_isfinite_check(weight):
                    logger.warning(f"Original weight contains invalid values for {module_name}, cleaning up...")
                    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

                if not safe_isfinite_check(lokr_weight):
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
                if not safe_isfinite_check(weight):
                    logger.warning(f"Merged weight contains invalid values for {module_name}, cleaning up...")
                    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

                # 檢查合併後的極端值並限制
                merged_max = torch.max(torch.abs(weight))
                if merged_max > 10000:
                    logger.warning(f"Merged weight has extreme values: {merged_max}, clamping...")
                    weight = torch.clamp(weight, -10000, 10000)

                # 最終的數值有效性檢查
                if not safe_isfinite_check(weight):
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

    return flux_state_dict, clip_l_state_dict, t5xxl_state_dict


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
                # 在 LoKr 中，.alpha 欄位可能為空或無效
                try:
                    alpha_tensor = lokr_sd[key]
                    if alpha_tensor.numel() == 0:
                        # alpha 為空，在 LoKr 中這是正常的
                        logger.info(f"Empty alpha for {lokr_module_name}, using factor-based approach")
                        dims[lokr_module_name] = None
                        alphas[lokr_module_name] = None
                        if lokr_module_name not in base_dims:
                            base_dims[lokr_module_name] = None
                        if lokr_module_name not in base_alphas:
                            base_alphas[lokr_module_name] = None
                        continue

                    dim_value = float(alpha_tensor.detach().numpy())
                except Exception as e:
                    logger.warning(f"Error processing alpha for {lokr_module_name}: {e}, using factor-based approach")
                    dims[lokr_module_name] = None
                    alphas[lokr_module_name] = None
                    if lokr_module_name not in base_dims:
                        base_dims[lokr_module_name] = None
                    if lokr_module_name not in base_alphas:
                        base_alphas[lokr_module_name] = None
                    continue

                # 檢查 dim 值的有效性
                if not math.isfinite(dim_value):
                    if math.isnan(dim_value):
                        logger.warning(f"NaN dim value for {lokr_module_name}, using factor-based approach")
                        dims[lokr_module_name] = None
                        alphas[lokr_module_name] = None
                    elif math.isinf(dim_value):
                        if dim_value > 0:
                            logger.warning(f"Positive infinity dim value for {lokr_module_name}, clamping to 10000")
                            dim_value = 10000
                        else:
                            logger.warning(f"Negative infinity dim value for {lokr_module_name}, using factor-based approach")
                            dims[lokr_module_name] = None
                            alphas[lokr_module_name] = None
                    else:
                        logger.warning(f"Unknown invalid dim value for {lokr_module_name}: {dim_value}, using factor-based approach")
                        dims[lokr_module_name] = None
                        alphas[lokr_module_name] = None

                    if lokr_module_name not in base_dims:
                        base_dims[lokr_module_name] = None
                    if lokr_module_name not in base_alphas:
                        base_alphas[lokr_module_name] = None
                    continue

                # 限制 dim 的範圍
                if dim_value > 10000:
                    logger.warning(f"Dim value {dim_value} is too large for {lokr_module_name}, clamping to 10000")
                    dim_value = 10000
                elif dim_value < 0.001:
                    logger.warning(f"Dim value {dim_value} is too small for {lokr_module_name}, clamping to 1")
                    dim_value = 1

                dims[lokr_module_name] = dim_value
                if lokr_module_name not in base_dims:
                    base_dims[lokr_module_name] = dim_value

                # 在 LoKr 中，alpha 可能為空，此時應該使用 factor-based 方法
                alphas[lokr_module_name] = dim_value
                if lokr_module_name not in base_alphas:
                    base_alphas[lokr_module_name] = dim_value

            elif "lokr_w1" in key or "lokr_w1_a" in key:
                if "lokr_w1_a" in key:
                    lokr_module_name = key[:key.rfind(".lokr_w1_a")]
                    # 在 LoKr 中，這實際上是 factor 值，這是 LoKr 的核心參數
                    factor = lokr_sd[key].size()[1]
                    logger.info(f"Found LoKr factor for {lokr_module_name}: {factor}")
                    # 如果 dims 中還沒有這個模組，使用 factor 作為 dim 的預設值
                    if lokr_module_name not in dims:
                        dims[lokr_module_name] = factor
                        if lokr_module_name not in base_dims:
                            base_dims[lokr_module_name] = factor
                else:
                    lokr_module_name = key[:key.rfind(".lokr_w1")]
                    # 這是完整的權重矩陣，第一個維度是輸出維度
                    out_dim = lokr_sd[key].size()[0]
                    logger.info(f"Found LoKr w1 output dim for {lokr_module_name}: {out_dim}")
                    if lokr_module_name not in dims:
                        dims[lokr_module_name] = out_dim
                        if lokr_module_name not in base_dims:
                            base_dims[lokr_module_name] = out_dim

        # 為沒有 dim 的模組設定預設值
        for lokr_module_name in dims.keys():
            if lokr_module_name not in alphas:
                # 在 LoKr 中，如果 alpha 為 None，則使用 factor-based 方法
                if dims[lokr_module_name] is not None:
                    alpha = dims[lokr_module_name]
                else:
                    alpha = None
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
                concat_dim = 1 if "lokr_w2_a" in key else 0
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

            # 在 LoKr 中，alpha 可能為 None，此時使用 factor-based 方法
            if alpha is None or base_alpha is None:
                # 使用 factor-based 方法，直接使用 ratio
                scale = ratio
                logger.info(f"Using factor-based scale for {lokr_module_name}: {scale}")
            else:
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
                ), "factor 權重大小不匹配，維度可能不同"
                if concat_dim is not None:
                    scaled_weight = lokr_sd[key] * scale
                    # 檢查縮放後權重的有效性
                    if not safe_isfinite_check(scaled_weight):
                        logger.warning(f"Scaled weight contains invalid values for {key}, cleaning up...")
                        scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                    merged_sd[key] = torch.cat([merged_sd[key], scaled_weight], dim=concat_dim)
                else:
                    scaled_weight = lokr_sd[key] * scale
                    # 檢查縮放後權重的有效性
                    if not safe_isfinite_check(scaled_weight):
                        logger.warning(f"Scaled weight contains invalid values for {key}, cleaning up...")
                        scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                    merged_sd[key] = merged_sd[key] + scaled_weight
            else:
                scaled_weight = lokr_sd[key] * scale
                # 檢查縮放後權重的有效性
                if not safe_isfinite_check(scaled_weight):
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
        args.save_to or args.clip_l_save_to or args.t5xxl_save_to
    ), "必須指定 save_to 或 clip_l_save_to 或 t5xxl_save_to"

    dest_dir = os.path.dirname(args.save_to or args.clip_l_save_to or args.t5xxl_save_to)
    if not os.path.exists(dest_dir):
        logger.info(f"建立目錄: {dest_dir}")
        os.makedirs(dest_dir)

    if args.flux_model is not None or args.clip_l is not None or args.t5xxl is not None:
        # 合併到基礎模型
        assert (args.clip_l is None and args.clip_l_save_to is None) or (
            args.clip_l is not None and args.clip_l_save_to is not None
        ), "如果指定了 clip_l，也必須指定 clip_l_save_to"

        assert (args.t5xxl is None and args.t5xxl_save_to is None) or (
            args.t5xxl is not None and args.t5xxl_save_to is not None
        ), "如果指定了 t5xxl，也必須指定 t5xxl_save_to"

        flux_state_dict, clip_l_state_dict, t5xxl_state_dict = merge_to_flux_model(
            args.loading_device,
            args.working_device,
            args.flux_model,
            args.clip_l,
            args.t5xxl,
            args.models,
            args.ratios,
            merge_dtype,
            save_dtype,
            args.mem_eff_load_save,
        )

        if args.no_metadata or (flux_state_dict is None or len(flux_state_dict) == 0):
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.flux_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]

            if args.model_type == "dev":
                model_config = {"flux": "dev"}
            elif args.model_type == "schnell":
                model_config = {"flux": "schnell"}
            elif args.model_type == "chroma":
                model_config = {"flux": "chroma"}
            else:
                model_config = None
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, False, False, False, time.time(), title=title, merged_from=merged_from, model_config=model_config
            )

        if flux_state_dict is not None and len(flux_state_dict) > 0:
            logger.info(f"保存 FLUX 模型至: {args.save_to}")
            save_to_file(args.save_to, flux_state_dict, save_dtype, sai_metadata, args.mem_eff_load_save)

        if clip_l_state_dict is not None and len(clip_l_state_dict) > 0:
            logger.info(f"保存 clip_l 模型至: {args.clip_l_save_to}")
            save_to_file(args.clip_l_save_to, clip_l_state_dict, save_dtype, None, args.mem_eff_load_save)

        if t5xxl_state_dict is not None and len(t5xxl_state_dict) > 0:
            logger.info(f"保存 t5xxl 模型至: {args.t5xxl_save_to}")
            save_to_file(args.t5xxl_save_to, t5xxl_state_dict, save_dtype, None, args.mem_eff_load_save)

    else:
        # 只合併 LoKr 模型
        flux_state_dict, metadata = merge_lokr_models(args.models, args.ratios, merge_dtype, args.concat, args.shuffle)

        logger.info("計算雜湊值並建立元數據...")

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(flux_state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if not args.no_metadata:
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]

            if args.model_type == "dev":
                model_config = {"flux": "dev"}
            elif args.model_type == "schnell":
                model_config = {"flux": "schnell"}
            elif args.model_type == "chroma":
                model_config = {"flux": "chroma"}
            else:
                model_config = None
            sai_metadata = sai_model_spec.build_metadata(
                flux_state_dict, False, False, False, True, False, time.time(), title=title, merged_from=merged_from, model_config=model_config
            )
            metadata.update(sai_metadata)

        logger.info(f"保存模型至: {args.save_to}")
        save_to_file(args.save_to, flux_state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    """
    設定參數解析器
    """
    parser = argparse.ArgumentParser(description="合併 LoKr 模型到 FLUX 模型")
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
        "--flux_model",
        type=str,
        default=None,
        help="要載入的 FLUX.1 模型，如果省略則合併 LoKr 模型",
    )
    parser.add_argument(
        "--clip_l",
        type=str,
        default=None,
        help="clip_l 的路徑（*.sft 或 *.safetensors），應該是 float16",
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
        help="對 FLUX.1 模型使用自訂的記憶體高效載入和保存函數",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="載入 FLUX.1 模型的設備。LoKr 模型在 CPU 上載入",
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
        "--t5xxl_save_to",
        type=str,
        default=None,
        help="t5xxl 的目標檔案名稱：safetensors 檔案",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="dev",
        help="FLUX.1 模型類型：例如 dev, schnell, chroma，預設為 dev",
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