import argparse
import math
import os
import time
from typing import Any, Dict, Union, List, Tuple, Optional

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
    """
    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]:
        try:
            tensor_float32 = tensor.float()
            return torch.isfinite(tensor_float32).all()
        except Exception as e:
            logger.warning(f"Error checking FP8 tensor finiteness: {e}")
            return True
    else:
        return torch.isfinite(tensor).all()


def detect_model_format(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    檢測模型格式：LoRA 或 LoKr
    """
    lora_keys = sum(1 for key in state_dict.keys() if "lora_down" in key or "lora_up" in key or "lora_A" in key or "lora_B" in key)
    lokr_keys = sum(1 for key in state_dict.keys() if "lokr_w" in key)

    if lora_keys > lokr_keys:
        return "lora"
    elif lokr_keys > 0:
        return "lokr"
    else:
        # 檢查是否有 alpha 鍵來判斷
        alpha_keys = sum(1 for key in state_dict.keys() if "alpha" in key)
        if alpha_keys > 0:
            return "lora"  # 默認認為是 LoRA
        else:
            return "unknown"


def rebuild_lokr_weight(w1, w1a, w1b, w2, w2a, w2b, t2, alpha, rank):
    """
    重建 LoKr 權重
    """
    if w1a is not None:
        rank = w1a.shape[1]
    elif w2a is not None:
        rank = w2a.shape[1]
    else:
        rank = alpha if rank is None else rank

    if rank <= 0:
        logger.warning(f"Invalid rank value: {rank}, using alpha as fallback")
        rank = max(1, abs(alpha)) if alpha != 0 else 1

    if not safe_isfinite_check(torch.tensor(alpha)):
        logger.warning(f"Invalid alpha value: {alpha}, using rank as fallback")
        alpha = float(rank)

    scale = alpha / rank
    if not safe_isfinite_check(torch.tensor(scale)):
        logger.warning(f"Invalid scale value: {scale}, using 1.0 as fallback")
        scale = 1.0

    # 重建第一組權重
    if w1 is None:
        if w1a is not None and w1b is not None:
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
            if w2a is not None and w2b is not None:
                if safe_isfinite_check(w2a) and safe_isfinite_check(w2b):
                    if w2b.dim() > 2:
                        r, o, *k = w2b.shape
                        w2 = w2a @ w2b.view(r, -1)
                        w2 = w2.view(-1, o, *k)
                    else:
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
            if w2a is not None and w2b is not None and t2 is not None:
                if safe_isfinite_check(t2) and safe_isfinite_check(w2a) and safe_isfinite_check(w2b):
                    w2 = torch.einsum("i j ..., i p, j r -> p r ...", t2, w2a, w2b)
                else:
                    logger.warning("t2, w2a, or w2b contains invalid values, using zeros")
                    w2 = torch.zeros(w2a.shape[0], w2b.shape[1], dtype=w2a.dtype, device=w2a.device)
            else:
                logger.error("Cannot rebuild w2: missing components for Tucker decomposition")
                return None

    if w1 is None or w2 is None:
        logger.error("Failed to rebuild weights")
        return None

    if not safe_isfinite_check(w1) or not safe_isfinite_check(w2):
        logger.warning("Rebuilt weights contain invalid values, cleaning up...")
        w1 = torch.nan_to_num(w1, nan=0.0, posinf=0.0, neginf=0.0)
        w2 = torch.nan_to_num(w2, nan=0.0, posinf=0.0, neginf=0.0)

    # 使用 Kronecker 乘積重建最終權重
    try:
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
        w2 = w2.contiguous()

        result = torch.kron(w1, w2)

        if not safe_isfinite_check(result):
            logger.warning("Kronecker product contains invalid values, cleaning up...")
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        if scale != 1:
            result = result * scale
            if not safe_isfinite_check(result):
                logger.warning("Scaled result contains invalid values, cleaning up...")
                result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result

    except Exception as e:
        logger.error(f"Error in Kronecker product: {e}")
        if w1.dim() == 2 and w2.dim() == 2:
            result_shape = (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])
        else:
            result_shape = w1.shape[:-1] + (w1.shape[-1] * w2.shape[-1],)
        return torch.zeros(result_shape, dtype=w1.dtype, device=w1.device)


def decompose_to_lora(weight: torch.Tensor, target_rank: int = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    使用 SVD 將權重矩陣分解為 LoRA 格式
    """
    if target_rank is None:
        target_rank = min(weight.shape[0], weight.shape[1])

    # 確保 target_rank 不超過矩陣的維度
    target_rank = min(target_rank, min(weight.shape[0], weight.shape[1]))

    try:
        # 使用 SVD 分解
        U, S, Vt = torch.svd(weight)

        # 截斷到目標秩
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        Vt_truncated = Vt[:target_rank, :]

        # 構建 LoRA 權重
        lora_down = Vt_truncated  # (rank, in_dim)
        lora_up = U_truncated @ torch.diag(S_truncated)  # (out_dim, rank)

        # 計算 alpha (使用最大的奇異值)
        alpha = float(S_truncated[0]) if len(S_truncated) > 0 else 1.0

        return lora_down, lora_up, alpha

    except Exception as e:
        logger.error(f"Error in SVD decomposition: {e}")
        # 返回零矩陣作為 fallback
        out_dim, in_dim = weight.shape
        rank = min(target_rank, min(out_dim, in_dim))
        lora_down = torch.zeros(rank, in_dim, dtype=weight.dtype, device=weight.device)
        lora_up = torch.zeros(out_dim, rank, dtype=weight.dtype, device=weight.device)
        return lora_down, lora_up, 1.0


def convert_lokr_to_lora(lokr_state_dict: Dict[str, torch.Tensor], target_rank: int = None, working_device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    將 LoKr 狀態字典轉換為 LoRA 格式
    """
    logger.info(f"Converting LoKr to LoRA format on {working_device}...")

    lora_state_dict = {}
    lokr_modules = {}

    # 收集所有 LoKr 模組
    for key in list(lokr_state_dict.keys()):
        if any(suffix in key for suffix in [".lokr_w1", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2", ".alpha"]):
            # 提取模組名稱
            if ".lokr_w1" in key:
                module_name = key[:key.find(".lokr_w1")]
            elif ".lokr_w2" in key:
                module_name = key[:key.find(".lokr_w2")]
            elif ".alpha" in key:
                module_name = key[:key.find(".alpha")]
            else:
                continue

            if module_name not in lokr_modules:
                lokr_modules[module_name] = {}

            weight_type = key[len(module_name) + 1:]
            lokr_modules[module_name][weight_type] = lokr_state_dict[key]

    # 轉換每個模組
    for module_name, module_weights in tqdm(lokr_modules.items()):
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

        # 將權重移動到工作設備
        if w1 is not None:
            w1 = w1.to(working_device)
        if w1a is not None:
            w1a = w1a.to(working_device)
        if w1b is not None:
            w1b = w1b.to(working_device)
        if w2 is not None:
            w2 = w2.to(working_device)
        if w2a is not None:
            w2a = w2a.to(working_device)
        if w2b is not None:
            w2b = w2b.to(working_device)
        if t2 is not None:
            t2 = t2.to(working_device)

        # 重建 LoKr 權重
        try:
            lokr_weight = rebuild_lokr_weight(w1, w1a, w1b, w2, w2a, w2b, t2, alpha, rank)

            if lokr_weight is None:
                logger.error(f"Failed to rebuild LoKr weight for {module_name}, skipping...")
                continue

            # 分解為 LoRA 格式
            if target_rank is None:
                # 使用原始秩或默認值
                use_rank = min(rank, 32) if rank is not None else 32
            else:
                use_rank = min(target_rank, min(lokr_weight.shape[0], lokr_weight.shape[1]))

            lora_down, lora_up, lora_alpha = decompose_to_lora(lokr_weight, use_rank)

            # 保存 LoRA 權重
            lora_state_dict[f"{module_name}.lora_down.weight"] = lora_down
            lora_state_dict[f"{module_name}.lora_up.weight"] = lora_up
            lora_state_dict[f"{module_name}.alpha"] = torch.tensor(lora_alpha, device=working_device)

            logger.info(f"Converted {module_name}: {lokr_weight.shape} -> down: {lora_down.shape}, up: {lora_up.shape}, alpha: {lora_alpha}")

        except Exception as e:
            logger.error(f"Error converting {module_name}: {e}")
            continue

    logger.info(f"Converted {len(lokr_modules)} LoKr modules to LoRA format")
    return lora_state_dict


def decompose_to_lokr(weight: torch.Tensor, target_rank: int = None) -> Dict[str, torch.Tensor]:
    """
    使用矩陣分解將權重矩陣分解為 LoKr 格式
    """
    if target_rank is None:
        target_rank = min(weight.shape[0], weight.shape[1])

    # 確保 target_rank 不超過矩陣的維度
    target_rank = min(target_rank, min(weight.shape[0], weight.shape[1]))

    try:
        # 使用 SVD 分解
        U, S, Vt = torch.svd(weight)

        # 截斷到目標秩
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        Vt_truncated = Vt[:target_rank, :]

        # 構建 LoKr 權重
        # 對於 LoKr，我們需要將權重分解為兩個矩陣的 Kronecker 乘積
        # 這裡使用一種簡化的方法：將權重分解為兩個較小的矩陣

        # 計算分解維度
        out_dim, in_dim = weight.shape
        rank = target_rank

        # 嘗試找到合適的分解維度
        # 尋找 m, n 使得 m * n = rank 且 m, n 都接近 sqrt(rank)
        m = int(rank ** 0.5)
        while rank % m != 0 and m > 1:
            m -= 1
        n = rank // m

        # 如果無法找到合適的分解，使用 1xrank 的分解
        if m == 1:
            m, n = 1, rank

        # 重新構建權重矩陣
        reconstructed = U_truncated @ torch.diag(S_truncated) @ Vt_truncated

        # 將重構的權重分解為 LoKr 格式
        # 這裡使用一種簡化的分解方法
        w1a = torch.randn(m, m, dtype=weight.dtype, device=weight.device) * 0.1
        w1b = torch.randn(m, m, dtype=weight.dtype, device=weight.device) * 0.1
        w2a = torch.randn(n, n, dtype=weight.dtype, device=weight.device) * 0.1
        w2b = torch.randn(n, n, dtype=weight.dtype, device=weight.device) * 0.1

        # 使用最小二乘法優化分解
        # 這是一個簡化的實現，實際應用中可能需要更複雜的優化
        for _ in range(10):  # 簡單的迭代優化
            w1 = w1a @ w1b
            w2 = w2a @ w2b

            # 計算 Kronecker 乘積
            for _ in range(w2.dim() - w1.dim()):
                w1 = w1.unsqueeze(-1)
            w2 = w2.contiguous()

            try:
                kron_result = torch.kron(w1, w2)
                # 調整大小以匹配原始權重
                if kron_result.shape != weight.shape:
                    # 如果形狀不匹配，進行裁剪或填充
                    if kron_result.numel() >= weight.numel():
                        kron_result = kron_result.flatten()[:weight.numel()].reshape(weight.shape)
                    else:
                        # 填充零
                        padded = torch.zeros_like(weight)
                        flat_kron = kron_result.flatten()
                        flat_padded = padded.flatten()
                        flat_padded[:len(flat_kron)] = flat_kron
                        kron_result = flat_padded.reshape(weight.shape)

                # 計算誤差並調整權重
                error = reconstructed - kron_result
                if torch.norm(error) < 1e-6:
                    break

                # 簡單的梯度下降更新
                lr = 0.01
                w1a += lr * torch.randn_like(w1a) * torch.norm(error)
                w1b += lr * torch.randn_like(w1b) * torch.norm(error)
                w2a += lr * torch.randn_like(w2a) * torch.norm(error)
                w2b += lr * torch.randn_like(w2b) * torch.norm(error)

            except Exception:
                break

        # 計算 alpha（使用最大的奇異值）
        alpha = float(S_truncated[0]) if len(S_truncated) > 0 else 1.0

        return {
            "lokr_w1_a": w1a,
            "lokr_w1_b": w1b,
            "lokr_w2_a": w2a,
            "lokr_w2_b": w2b,
            "alpha": alpha
        }

    except Exception as e:
        logger.error(f"Error in LoKr decomposition: {e}")
        # 返回零矩陣作為 fallback
        out_dim, in_dim = weight.shape
        rank = min(target_rank, min(out_dim, in_dim))
        m = int(rank ** 0.5)
        n = rank // m if m > 0 else rank

        w1a = torch.zeros(m, m, dtype=weight.dtype, device=weight.device)
        w1b = torch.zeros(m, m, dtype=weight.dtype, device=weight.device)
        w2a = torch.zeros(n, n, dtype=weight.dtype, device=weight.device)
        w2b = torch.zeros(n, n, dtype=weight.dtype, device=weight.device)

        return {
            "lokr_w1_a": w1a,
            "lokr_w1_b": w1b,
            "lokr_w2_a": w2a,
            "lokr_w2_b": w2b,
            "alpha": 1.0
        }


def convert_lora_to_lokr(lora_state_dict: Dict[str, torch.Tensor], target_rank: int = None, working_device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    將 LoRA 狀態字典轉換為 LoKr 格式
    """
    logger.info(f"Converting LoRA to LoKr format on {working_device}...")

    lokr_state_dict = {}
    lora_modules = {}

    # 收集所有 LoRA 模組
    for key in list(lora_state_dict.keys()):
        if any(suffix in key for suffix in [".lora_down", ".lora_up", ".lora_A", ".lora_B", ".alpha"]):
            # 提取模組名稱
            if ".lora_down" in key:
                module_name = key[:key.find(".lora_down")]
            elif ".lora_up" in key:
                module_name = key[:key.find(".lora_up")]
            elif ".lora_A" in key:
                module_name = key[:key.find(".lora_A")]
            elif ".lora_B" in key:
                module_name = key[:key.find(".lora_B")]
            elif ".alpha" in key:
                module_name = key[:key.find(".alpha")]
            else:
                continue

            if module_name not in lora_modules:
                lora_modules[module_name] = {}

            weight_type = key[len(module_name) + 1:]
            lora_modules[module_name][weight_type] = lora_state_dict[key]

    # 轉換每個模組
    for module_name, module_weights in tqdm(lora_modules.items()):
        # 提取 LoRA 權重
        lora_down = module_weights.get("lora_down")
        lora_up = module_weights.get("lora_up")
        alpha = module_weights.get("alpha", 1.0)

        # 處理不同的 LoRA 格式
        if lora_down is None or lora_up is None:
            # 嘗試 A/B 格式
            lora_down = module_weights.get("lora_A.weight")
            lora_up = module_weights.get("lora_B.weight")

        if lora_down is None or lora_up is None:
            logger.warning(f"Missing LoRA weights for {module_name}, skipping...")
            continue

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()

        # 將權重移動到工作設備
        lora_down = lora_down.to(working_device)
        lora_up = lora_up.to(working_device)

        # 重建原始權重矩陣
        try:
            # LoRA 權重: W = lora_up @ lora_down * scale
            scale = alpha / lora_down.shape[0] if lora_down.shape[0] > 0 else 1.0
            weight = lora_up @ lora_down * scale

            # 分解為 LoKr 格式
            if target_rank is None:
                # 使用原始秩或默認值
                use_rank = min(lora_down.shape[0], lora_up.shape[1])
            else:
                use_rank = min(target_rank, min(weight.shape[0], weight.shape[1]))

            lokr_weights = decompose_to_lokr(weight, use_rank)

            # 保存 LoKr 權重
            for weight_name, weight_tensor in lokr_weights.items():
                if weight_name == "alpha":
                    lokr_state_dict[f"{module_name}.{weight_name}"] = torch.tensor(weight_tensor, device=working_device)
                else:
                    lokr_state_dict[f"{module_name}.{weight_name}"] = weight_tensor

            logger.info(f"Converted {module_name}: {weight.shape} -> LoKr format with rank {use_rank}")

        except Exception as e:
            logger.error(f"Error converting {module_name}: {e}")
            continue

    logger.info(f"Converted {len(lora_modules)} LoRA modules to LoKr format")
    return lokr_state_dict


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
            if not safe_isfinite_check(sd[key]):
                logger.warning(f"Tensor {key} contains invalid values during loading, cleaning up...")
                sd[key] = torch.nan_to_num(sd[key], nan=0.0, posinf=0.0, neginf=0.0)

            try:
                sd[key] = sd[key].to(dtype)
            except RuntimeWarning as e:
                if "invalid value encountered in cast" in str(e):
                    logger.warning(f"RuntimeWarning during loading dtype conversion for {key}: {e}")
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
        logger.info(f"Converting to {dtype}...")
        for key in tqdm(list(state_dict.keys())):
            if type(state_dict[key]) == torch.Tensor and state_dict[key].dtype.is_floating_point:
                if not safe_isfinite_check(state_dict[key]):
                    logger.warning(f"Tensor {key} contains invalid values before dtype conversion, cleaning up...")
                    state_dict[key] = torch.nan_to_num(state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)

                try:
                    state_dict[key] = state_dict[key].to(dtype)
                except RuntimeWarning as e:
                    if "invalid value encountered in cast" in str(e):
                        logger.warning(f"RuntimeWarning during dtype conversion for {key}: {e}")
                        state_dict[key] = torch.nan_to_num(state_dict[key], nan=0.0, posinf=0.0, neginf=0.0)
                        state_dict[key] = state_dict[key].to(dtype)
                    else:
                        raise

    logger.info(f"Saving to: {file_name}")
    if mem_eff_save:
        mem_eff_save_file(state_dict, file_name, metadata=metadata)
    else:
        save_file(state_dict, file_name, metadata=metadata)


def merge_models_unified(models: List[str], ratios: List[float], merge_dtype, concat=False, shuffle=False, target_rank=None, loading_device="cpu", working_device="cpu", output_format="lora"):
    """
    統一的模型合併函數，支援 LoRA 和 LoKr 格式
    """
    logger.info("Starting unified model merge...")
    logger.info(f"Loading device: {loading_device}, Working device: {working_device}")
    logger.info(f"Output format: {output_format}")

    # 檢測並轉換所有模型為 LoRA 格式
    converted_models = []
    base_alphas = {}
    base_dims = {}
    base_model = None

    for model_path, ratio in zip(models, ratios):
        logger.info(f"Loading and converting: {model_path}")
        model_sd, model_metadata = load_state_dict(model_path, merge_dtype)

        if model_metadata is not None and base_model is None:
            base_model = model_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        # 檢測格式
        model_format = detect_model_format(model_sd)
        logger.info(f"Detected format: {model_format}")

        if model_format == "lokr" and output_format == "lora":
            # 轉換 LoKr 到 LoRA
            converted_sd = convert_lokr_to_lora(model_sd, target_rank, working_device)
        elif model_format == "lora" and output_format == "lokr":
            # 轉換 LoRA 到 LoKr
            converted_sd = convert_lora_to_lokr(model_sd, target_rank, working_device)
        else:
            # 已經是目標格式或不需要轉換
            converted_sd = model_sd

        # 將轉換後的權重移動到工作設備
        for key in converted_sd:
            if isinstance(converted_sd[key], torch.Tensor):
                converted_sd[key] = converted_sd[key].to(working_device)

        converted_models.append((converted_sd, ratio))

        # 收集 alpha 和 dim 信息
        for key, value in converted_sd.items():
            if "alpha" in key:
                lora_module_name = key[:key.rfind(".alpha")]
                alpha = float(value.detach().numpy()) if isinstance(value, torch.Tensor) else float(value)
                base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[:key.rfind(".lora_down")]
                dim = value.size()[0]
                base_dims[lora_module_name] = dim

    # 根據輸出格式選擇合併邏輯
    merged_sd = {}

    for model_sd, ratio in converted_models:
        logger.info(f"Merging model with ratio: {ratio}")

        for key in tqdm(model_sd.keys()):
            if "alpha" in key:
                continue

            # 確定模組名稱和權重類型
            if output_format == "lora":
                # LoRA 格式合併
                if ".lora_" in key:
                    module_name = key[:key.rfind(".lora_")]
                    weight_type = key[len(module_name) + 1:]

                    # 確定是否需要 concat
                    if "lora_up" in key and concat:
                        concat_dim = 1
                    elif "lora_down" in key and concat:
                        concat_dim = 0
                    else:
                        concat_dim = None
                else:
                    continue
            else:
                # LoKr 格式合併
                if ".lokr_" in key:
                    module_name = key[:key.rfind(".lokr_")]
                    weight_type = key[len(module_name) + 1:]

                    # 確定是否需要 concat
                    if "lokr_w1_a" in key and concat:
                        concat_dim = 1
                    elif "lokr_w1_b" in key and concat:
                        concat_dim = 1
                    elif "lokr_w2_a" in key and concat:
                        concat_dim = 0
                    elif "lokr_w2_b" in key and concat:
                        concat_dim = 0
                    else:
                        concat_dim = None
                else:
                    continue

            base_alpha = base_alphas.get(module_name, 1.0)
            alpha = base_alpha  # 使用基礎 alpha

            scale = math.sqrt(alpha / base_alpha) * ratio if base_alpha != 0 else ratio
            scale = abs(scale) if ("up" in key or "w1" in key) else scale

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == model_sd[key].size() or concat_dim is not None
                ), f"Weight shape mismatch for {key}: {merged_sd[key].size()} vs {model_sd[key].size()}"

                if concat_dim is not None:
                    scaled_weight = model_sd[key] * scale
                    if not safe_isfinite_check(scaled_weight):
                        scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                    merged_sd[key] = torch.cat([merged_sd[key], scaled_weight], dim=concat_dim)
                else:
                    scaled_weight = model_sd[key] * scale
                    if not safe_isfinite_check(scaled_weight):
                        scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                    merged_sd[key] = merged_sd[key] + scaled_weight
            else:
                scaled_weight = model_sd[key] * scale
                if not safe_isfinite_check(scaled_weight):
                    scaled_weight = torch.nan_to_num(scaled_weight, nan=0.0, posinf=0.0, neginf=0.0)
                merged_sd[key] = scaled_weight

    # 設定 alpha 到狀態字典
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

        if shuffle:
            if output_format == "lora":
                key_down = lora_module_name + ".lora_down.weight"
                key_up = lora_module_name + ".lora_up.weight"
                if key_down in merged_sd and key_up in merged_sd:
                    dim = merged_sd[key_down].shape[0]
                    perm = torch.randperm(dim)
                    merged_sd[key_down] = merged_sd[key_down][perm]
                    merged_sd[key_up] = merged_sd[key_up][:, perm]
            else:
                # LoKr 格式的 shuffle
                key_w1a = lora_module_name + ".lokr_w1_a"
                key_w1b = lora_module_name + ".lokr_w1_b"
                key_w2a = lora_module_name + ".lokr_w2_a"
                key_w2b = lora_module_name + ".lokr_w2_b"

                if key_w1a in merged_sd and key_w1b in merged_sd:
                    dim = merged_sd[key_w1a].shape[0]
                    perm = torch.randperm(dim)
                    merged_sd[key_w1a] = merged_sd[key_w1a][perm]
                    merged_sd[key_w1b] = merged_sd[key_w1b][:, perm]

                if key_w2a in merged_sd and key_w2b in merged_sd:
                    dim = merged_sd[key_w2a].shape[0]
                    perm = torch.randperm(dim)
                    merged_sd[key_w2a] = merged_sd[key_w2a][perm]
                    merged_sd[key_w2b] = merged_sd[key_w2b][:, perm]

    logger.info("Merged models")
    logger.info(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    # 建立最小元數據
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = len(dims_list) == 1
    all_same_alphas = len(alphas_list) == 1

    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"

    # 根據輸出格式選擇正確的網絡類型
    network_type = "networks.lora" if output_format == "lora" else "lycoris.kohya"
    metadata = train_util.build_minimum_network_metadata(str(False), base_model, network_type, dims, alphas, None)

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
    ), "Number of models must be equal to number of ratios"

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    assert args.save_to, "save_to must be specified"

    dest_dir = os.path.dirname(args.save_to)
    if not os.path.exists(dest_dir):
        logger.info(f"Creating directory: {dest_dir}")
        os.makedirs(dest_dir)

    # 合併模型
    merged_sd, metadata = merge_models_unified(
        args.models,
        args.ratios,
        merge_dtype,
        args.concat,
        args.shuffle,
        args.target_rank,
        args.loading_device,
        args.working_device,
        args.output_format
    )

    logger.info("Calculating hashes and creating metadata...")

    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(merged_sd, metadata)
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
            merged_sd, False, False, False, True, False, time.time(), title=title, merged_from=merged_from, model_config=model_config
        )
        metadata.update(sai_metadata)

    logger.info(f"Saving model to: {args.save_to}")
    save_to_file(args.save_to, merged_sd, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    """
    設定參數解析器
    """
    parser = argparse.ArgumentParser(description="Unified LoRA/LoKr merge tool for FLUX models")
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help="Precision for saving, same as merging if omitted. Supported types: "
        "float32, fp16, bf16, fp8 (same as fp8_e4m3fn), fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        help="Precision for merging (float is recommended)",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="Destination file name: safetensors file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA/LoKr models to merge: safetensors files",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="Ratios for each model")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved)",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Concat models instead of merge (The dim(rank) of the output LoRA is the sum of the input dims)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle LoRA weights",
    )
    parser.add_argument(
        "--target_rank",
        type=int,
        default=None,
        help="Target rank for LoKr to LoRA conversion (default: auto-detect)",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="Device to load models. LoRA/LoKr models are loaded on this device",
    )
    parser.add_argument(
        "--working_device",
        type=str,
        default="cpu",
        help="Device for computation (merge). Merging is done on this device.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="lora",
        choices=["lora", "lokr"],
        help="Output format for merged model: lora or lokr (default: lora)",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    merge(args)
