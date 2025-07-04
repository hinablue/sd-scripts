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

import lora_flux as lora_flux
from library import sai_model_spec, train_util


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
    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    if scale != 1:
        rebuild = rebuild * scale

    return rebuild


def rebuild_lokr_weight(w1, w1a, w1b, w2, w2a, w2b, t2, alpha, rank):
    """
    重建 LoKr 權重
    Args:
        w1, w1a, w1b: 第一組權重矩陣
        w2, w2a, w2b: 第二組權重矩陣
        t2: Tucker 分解矩陣（可選）
        alpha: 縮放係數
        rank: 秩
    Returns:
        torch.Tensor: 重建的權重差異
    """
    if w1a is not None:
        rank = w1a.shape[1]
    elif w2a is not None:
        rank = w2a.shape[1]
    else:
        rank = alpha if rank is None else rank

    scale = alpha / rank

    # 重建第一組權重
    if w1 is None:
        w1 = w1a @ w1b

    # 重建第二組權重
    if w2 is None:
        if t2 is None:
            # 標準 LoKr 分解
            if w2b.dim() > 2:
                # 處理卷積層
                r, o, *k = w2b.shape
                w2 = w2a @ w2b.view(r, -1)
                w2 = w2.view(-1, o, *k)
            else:
                # 處理線性層
                w2 = w2a @ w2b
        else:
            # Tucker 分解
            w2 = rebuild_tucker(t2, w2a, w2b)

    # 使用 Kronecker 乘積重建最終權重
    return make_kron(w1, w2, scale)


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
                state_dict[key] = state_dict[key].to(dtype)

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
                    flux_state_dict[key] = flux_file.get_tensor(key).to(loading_device)

        if clip_l_path is not None:
            with MemoryEfficientSafeOpen(clip_l_path) as clip_l_file:
                for key in tqdm(clip_l_file.keys()):
                    clip_l_state_dict[key] = clip_l_file.get_tensor(key).to(loading_device)

        if t5xxl_path is not None:
            with MemoryEfficientSafeOpen(t5xxl_path) as t5xxl_file:
                for key in tqdm(t5xxl_file.keys()):
                    t5xxl_state_dict[key] = t5xxl_file.get_tensor(key).to(loading_device)
    else:
        if flux_path is not None:
            flux_state_dict = load_file(flux_path, device=loading_device)
        if clip_l_path is not None:
            clip_l_state_dict = load_file(clip_l_path, device=loading_device)
        if t5xxl_path is not None:
            t5xxl_state_dict = load_file(t5xxl_path, device=loading_device)

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

                # 獲取原始權重
                weight = state_dict[module_weight_key]

                # 移動到工作設備
                weight = weight.to(working_device, merge_dtype)
                lokr_weight = lokr_weight.to(working_device, merge_dtype)

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
                alpha = float(lokr_sd[key].detach().numpy())
                alphas[lokr_module_name] = alpha
                if lokr_module_name not in base_alphas:
                    base_alphas[lokr_module_name] = alpha
            elif "lokr_w1" in key or "lokr_w1_a" in key:
                if "lokr_w1_a" in key:
                    lokr_module_name = key[:key.rfind(".lokr_w1_a")]
                    dim = lokr_sd[key].size()[1]  # lokr_w1_a 的第二個維度是 rank
                else:
                    lokr_module_name = key[:key.rfind(".lokr_w1")]
                    dim = lokr_sd[key].size()[0]  # lokr_w1 的第一個維度
                dims[lokr_module_name] = dim
                if lokr_module_name not in base_dims:
                    base_dims[lokr_module_name] = dim

        # 為沒有 alpha 的模組設定預設值
        for lokr_module_name in dims.keys():
            if lokr_module_name not in alphas:
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

            scale = math.sqrt(alpha / base_alpha) * ratio
            # 為上採樣權重使用絕對值
            if "lokr_w1" in key and "lokr_w1_a" not in key:
                scale = abs(scale)

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lokr_sd[key].size() or concat_dim is not None
                ), "權重大小不匹配，維度可能不同"
                if concat_dim is not None:
                    merged_sd[key] = torch.cat([merged_sd[key], lokr_sd[key] * scale], dim=concat_dim)
                else:
                    merged_sd[key] = merged_sd[key] + lokr_sd[key] * scale
            else:
                merged_sd[key] = lokr_sd[key] * scale

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
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, False, False, False, time.time(), title=title, merged_from=merged_from, flux="dev"
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
            sai_metadata = sai_model_spec.build_metadata(
                flux_state_dict, False, False, False, True, False, time.time(), title=title, merged_from=merged_from, flux="dev"
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