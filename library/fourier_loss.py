#!/usr/bin/env python3
"""
傅立葉損失函數模組

此模組提供了各種傅立葉域損失函數，用於增強深度學習模型的頻域表示能力。
主要包含以下功能：
- 基礎傅立葉損失
- 頻率加權傅立葉損失
- 多尺度傅立葉損失
- 自適應傅立葉損失
- 組合損失函數
- 設定和實用工具
"""

import torch
import logging
from typing import Optional, List, Dict, Any, Tuple
import json

def create_frequency_weight_mask(height: int, width: int, high_freq_weight: float = 2.0,
                                device: str = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    創建頻率權重遮罩，對高頻成分給予更高權重

    Args:
        height: 高度維度
        width: 寬度維度
        high_freq_weight: 高頻權重倍數
        device: 張量設備
        dtype: 張量數據類型

    Returns:
        頻率權重遮罩張量
    """
    # 限制權重範圍，防止過度放大
    high_freq_weight = max(1.0, min(high_freq_weight, 3.0))

    # 創建頻率座標
    freq_h = torch.fft.fftfreq(height, device=device, dtype=dtype)
    freq_w = torch.fft.fftfreq(width, device=device, dtype=dtype)

    # 創建 2D 頻率網格
    freq_h_grid, freq_w_grid = torch.meshgrid(freq_h, freq_w, indexing='ij')

    # 計算頻率幅度
    freq_magnitude = torch.sqrt(freq_h_grid**2 + freq_w_grid**2)

    # 正規化到 [0, 1] 範圍，添加平滑化
    max_freq = freq_magnitude.max()
    if max_freq > 0:
        freq_magnitude = freq_magnitude / max_freq
    else:
        freq_magnitude = torch.zeros_like(freq_magnitude)

    # 使用更平滑的權重分配 (sigmoid 函數而非線性)
    # 這可以減少極端權重值
    sigmoid_factor = 4.0  # 控制過渡的陡峭程度
    freq_sigmoid = torch.sigmoid(sigmoid_factor * (freq_magnitude - 0.5))

    # 創建權重：低頻為 1.0，高頻逐漸增加到 high_freq_weight
    weight_mask = 1.0 + (high_freq_weight - 1.0) * freq_sigmoid

    return weight_mask


def compute_fourier_magnitude_spectrum(tensor: torch.Tensor, dims: tuple = (-2, -1),
                                     eps: float = 1e-8, normalize: bool = True) -> torch.Tensor:
    """
    計算張量的傅立葉幅度譜

    Args:
        tensor: 輸入張量
        dims: 進行 FFT 的維度
        eps: 數值穩定性常數
        normalize: 是否對幅度譜進行正規化

    Returns:
        傅立葉幅度譜
    """
    # 計算多維 FFT
    fft_result = torch.fft.fftn(tensor, dim=dims)

    # 計算幅度譜並添加數值穩定性
    magnitude = torch.abs(fft_result) + eps

    if normalize:
        # 正規化：除以張量大小的平方根和最大值
        tensor_numel = 1
        for dim in dims:
            tensor_numel *= tensor.shape[dim]

        # 按張量大小正規化（類似於 FFTW 的正規化）
        magnitude = magnitude / (tensor_numel ** 0.5)

        # 進一步按輸入張量的數值範圍正規化
        input_scale = torch.std(tensor) + eps
        magnitude = magnitude / input_scale

    return magnitude


def fourier_latent_loss_basic(model_pred: torch.Tensor, target: torch.Tensor,
                             norm_type: str = "l2", dims: tuple = (-2, -1),
                             eps: float = 1e-8) -> torch.Tensor:
    """
    基礎傅立葉 Latent 損失計算

    Args:
        model_pred: 模型預測 (z_SR)
        target: 目標 (z_HR)
        norm_type: 損失範數類型 ("l1" 或 "l2")
        dims: FFT 計算維度
        eps: 數值穩定性常數

    Returns:
        傅立葉特徵損失
    """
    # 計算傅立葉幅度譜 (已正規化)
    mag_pred = compute_fourier_magnitude_spectrum(model_pred, dims, eps, normalize=True)
    mag_target = compute_fourier_magnitude_spectrum(target, dims, eps, normalize=True)

    # 計算損失
    if norm_type == "l1":
        loss = torch.mean(torch.abs(mag_target - mag_pred))
    elif norm_type == "l2":
        loss = torch.mean((mag_target - mag_pred) ** 2)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    # 進一步約束損失範圍，防止異常值
    loss = torch.clamp(loss, max=10.0)

    return loss


def fourier_latent_loss_weighted(model_pred: torch.Tensor, target: torch.Tensor,
                                high_freq_weight: float = 2.0, dims: tuple = (-2, -1),
                                norm_type: str = "l2", eps: float = 1e-8) -> torch.Tensor:
    """
    頻率加權傅立葉 Latent 損失

    Args:
        model_pred: 模型預測 (z_SR)
        target: 目標 (z_HR)
        high_freq_weight: 高頻成分權重倍數
        dims: FFT 計算維度
        norm_type: 損失範數類型
        eps: 數值穩定性常數

    Returns:
        加權傅立葉特徵損失
    """
    # 限制高頻權重範圍，防止過度放大
    high_freq_weight = torch.clamp(torch.tensor(high_freq_weight), min=1.0, max=3.0).item()

    # 計算傅立葉幅度譜 (已正規化)
    mag_pred = compute_fourier_magnitude_spectrum(model_pred, dims, eps, normalize=True)
    mag_target = compute_fourier_magnitude_spectrum(target, dims, eps, normalize=True)

    # 創建頻率權重遮罩
    height, width = model_pred.shape[dims[0]], model_pred.shape[dims[1]]
    weight_mask = create_frequency_weight_mask(
        height, width, high_freq_weight,
        device=model_pred.device, dtype=model_pred.dtype
    )

    # 擴展權重遮罩以匹配張量形狀
    while weight_mask.dim() < mag_pred.dim():
        weight_mask = weight_mask.unsqueeze(0)

    # 計算加權差異
    diff = mag_target - mag_pred
    if norm_type == "l1":
        weighted_diff = torch.abs(diff) * weight_mask
    elif norm_type == "l2":
        weighted_diff = (diff ** 2) * weight_mask
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    # 計算加權平均損失
    loss = torch.mean(weighted_diff)

    # 進一步約束損失範圍，防止異常值
    loss = torch.clamp(loss, max=10.0)

    return loss


def fourier_latent_loss_multiscale(model_pred: torch.Tensor, target: torch.Tensor,
                                  scales: List[int] = None, scale_weights: List[float] = None,
                                  dims: tuple = (-2, -1), norm_type: str = "l2",
                                  eps: float = 1e-8) -> torch.Tensor:
    """
    多尺度傅立葉 Latent 損失

    Args:
        model_pred: 模型預測 (z_SR)
        target: 目標 (z_HR)
        scales: 多個縮放尺度
        scale_weights: 各尺度權重（如果為 None 則自動計算）
        dims: FFT 計算維度
        norm_type: 損失範數類型
        eps: 數值穩定性常數

    Returns:
        多尺度傅立葉特徵損失
    """
    if scales is None:
        scales = [1, 2]

    if scale_weights is None:
        scale_weights = [1.0 / scale for scale in scales]

    if len(scale_weights) != len(scales):
        raise ValueError("scale_weights length must match scales length")

    total_loss = 0.0
    total_weight = 0.0

    for scale, weight in zip(scales, scale_weights):
        if scale == 1:
            pred_scaled = model_pred
            target_scaled = target
        else:
            # 檢查張量維度是否足夠進行池化
            if (model_pred.dim() >= 4 and
                model_pred.shape[-1] >= scale and model_pred.shape[-2] >= scale):
                # 使用平均池化進行下採樣
                pred_scaled = torch.nn.functional.avg_pool2d(model_pred, scale)
                target_scaled = torch.nn.functional.avg_pool2d(target, scale)
            else:
                # 跳過無效尺度
                continue

        # 計算該尺度的傅立葉損失
        scale_loss = fourier_latent_loss_basic(pred_scaled, target_scaled, norm_type, dims, eps)

        total_loss += weight * scale_loss
        total_weight += weight

    # 防止除零錯誤
    if total_weight == 0:
        return torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)

    # 約束最終損失值
    final_loss = total_loss / total_weight
    final_loss = torch.clamp(final_loss, max=10.0)

    return final_loss


def fourier_latent_loss_adaptive(model_pred: torch.Tensor, target: torch.Tensor,
                                current_step: int, total_steps: int,
                                max_weight: float = 2.0, min_weight: float = 0.5,
                                dims: tuple = (-2, -1), norm_type: str = "l2",
                                eps: float = 1e-8) -> torch.Tensor:
    """
    自適應傅立葉 Latent 損失（權重隨訓練進度調整）

    Args:
        model_pred: 模型預測 (z_SR)
        target: 目標 (z_HR)
        current_step: 當前訓練步數
        total_steps: 總訓練步數
        max_weight: 最大高頻權重
        min_weight: 最小高頻權重
        dims: FFT 計算維度
        norm_type: 損失範數類型
        eps: 數值穩定性常數

    Returns:
        自適應傅立葉特徵損失
    """
    # 限制權重範圍
    max_weight = max(1.0, min(max_weight, 3.0))
    min_weight = max(0.5, min(min_weight, max_weight))

    # 計算訓練進度 (0.0 到 1.0)
    progress = min(current_step / max(total_steps, 1), 1.0)

    # 早期訓練重視高頻，後期逐漸平衡
    high_freq_weight = max_weight - (max_weight - min_weight) * progress

    return fourier_latent_loss_weighted(
        model_pred, target, high_freq_weight, dims, norm_type, eps
    )


def conditional_loss_with_fourier(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
    current_step: int = 0,
    total_steps: int = 1000,
    # 傅立葉特徵損失參數
    fourier_weight: float = 0.05,
    fourier_mode: str = "weighted",  # "basic", "weighted", "multiscale", "adaptive"
    fourier_norm: str = "l2",
    fourier_dims: tuple = (-2, -1),
    fourier_high_freq_weight: float = 2.0,
    fourier_scales: List[int] = None,
    fourier_scale_weights: List[float] = None,
    fourier_adaptive_max_weight: float = 2.0,
    fourier_adaptive_min_weight: float = 0.5,
    fourier_eps: float = 1e-8,
    fourier_warmup_steps: int = 200
) -> torch.Tensor:
    """
    增強版 conditional_loss，支援傅立葉特徵損失

    Args:
        model_pred: 模型預測張量
        target: 目標張量
        loss_type: 基礎損失類型 ("fourier")
        reduction: 損失約簡方式 ("mean", "sum", "none")
        huber_c: Huber 損失參數
        current_step: 當前訓練步數
        total_steps: 總訓練步數

        # 傅立葉特徵損失參數
        fourier_weight: 傅立葉損失權重
        fourier_mode: 傅立葉損失模式
        fourier_norm: 傅立葉損失範數 ("l1" 或 "l2")
        fourier_dims: FFT 計算維度
        fourier_high_freq_weight: 高頻權重倍數 (weighted 模式)
        fourier_scales: 多尺度列表 (multiscale 模式)
        fourier_scale_weights: 尺度權重列表 (multiscale 模式)
        fourier_adaptive_max_weight: 自適應最大權重 (adaptive 模式)
        fourier_adaptive_min_weight: 自適應最小權重 (adaptive 模式)
        fourier_eps: 數值穩定性常數
        fourier_warmup_steps: 傅立葉損失預熱步數

    Returns:
        組合損失值
    """

    # 計算基礎損失
    if fourier_norm == "l1":
        base_loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    else:
        base_loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)

    # 如果不是 fourier 損失或權重為 0，直接返回基礎損失
    if loss_type != "fourier" or fourier_weight <= 0.0:
        return base_loss

    # 如果權重為 0，直接返回基礎損失
    if fourier_weight <= 0.0:
        return base_loss

    # 如果在預熱期內，直接返回基礎損失
    if current_step < fourier_warmup_steps:
        return base_loss

    # 檢查張量維度是否足夠
    if model_pred.dim() < 3 or target.dim() < 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"傅立葉損失需要至少 3D 張量，收到 {model_pred.dim()}D 和 {target.dim()}D 張量，跳過傅立葉損失計算"
        )
        return base_loss

    # 確保張量形狀一致
    if model_pred.shape != target.shape:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"model_pred 和 target 形狀不匹配：{model_pred.shape} vs {target.shape}，跳過傅立葉損失計算"
        )
        return base_loss

    try:
        # 根據模式計算傅立葉損失
        if fourier_mode == "basic":
            fourier_loss = fourier_latent_loss_basic(
                model_pred, target, fourier_norm, fourier_dims, fourier_eps
            )
        elif fourier_mode == "weighted":
            fourier_loss = fourier_latent_loss_weighted(
                model_pred, target, fourier_high_freq_weight, fourier_dims,
                fourier_norm, fourier_eps
            )
        elif fourier_mode == "multiscale":
            if fourier_scales is None:
                fourier_scales = [1, 2]
            fourier_loss = fourier_latent_loss_multiscale(
                model_pred, target, fourier_scales, fourier_scale_weights,
                fourier_dims, fourier_norm, fourier_eps
            )
        elif fourier_mode == "adaptive":
            fourier_loss = fourier_latent_loss_adaptive(
                model_pred, target, current_step, total_steps,
                fourier_adaptive_max_weight, fourier_adaptive_min_weight,
                fourier_dims, fourier_norm, fourier_eps
            )
        elif fourier_mode == "unified":
            # 使用整合模式，支持額外參數
            unified_kwargs = {}
            if fourier_scales is not None:
                unified_kwargs['scales'] = fourier_scales
            if fourier_scale_weights is not None:
                unified_kwargs['scale_weights'] = fourier_scale_weights

            fourier_loss = fourier_latent_loss_unified(
                model_pred, target,
                dims=fourier_dims,
                norm_type=fourier_norm,
                eps=fourier_eps,
                high_freq_weight=fourier_high_freq_weight,
                current_step=current_step,
                total_steps=total_steps,
                max_weight=fourier_adaptive_max_weight,
                min_weight=fourier_adaptive_min_weight,
                **unified_kwargs
            )
        elif fourier_mode in ["unified_basic", "unified_balanced", "unified_detail", "unified_adaptive"]:
            # 使用簡化版整合模式
            mode_map = {
                "unified_basic": "basic",
                "unified_balanced": "balanced",
                "unified_detail": "detail",
                "unified_adaptive": "adaptive"
            }
            fourier_loss = fourier_latent_loss_unified_simple(
                model_pred, target,
                mode=mode_map[fourier_mode],
                current_step=current_step,
                total_steps=total_steps
            )
        else:
            raise ValueError(f"Unsupported fourier_mode: {fourier_mode}")

        # 動態調整傅立葉損失權重，避免與基礎損失差距過大
        # 安全地將損失轉換為標量值進行比較
        try:
            # 確保基礎損失是標量
            if base_loss.numel() > 1:
                base_loss_magnitude = base_loss.detach().mean().item()
            else:
                base_loss_magnitude = base_loss.detach().item()

            # 確保傅立葉損失是標量
            if fourier_loss.numel() > 1:
                fourier_loss_magnitude = fourier_loss.detach().mean().item()
            else:
                fourier_loss_magnitude = fourier_loss.detach().item()

            # 計算適應性權重，確保傅立葉損失不會壓倒基礎損失
            adaptive_weight = fourier_weight
            if (fourier_loss_magnitude > 0 and base_loss_magnitude > 0 and
                not (torch.isnan(torch.tensor(fourier_loss_magnitude)) or
                     torch.isnan(torch.tensor(base_loss_magnitude)))):
                ratio = fourier_loss_magnitude / base_loss_magnitude
                if ratio > 10.0:  # 如果傅立葉損失過大，降低權重
                    adaptive_weight = fourier_weight / (ratio / 10.0)
                    adaptive_weight = max(adaptive_weight, fourier_weight * 0.1)
        except (RuntimeError, ValueError, AttributeError) as e:
            # 如果無法獲取標量值，使用原始權重
            logger = logging.getLogger(__name__)
            logger.debug(f"無法計算自適應權重，使用原始權重: {e}")
            adaptive_weight = fourier_weight

        # 組合基礎損失和傅立葉損失
        total_loss = base_loss + adaptive_weight * fourier_loss

        return total_loss

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"傅立葉損失計算失敗: {e}，回退到基礎損失")
        return base_loss


def fourier_latent_loss_unified(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    # 基本參數
    dims: tuple = (-2, -1),
    norm_type: str = "l2",
    eps: float = 1e-8,
    # 多尺度參數
    scales: List[int] = None,
    scale_weights: List[float] = None,
    enable_multiscale: bool = True,
    # 頻率加權參數
    enable_frequency_weighting: bool = True,
    high_freq_weight: float = 2.0,
    freq_weight_per_scale: List[float] = None,  # 每個尺度的頻率權重
    # 自適應參數
    enable_adaptive: bool = True,
    current_step: int = 0,
    total_steps: int = 1000,
    adaptive_mode: str = "linear",  # "linear", "cosine", "exponential"
    max_weight: float = 2.5,
    min_weight: float = 0.8,
    # 整合策略參數
    multiscale_weight: float = 0.6,  # 多尺度損失權重
    weighted_weight: float = 0.4,   # 加權損失權重
    adaptive_scaling: bool = True,   # 是否對權重進行自適應縮放
) -> torch.Tensor:
    """
    整合型傅立葉潛在損失計算

    結合了多尺度、頻率加權和自適應三種策略的統一實現

    Args:
        model_pred: 模型預測張量
        target: 目標張量
        dims: FFT 計算維度
        norm_type: 損失範數類型
        eps: 數值穩定性常數

        # 多尺度參數
        scales: 多尺度列表
        scale_weights: 各尺度權重
        enable_multiscale: 是否啟用多尺度

        # 頻率加權參數
        enable_frequency_weighting: 是否啟用頻率加權
        high_freq_weight: 基礎高頻權重
        freq_weight_per_scale: 每個尺度的頻率權重覆寫

        # 自適應參數
        enable_adaptive: 是否啟用自適應
        current_step: 當前步數
        total_steps: 總步數
        adaptive_mode: 自適應模式
        max_weight: 最大權重
        min_weight: 最小權重

        # 整合策略參數
        multiscale_weight: 多尺度分量權重
        weighted_weight: 加權分量權重
        adaptive_scaling: 是否自適應縮放組合權重

    Returns:
        整合的傅立葉損失
    """

    # 參數驗證和預設值設定
    if scales is None:
        scales = [1, 2, 4] if enable_multiscale else [1]

    if scale_weights is None:
        # 動態計算尺度權重，大尺度給予更小權重
        scale_weights = [1.0 / (scale ** 0.5) for scale in scales]
        # 正規化權重
        total_scale_weight = sum(scale_weights)
        scale_weights = [w / total_scale_weight for w in scale_weights]

    if freq_weight_per_scale is None:
        freq_weight_per_scale = [high_freq_weight] * len(scales)
    elif len(freq_weight_per_scale) != len(scales):
        # 擴展或截斷到正確長度
        freq_weight_per_scale = (freq_weight_per_scale + [high_freq_weight] * len(scales))[:len(scales)]

    # 計算自適應權重因子
    adaptive_factor = 1.0
    if enable_adaptive:
        # 計算訓練進度
        progress = min(current_step / max(total_steps, 1), 1.0)

        # 根據不同模式計算自適應因子
        if adaptive_mode == "linear":
            # 線性衰減：從 max_weight 到 min_weight
            adaptive_factor = max_weight - (max_weight - min_weight) * progress
        elif adaptive_mode == "cosine":
            # 餘弦衰減：更平滑的過渡
            import math
            adaptive_factor = min_weight + (max_weight - min_weight) * 0.5 * (1 + math.cos(math.pi * progress))
        elif adaptive_mode == "exponential":
            # 指數衰減：早期快速下降，後期緩慢
            import math
            adaptive_factor = min_weight + (max_weight - min_weight) * math.exp(-5 * progress)
        else:
            raise ValueError(f"Unsupported adaptive_mode: {adaptive_mode}")

    # 計算多尺度損失分量
    multiscale_loss = 0.0
    if enable_multiscale and len(scales) > 1:
        total_loss = 0.0
        total_weight = 0.0

        for i, (scale, scale_weight) in enumerate(zip(scales, scale_weights)):
            # 獲取該尺度的張量
            if scale == 1:
                pred_scaled = model_pred
                target_scaled = target
            else:
                # 檢查張量維度
                if (model_pred.dim() >= 4 and
                    model_pred.shape[-1] >= scale and model_pred.shape[-2] >= scale):
                    pred_scaled = torch.nn.functional.avg_pool2d(model_pred, scale)
                    target_scaled = torch.nn.functional.avg_pool2d(target, scale)
                else:
                    continue

            # 計算該尺度的頻率加權損失
            if enable_frequency_weighting:
                current_freq_weight = freq_weight_per_scale[i] * adaptive_factor
                scale_loss = fourier_latent_loss_weighted(
                    pred_scaled, target_scaled, current_freq_weight, dims, norm_type, eps
                )
            else:
                scale_loss = fourier_latent_loss_basic(
                    pred_scaled, target_scaled, norm_type, dims, eps
                )

            total_loss += scale_weight * scale_loss
            total_weight += scale_weight

        if total_weight > 0:
            multiscale_loss = total_loss / total_weight

    # 計算單一尺度的加權損失分量（基準尺度）
    weighted_loss = 0.0
    if enable_frequency_weighting:
        current_freq_weight = high_freq_weight * adaptive_factor
        weighted_loss = fourier_latent_loss_weighted(
            model_pred, target, current_freq_weight, dims, norm_type, eps
        )
    else:
        weighted_loss = fourier_latent_loss_basic(
            model_pred, target, norm_type, dims, eps
        )

    # 組合損失
    if enable_multiscale and len(scales) > 1:
        # 自適應調整組合權重
        if adaptive_scaling:
            # 根據訓練進度調整多尺度和加權損失的比例
            progress = min(current_step / max(total_steps, 1), 1.0)
            # 早期更重視多尺度，後期更重視細節
            current_multiscale_weight = multiscale_weight * (1.0 + 0.5 * (1.0 - progress))
            current_weighted_weight = weighted_weight * (1.0 + 0.5 * progress)

            # 正規化權重
            total_weight = current_multiscale_weight + current_weighted_weight
            current_multiscale_weight /= total_weight
            current_weighted_weight /= total_weight
        else:
            current_multiscale_weight = multiscale_weight
            current_weighted_weight = weighted_weight

        final_loss = (current_multiscale_weight * multiscale_loss +
                     current_weighted_weight * weighted_loss)
    else:
        # 如果沒有多尺度，只使用加權損失
        final_loss = weighted_loss

    # 最終約束
    final_loss = torch.clamp(final_loss, max=10.0)

    return final_loss

def get_fourier_loss_unified_config(mode: str = "balanced") -> Dict[str, Any]:
    """
    獲取預設的整合傅立葉損失設定
    """
    # 預設配置
    configs = {
        "basic": {
            "enable_multiscale": False,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "high_freq_weight": 1.5,
            "adaptive_mode": "linear",
            "max_weight": 2.0,
            "min_weight": 1.0,
        },
        "balanced": {
            "enable_multiscale": True,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "scales": [1, 2],
            "high_freq_weight": 2.0,
            "adaptive_mode": "linear",
            "max_weight": 2.5,
            "min_weight": 0.8,
            "multiscale_weight": 0.6,
            "weighted_weight": 0.4,
        },
        "detail": {
            "enable_multiscale": True,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "scales": [1, 2, 4],
            "high_freq_weight": 2.5,
            "freq_weight_per_scale": [2.0, 2.5, 3.0],
            "adaptive_mode": "cosine",
            "max_weight": 3.0,
            "min_weight": 1.0,
            "multiscale_weight": 0.7,
            "weighted_weight": 0.3,
        },
        "adaptive": {
            "enable_multiscale": True,
            "enable_frequency_weighting": True,
            "enable_adaptive": True,
            "scales": [1, 2],
            "adaptive_mode": "exponential",
            "max_weight": 2.8,
            "min_weight": 0.5,
            "adaptive_scaling": True,
        }
    }

    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(configs.keys())}")

    return configs[mode]

def fourier_latent_loss_unified_simple(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "balanced",
    current_step: int = 0,
    total_steps: int = 1000,
    **kwargs
) -> torch.Tensor:
    """
    簡化版整合傅立葉損失，提供預設配置

    Args:
        model_pred: 模型預測張量
        target: 目標張量
        mode: 預設模式
            - "basic": 基礎模式，主要使用單尺度加權
            - "balanced": 平衡模式，結合多尺度和加權
            - "detail": 細節模式，重視高頻和多尺度
            - "adaptive": 自適應模式，強調動態調整
        current_step: 當前步數
        total_steps: 總步數
        **kwargs: 其他參數覆寫

    Returns:
        傅立葉損失
    """

    # 合併配置和用戶參數
    config = get_fourier_loss_unified_config(mode).copy()
    config.update(kwargs)

    return fourier_latent_loss_unified(
        model_pred, target,
        current_step=current_step,
        total_steps=total_steps,
        **config
    )


# 便利函數：預設設定
def get_fourier_loss_config(mode: str = "balanced") -> Dict[str, Any]:
    """
    獲取預設的傅立葉損失設定

    Args:
        mode: 設定模式
            - "conservative": 保守設定，較小的傅立葉權重
            - "balanced": 平衡設定，中等的傅立葉權重
            - "aggressive": 激進設定，較大的傅立葉權重
            - "super_resolution": 專門針對超解析度任務
            - "fine_detail": 專注於細節增強

    Returns:
        設定字典
    """
    configs = {
        "conservative": {
            "fourier_weight": 0.01,
            "fourier_mode": "basic",
            "fourier_norm": "l2",
            "fourier_high_freq_weight": 1.5,
            "fourier_warmup_steps": 500
        },
        "balanced": {
            "fourier_weight": 0.05,
            "fourier_mode": "weighted",
            "fourier_norm": "l2",
            "fourier_high_freq_weight": 2.0,
            "fourier_warmup_steps": 300
        },
        "aggressive": {
            "fourier_weight": 0.1,
            "fourier_mode": "multiscale",
            "fourier_norm": "l1",
            "fourier_scales": [1, 2, 4],
            "fourier_warmup_steps": 200
        },
        "super_resolution": {
            "fourier_weight": 0.08,
            "fourier_mode": "adaptive",
            "fourier_norm": "l2",
            "fourier_adaptive_max_weight": 3.0,
            "fourier_adaptive_min_weight": 1.0,
            "fourier_warmup_steps": 400
        },
        "fine_detail": {
            "fourier_weight": 0.12,
            "fourier_mode": "weighted",
            "fourier_norm": "l1",
            "fourier_high_freq_weight": 2.5,
            "fourier_warmup_steps": 100
        },
        "unified_balanced": {
            "fourier_weight": 0.06,
            "fourier_mode": "unified_balanced",
            "fourier_norm": "l2",
            "fourier_warmup_steps": 250
        },
        "unified_detail": {
            "fourier_weight": 0.08,
            "fourier_mode": "unified_detail",
            "fourier_norm": "l2",
            "fourier_warmup_steps": 200
        },
        "unified_adaptive": {
            "fourier_weight": 0.07,
            "fourier_mode": "unified_adaptive",
            "fourier_norm": "l2",
            "fourier_warmup_steps": 300
        },
        "unified_custom": {
            "fourier_weight": 0.05,
            "fourier_mode": "unified",
            "fourier_norm": "l2",
            "fourier_high_freq_weight": 2.0,
            "fourier_scales": [1, 2, 4],
            "fourier_adaptive_max_weight": 2.5,
            "fourier_adaptive_min_weight": 0.8,
            "fourier_warmup_steps": 250
        }
    }

    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {list(configs.keys())}")

    return configs[mode]


# 為訓練腳本提供的便利函數
def apply_fourier_loss_to_args(args, quick_mode: str = "balanced"):
    """
    將傅立葉損失設定應用到訓練參數

    Args:
        args: 訓練參數對象
        quick_mode: 設定模式
    """
    quick_mode = quick_mode if quick_mode in ["conservative", "balanced", "aggressive", "super_resolution", "fine_detail", "unified_balanced", "unified_detail", "unified_adaptive", "unified_custom"] else "balanced"

    config = get_fourier_loss_config(quick_mode)

    # 設置損失類型為 fourier
    args.loss_type = "fourier"

    # 設置傅立葉相關參數
    for key, value in config.items():
        setattr(args, key, value)

    # 如果參數不存在，則設置為預設值
    if hasattr(args, "fourier_weight") is False:
        args.fourier_weight = 0.05
    if hasattr(args, "fourier_mode") is False:
        args.fourier_mode = "weighted"
    if hasattr(args, "fourier_norm") is False:
        args.fourier_norm = "l2"
    if hasattr(args, "fourier_dims") is False:
        args.fourier_dims = (-2, -1)
    if hasattr(args, "fourier_high_freq_weight") is False:
        args.fourier_high_freq_weight = 2.0
    if hasattr(args, "fourier_scales") is False:
        args.fourier_scales = None
    if hasattr(args, "fourier_scale_weights") is False:
        args.fourier_scale_weights = None
    if hasattr(args, "fourier_adaptive_max_weight") is False:
        args.fourier_adaptive_max_weight = 2.0
    if hasattr(args, "fourier_adaptive_min_weight") is False:
        args.fourier_adaptive_min_weight = 0.5
    if hasattr(args, "fourier_eps") is False:
        args.fourier_eps = 1e-8
    if hasattr(args, "fourier_warmup_steps") is False:
        args.fourier_warmup_steps = 300

    if args.fourier_mode in ["unified_basic", "unified_balanced", "unified_detail", "unified_adaptive"]:
        # 使用簡化版整合模式
        mode_map = {
            "unified_basic": "basic",
            "unified_balanced": "balanced",
            "unified_detail": "detail",
            "unified_adaptive": "adaptive"
        }
        args.fourier_unified_config = get_fourier_loss_unified_config(mode_map[args.fourier_mode])

    return args

