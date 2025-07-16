#!/usr/bin/env python3
"""
整合傅立葉損失測試腳本

測試新的 fourier_latent_loss_unified 和 fourier_latent_loss_unified_simple 函數的功能
"""

import torch
import sys
import os

# 添加項目根目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from library.fourier_loss import (
    fourier_latent_loss_weighted,
    fourier_latent_loss_multiscale,
    fourier_latent_loss_adaptive,
    fourier_latent_loss_unified,
    fourier_latent_loss_unified_simple,
    get_fourier_loss_config
)


def create_test_tensors(batch_size=2, channels=4, height=64, width=64, device='cpu'):
    """創建測試用張量"""
    torch.manual_seed(42)

    # 創建具有不同頻率特徵的測試張量
    model_pred = torch.randn(batch_size, channels, height, width, device=device)

    # 目標張量：添加一些高頻細節
    target = model_pred.clone()
    # 添加高頻噪聲來模擬細節差異
    high_freq_noise = torch.randn_like(target) * 0.1
    target = target + high_freq_noise

    return model_pred, target


def test_individual_modes():
    """測試各個獨立模式"""
    print("=== 測試各個獨立模式 ===")

    model_pred, target = create_test_tensors()
    current_step, total_steps = 500, 1000

    # 測試加權模式
    loss_weighted = fourier_latent_loss_weighted(model_pred, target, high_freq_weight=2.0)
    print(f"Weighted Loss: {loss_weighted.item():.6f}")

    # 測試多尺度模式
    loss_multiscale = fourier_latent_loss_multiscale(model_pred, target, scales=[1, 2, 4])
    print(f"Multiscale Loss: {loss_multiscale.item():.6f}")

    # 測試自適應模式
    loss_adaptive = fourier_latent_loss_adaptive(
        model_pred, target, current_step, total_steps, max_weight=2.5, min_weight=0.8
    )
    print(f"Adaptive Loss: {loss_adaptive.item():.6f}")

    print()


def test_unified_mode():
    """測試整合模式"""
    print("=== 測試整合模式 ===")

    model_pred, target = create_test_tensors()
    current_step, total_steps = 500, 1000

    # 測試完整自定義配置
    loss_unified_custom = fourier_latent_loss_unified(
        model_pred, target,
        enable_multiscale=True,
        enable_frequency_weighting=True,
        enable_adaptive=True,
        scales=[1, 2, 4],
        high_freq_weight=2.0,
        adaptive_mode="cosine",
        max_weight=2.5,
        min_weight=0.8,
        current_step=current_step,
        total_steps=total_steps,
        multiscale_weight=0.6,
        weighted_weight=0.4
    )
    print(f"Unified Custom Loss: {loss_unified_custom.item():.6f}")

    # 測試只啟用部分功能
    loss_unified_partial = fourier_latent_loss_unified(
        model_pred, target,
        enable_multiscale=False,  # 禁用多尺度
        enable_frequency_weighting=True,
        enable_adaptive=True,
        current_step=current_step,
        total_steps=total_steps
    )
    print(f"Unified Partial Loss: {loss_unified_partial.item():.6f}")

    print()


def test_unified_simple_modes():
    """測試簡化版整合模式"""
    print("=== 測試簡化版整合模式 ===")

    model_pred, target = create_test_tensors()
    current_step, total_steps = 500, 1000

    modes = ["basic", "balanced", "detail", "adaptive"]

    for mode in modes:
        loss = fourier_latent_loss_unified_simple(
            model_pred, target,
            mode=mode,
            current_step=current_step,
            total_steps=total_steps
        )
        print(f"Unified Simple {mode.capitalize()} Loss: {loss.item():.6f}")

    print()


def test_adaptive_curves():
    """測試不同自適應曲線"""
    print("=== 測試不同自適應曲線 ===")

    model_pred, target = create_test_tensors()
    total_steps = 1000
    adaptive_modes = ["linear", "cosine", "exponential"]

    # 測試不同訓練階段
    test_steps = [0, 250, 500, 750, 1000]

    for mode in adaptive_modes:
        print(f"\n{mode.capitalize()} 模式:")
        losses = []
        for step in test_steps:
            loss = fourier_latent_loss_unified(
                model_pred, target,
                enable_adaptive=True,
                adaptive_mode=mode,
                current_step=step,
                total_steps=total_steps,
                max_weight=3.0,
                min_weight=1.0
            )
            losses.append(loss.item())
            print(f"  Step {step:4d}: {loss.item():.6f}")

    print()


def test_configuration_presets():
    """測試預設配置"""
    print("=== 測試預設配置 ===")

    modes = ["conservative", "balanced", "aggressive", "super_resolution", "fine_detail",
             "unified_balanced", "unified_detail", "unified_adaptive", "unified_custom"]

    for mode in modes:
        try:
            config = get_fourier_loss_config(mode)
            print(f"{mode:20s}: fourier_mode={config.get('fourier_mode', 'N/A'):15s}, "
                  f"weight={config.get('fourier_weight', 'N/A')}")
        except ValueError as e:
            print(f"{mode:20s}: {e}")

    print()


def test_performance_comparison():
    """性能比較測試"""
    print("=== 性能比較測試 ===")

    model_pred, target = create_test_tensors(batch_size=4, height=128, width=128)
    current_step, total_steps = 500, 1000

    import time

    # 測試各種模式的執行時間
    tests = [
        ("Weighted", lambda: fourier_latent_loss_weighted(model_pred, target)),
        ("Multiscale", lambda: fourier_latent_loss_multiscale(model_pred, target)),
        ("Adaptive", lambda: fourier_latent_loss_adaptive(model_pred, target, current_step, total_steps)),
        ("Unified Custom", lambda: fourier_latent_loss_unified(model_pred, target, current_step=current_step, total_steps=total_steps)),
        ("Unified Simple", lambda: fourier_latent_loss_unified_simple(model_pred, target, "balanced", current_step, total_steps))
    ]

    iterations = 10

    for name, func in tests:
        # 預熱
        for _ in range(3):
            _ = func()

        # 計時
        start_time = time.time()
        for _ in range(iterations):
            loss = func()
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations * 1000  # 轉換為毫秒
        print(f"{name:15s}: {loss.item():.6f} (平均 {avg_time:.2f}ms)")

    print()


def test_gradient_flow():
    """測試梯度流動"""
    print("=== 測試梯度流動 ===")

    model_pred, target = create_test_tensors()
    model_pred.requires_grad_(True)

    # 測試不同模式的梯度計算
    loss_unified = fourier_latent_loss_unified_simple(
        model_pred, target, "balanced", current_step=500, total_steps=1000
    )

    loss_unified.backward()

    grad_norm = model_pred.grad.norm().item()
    grad_max = model_pred.grad.abs().max().item()
    grad_mean = model_pred.grad.abs().mean().item()

    print(f"梯度統計:")
    print(f"  Norm: {grad_norm:.6f}")
    print(f"  Max:  {grad_max:.6f}")
    print(f"  Mean: {grad_mean:.6f}")

    # 檢查是否有 NaN 或 Inf
    has_nan = torch.isnan(model_pred.grad).any().item()
    has_inf = torch.isinf(model_pred.grad).any().item()
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

    print()


def main():
    """主測試函數"""
    print("開始測試整合傅立葉損失實現...")
    print("=" * 50)

    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    print()

    # 運行各項測試
    test_individual_modes()
    test_unified_mode()
    test_unified_simple_modes()
    test_adaptive_curves()
    test_configuration_presets()
    test_performance_comparison()
    test_gradient_flow()

    print("=" * 50)
    print("所有測試完成！")


if __name__ == "__main__":
    main()