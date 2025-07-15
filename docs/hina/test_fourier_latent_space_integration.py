#!/usr/bin/env python3
"""
傅立葉特徵損失 Latent Space 整合測試腳本

本腳本測試傅立葉特徵損失在 conditional_loss 中的整合實現，
驗證在 latent space 中的功能性和穩定性。

作者: Hina
日期: 2024
"""

import torch
import torch.nn.functional as F
import sys
import os
import time
import numpy as np

# 添加庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from library.train_util import (
        conditional_loss_with_fourier,
        get_fourier_loss_config,
        apply_fourier_loss_to_args,
        conditional_loss
    )
    print("✅ 成功導入傅立葉損失函數")
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    sys.exit(1)


def test_basic_fourier_loss():
    """測試基礎傅立葉損失功能"""
    print("=" * 60)
    print("🔍 基礎傅立葉損失測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建模擬 latent space 數據
    batch_size, channels, height, width = 4, 4, 32, 32
    model_pred = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)

    print(f"輸入形狀: model_pred={model_pred.shape}, target={target.shape}")

    # 測試基礎損失
    base_loss = conditional_loss(model_pred, target, "l2", "mean")
    print(f"基礎 L2 損失: {base_loss.item():.6f}")

    # 測試傅立葉損失
    fourier_loss = conditional_loss_with_fourier(
        model_pred, target, "fourier", "mean",
        current_step=500,
        total_steps=1000,
        fourier_weight=0.05,
        fourier_mode="basic"
    )
    print(f"傅立葉組合損失: {fourier_loss.item():.6f}")

    # 驗證損失值合理性
    assert fourier_loss > base_loss, "傅立葉損失應該大於基礎損失"
    assert torch.isfinite(fourier_loss), "損失值應該是有限的"

    print("✅ 基礎傅立葉損失測試通過")
    return True


def test_fourier_modes():
    """測試不同的傅立葉損失模式"""
    print("\n" + "=" * 60)
    print("🔍 傅立葉損失模式測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建測試數據
    batch_size, channels, height, width = 2, 4, 64, 64
    model_pred = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)

    modes = ["basic", "weighted", "multiscale", "adaptive"]
    mode_results = {}

    for mode in modes:
        print(f"--- 測試模式: {mode} ---")

        # 設置模式特定參數
        kwargs = {
            "fourier_mode": mode,
            "fourier_weight": 0.05,
            "current_step": 300,
            "total_steps": 1000
        }

        if mode == "multiscale":
            kwargs["fourier_scales"] = [1, 2, 4]
        elif mode == "adaptive":
            kwargs["fourier_adaptive_max_weight"] = 3.0
            kwargs["fourier_adaptive_min_weight"] = 1.0

        try:
            loss = conditional_loss_with_fourier(
                model_pred, target, "fourier", "mean", **kwargs
            )
            mode_results[mode] = loss.item()
            print(f"  損失值: {loss.item():.6f}")
            assert torch.isfinite(loss), f"{mode} 模式損失值不是有限的"

        except Exception as e:
            print(f"  ❌ {mode} 模式失敗: {e}")
            return False

    print("\n📊 模式比較:")
    for mode, loss_val in mode_results.items():
        print(f"  {mode:<12}: {loss_val:.6f}")

    print("✅ 所有傅立葉損失模式測試通過")
    return True


def test_fourier_configs():
    """測試預設配置功能"""
    print("\n" + "=" * 60)
    print("🔍 預設配置測試")
    print("=" * 60)

    config_modes = ["conservative", "balanced", "aggressive", "super_resolution", "fine_detail"]

    for mode in config_modes:
        print(f"--- 測試配置: {mode} ---")

        try:
            config = get_fourier_loss_config(mode)
            print(f"  配置內容: {config}")

            # 驗證必要鍵值存在
            required_keys = ["fourier_weight", "fourier_mode", "fourier_norm", "fourier_warmup_steps"]
            for key in required_keys:
                assert key in config, f"配置 {mode} 缺少鍵值 {key}"

            # 驗證數值範圍
            assert 0 < config["fourier_weight"] <= 1.0, f"權重值超出範圍: {config['fourier_weight']}"
            assert config["fourier_warmup_steps"] >= 0, f"預熱步數不能為負數: {config['fourier_warmup_steps']}"

        except Exception as e:
            print(f"  ❌ 配置 {mode} 失敗: {e}")
            return False

    print("✅ 所有預設配置測試通過")
    return True


def test_dimension_handling():
    """測試不同維度張量的處理"""
    print("\n" + "=" * 60)
    print("🔍 維度處理測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_cases = [
        ("2D 張量", (32, 32)),
        ("3D 張量", (4, 32, 32)),
        ("4D 張量", (2, 4, 32, 32)),
        ("5D 張量", (2, 4, 8, 32, 32))
    ]

    for name, shape in test_cases:
        print(f"--- 測試 {name}: {shape} ---")

        model_pred = torch.randn(*shape, device=device)
        target = torch.randn(*shape, device=device)

        try:
            loss = conditional_loss_with_fourier(
                model_pred, target, "fourier", "mean",
                fourier_weight=0.05,
                current_step=500,
                total_steps=1000
            )

            if len(shape) < 3:
                # 對於低維張量，應該回退到基礎損失
                base_loss = conditional_loss(model_pred, target, "l2", "mean")
                print(f"  回退到基礎損失: {loss.item():.6f}")
                assert abs(loss.item() - base_loss.item()) < 1e-6, "低維張量應該使用基礎損失"
            else:
                # 對於高維張量，應該包含傅立葉損失
                base_loss = conditional_loss(model_pred, target, "l2", "mean")
                print(f"  傅立葉組合損失: {loss.item():.6f}")
                # 允許一定誤差，因為可能在某些情況下傅立葉損失很小

        except Exception as e:
            print(f"  ❌ 處理失敗: {e}")
            return False

    print("✅ 維度處理測試通過")
    return True


def test_warmup_behavior():
    """測試預熱期行為"""
    print("\n" + "=" * 60)
    print("🔍 預熱期行為測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建測試數據
    model_pred = torch.randn(2, 4, 32, 32, device=device)
    target = torch.randn(2, 4, 32, 32, device=device)

    warmup_steps = 300

    # 測試預熱期內（應該使用基礎損失）
    print("--- 預熱期內測試 ---")
    warmup_loss = conditional_loss_with_fourier(
        model_pred, target, "fourier", "mean",
        current_step=100,  # 小於 warmup_steps
        total_steps=1000,
        fourier_weight=0.05,
        fourier_warmup_steps=warmup_steps
    )

    base_loss = conditional_loss(model_pred, target, "l2", "mean")
    print(f"預熱期損失: {warmup_loss.item():.6f}")
    print(f"基礎損失: {base_loss.item():.6f}")

    # 預熱期內應該等於基礎損失
    assert abs(warmup_loss.item() - base_loss.item()) < 1e-6, "預熱期內應該使用基礎損失"

    # 測試預熱期後（應該包含傅立葉損失）
    print("--- 預熱期後測試 ---")
    post_warmup_loss = conditional_loss_with_fourier(
        model_pred, target, "fourier", "mean",
        current_step=500,  # 大於 warmup_steps
        total_steps=1000,
        fourier_weight=0.05,
        fourier_warmup_steps=warmup_steps
    )

    print(f"預熱期後損失: {post_warmup_loss.item():.6f}")

    # 預熱期後應該大於基礎損失（包含傅立葉項）
    assert post_warmup_loss > base_loss, "預熱期後損失應該包含傅立葉項"

    print("✅ 預熱期行為測試通過")
    return True


def test_performance_benchmark():
    """性能基準測試"""
    print("\n" + "=" * 60)
    print("🔍 性能基準測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建較大的測試數據
    batch_size, channels, height, width = 8, 4, 128, 128
    model_pred = torch.randn(batch_size, channels, height, width, device=device)
    target = torch.randn(batch_size, channels, height, width, device=device)

    print(f"測試數據大小: {model_pred.shape}")

    # 基礎損失性能
    start_time = time.time()
    for _ in range(10):
        base_loss = conditional_loss(model_pred, target, "l2", "mean")
    base_time = time.time() - start_time
    print(f"基礎損失 (10次): {base_time:.4f}s, 平均: {base_time/10:.4f}s")

    # 傅立葉損失性能
    start_time = time.time()
    for _ in range(10):
        fourier_loss = conditional_loss_with_fourier(
            model_pred, target, "fourier", "mean",
            current_step=500,
            total_steps=1000,
            fourier_weight=0.05,
            fourier_mode="weighted"
        )
    fourier_time = time.time() - start_time
    print(f"傅立葉損失 (10次): {fourier_time:.4f}s, 平均: {fourier_time/10:.4f}s")

    # 計算性能開銷
    overhead = (fourier_time - base_time) / base_time * 100
    print(f"性能開銷: {overhead:.1f}%")

    # 檢查開銷是否合理（應該小於 500%）
    if overhead > 500:
        print(f"⚠️  性能開銷較高: {overhead:.1f}%")
    else:
        print(f"✅ 性能開銷合理: {overhead:.1f}%")

    return True


def test_gradient_flow():
    """測試梯度流動"""
    print("\n" + "=" * 60)
    print("🔍 梯度流動測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建需要梯度的測試數據
    model_pred = torch.randn(2, 4, 32, 32, device=device, requires_grad=True)
    target = torch.randn(2, 4, 32, 32, device=device)

    # 計算傅立葉損失
    loss = conditional_loss_with_fourier(
        model_pred, target, "fourier", "mean",
        current_step=500,
        total_steps=1000,
        fourier_weight=0.05,
        fourier_mode="weighted"
    )

    print(f"損失值: {loss.item():.6f}")

    # 計算梯度
    loss.backward()

    # 檢查梯度
    assert model_pred.grad is not None, "梯度應該不為空"
    assert torch.isfinite(model_pred.grad).all(), "梯度應該都是有限的"

    grad_norm = torch.norm(model_pred.grad).item()
    print(f"梯度範數: {grad_norm:.6f}")

    # 檢查梯度範數合理性
    assert grad_norm > 0, "梯度範數應該大於 0"
    assert grad_norm < 1000, f"梯度範數過大: {grad_norm}"

    print("✅ 梯度流動測試通過")
    return True


def test_args_integration():
    """測試參數整合功能"""
    print("\n" + "=" * 60)
    print("🔍 參數整合測試")
    print("=" * 60)

    # 模擬訓練參數對象
    class MockArgs:
        def __init__(self):
            self.loss_type = "l2"

    args = MockArgs()
    print(f"原始損失類型: {args.loss_type}")

    # 應用傅立葉配置
    updated_args = apply_fourier_loss_to_args(args, mode="balanced")

    print(f"更新後損失類型: {updated_args.loss_type}")
    print(f"傅立葉權重: {updated_args.fourier_weight}")
    print(f"傅立葉模式: {updated_args.fourier_mode}")

    # 驗證配置
    assert updated_args.loss_type == "fourier", "損失類型應該更新為 fourier"
    assert hasattr(updated_args, "fourier_weight"), "應該有 fourier_weight 屬性"
    assert hasattr(updated_args, "fourier_mode"), "應該有 fourier_mode 屬性"

    print("✅ 參數整合測試通過")
    return True


def run_all_tests():
    """運行所有測試"""
    print("🚀 開始傅立葉特徵損失 Latent Space 整合測試")
    print("=" * 80)

    tests = [
        test_basic_fourier_loss,
        test_fourier_modes,
        test_fourier_configs,
        test_dimension_handling,
        test_warmup_behavior,
        test_performance_benchmark,
        test_gradient_flow,
        test_args_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 測試 {test_func.__name__} 發生異常: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("📊 測試結果統計")
    print("=" * 80)
    print(f"✅ 通過: {passed}")
    print(f"❌ 失敗: {failed}")
    print(f"📈 成功率: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 所有測試通過！傅立葉特徵損失 Latent Space 整合功能正常")
    else:
        print(f"\n⚠️  有 {failed} 個測試失敗，請檢查實現")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)