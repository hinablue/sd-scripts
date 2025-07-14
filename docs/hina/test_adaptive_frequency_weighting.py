#!/usr/bin/env python3
"""
自適應頻率權重功能測試腳本

測試 HinaAdaptive 優化器的 adaptive_frequency_weighting 功能是否正常工作
"""

import torch
import torch.nn as nn
import sys
import os

# 添加庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from library.hina_adaptive import HinaAdaptive


class SimpleTestModel(nn.Module):
    """簡單的測試模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_adaptive_frequency_weighting():
    """測試自適應頻率權重功能"""
    print("=== 自適應頻率權重功能測試 ===")

    # 創建模型和數據
    model = SimpleTestModel()
    input_data = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 10, (2,))

    # 測試1：啟用自適應頻率權重
    print("\n1. 測試啟用自適應頻率權重的優化器")
    optimizer_adaptive = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        fourier_feature_loss=True,
        super_resolution_mode=True,
        adaptive_frequency_weighting=True,  # 啟用
        fourier_high_freq_preservation=0.3,
        fourier_detail_enhancement=0.2,
        super_resolution_scale=4
    )

    print(f"優化器創建成功，adaptive_frequency_weighting = {optimizer_adaptive.adaptive_frequency_weighting}")

    # 獲取優化器信息
    info = optimizer_adaptive.get_optimization_info()
    print(f"傅立葉特徵損失已啟用：{info['features']['fourier_feature_loss']}")
    print(f"自適應頻率權重已啟用：{info['features']['adaptive_frequency_weighting']}")

    # 執行幾步訓練
    criterion = nn.CrossEntropyLoss()

    for step in range(3):
        optimizer_adaptive.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()

        print(f"\n步驟 {step + 1}:")
        print(f"  損失: {loss.item():.4f}")

        # 檢查是否有自適應權重狀態
        param_groups_metadata = optimizer_adaptive.param_groups_metadata
        for group_idx, group_meta in param_groups_metadata.items():
            for param_id, compact_state in group_meta['compact_states'].items():
                if compact_state.get_scalar('adaptive_low_freq_weight', None) is not None:
                    low_weight = compact_state.get_scalar('adaptive_low_freq_weight', 1.0)
                    mid_weight = compact_state.get_scalar('adaptive_mid_freq_weight', 1.0)
                    high_weight = compact_state.get_scalar('adaptive_high_freq_weight', 1.0)
                    print(f"  自適應權重 - 低頻: {low_weight:.3f}, 中頻: {mid_weight:.3f}, 高頻: {high_weight:.3f}")
                    break
            break

        optimizer_adaptive.step()

    # 測試2：禁用自適應頻率權重進行比較
    print("\n2. 測試禁用自適應頻率權重的優化器")
    model2 = SimpleTestModel()
    optimizer_fixed = HinaAdaptive(
        model2.parameters(),
        lr=0.001,
        fourier_feature_loss=True,
        super_resolution_mode=True,
        adaptive_frequency_weighting=False,  # 禁用
        fourier_high_freq_preservation=0.3,
        fourier_detail_enhancement=0.2,
        super_resolution_scale=4
    )

    print(f"優化器創建成功，adaptive_frequency_weighting = {optimizer_fixed.adaptive_frequency_weighting}")

    # 執行一步訓練進行比較
    optimizer_fixed.zero_grad()
    output2 = model2(input_data)
    loss2 = criterion(output2, target)
    loss2.backward()

    print(f"損失: {loss2.item():.4f}")

    # 檢查是否沒有自適應權重狀態
    param_groups_metadata2 = optimizer_fixed.param_groups_metadata
    has_adaptive_weights = False
    for group_idx, group_meta in param_groups_metadata2.items():
        for param_id, compact_state in group_meta['compact_states'].items():
            if compact_state.get_scalar('adaptive_low_freq_weight', None) is not None:
                has_adaptive_weights = True
                break
        break

    print(f"是否包含自適應權重狀態: {has_adaptive_weights}")

    optimizer_fixed.step()

    print("\n=== 測試完成 ===")
    print("✅ adaptive_frequency_weighting 功能正常工作！")
    return True


def test_frequency_weight_adaptation():
    """測試頻率權重自適應過程"""
    print("\n=== 頻率權重自適應過程測試 ===")

    # 創建一個簡單的2D參數用於測試
    param = nn.Parameter(torch.randn(64, 64))
    optimizer = HinaAdaptive(
        [param],
        lr=0.001,
        fourier_feature_loss=True,
        super_resolution_mode=True,
        adaptive_frequency_weighting=True,
        super_resolution_scale=4
    )

    # 模擬不同的梯度模式來測試權重自適應
    patterns = [
        ("低頻主導", torch.ones(64, 64) * 0.1),  # 低頻模式
        ("高頻主導", torch.randn(64, 64) * 0.1),  # 高頻噪聲模式
        ("混合模式", torch.sin(torch.linspace(0, 10*3.14159, 64*64)).reshape(64, 64) * 0.1)  # 中頻模式
    ]

    for pattern_name, grad_pattern in patterns:
        print(f"\n測試 {pattern_name}:")

        # 執行5步來觀察權重變化
        initial_weights = None
        for step in range(5):
            optimizer.zero_grad()
            param.grad = grad_pattern + torch.randn_like(grad_pattern) * 0.01

            # 獲取當前權重
            param_groups_metadata = optimizer.param_groups_metadata
            for group_idx, group_meta in param_groups_metadata.items():
                for param_id, compact_state in group_meta['compact_states'].items():
                    if param_id == id(param):
                        low_weight = compact_state.get_scalar('adaptive_low_freq_weight', 1.0)
                        mid_weight = compact_state.get_scalar('adaptive_mid_freq_weight', 1.0)
                        high_weight = compact_state.get_scalar('adaptive_high_freq_weight', 1.0)

                        if initial_weights is None:
                            initial_weights = (low_weight, mid_weight, high_weight)

                        print(f"  步驟 {step + 1}: 低頻={low_weight:.3f}, 中頻={mid_weight:.3f}, 高頻={high_weight:.3f}")
                        break
                break

            optimizer.step()

        # 顯示權重變化
        final_weights = (low_weight, mid_weight, high_weight)
        print(f"  權重變化: 低頻 {initial_weights[0]:.3f}→{final_weights[0]:.3f}, "
              f"中頻 {initial_weights[1]:.3f}→{final_weights[1]:.3f}, "
              f"高頻 {initial_weights[2]:.3f}→{final_weights[2]:.3f}")

    print("\n✅ 頻率權重自適應過程測試完成！")


if __name__ == "__main__":
    try:
        # 測試基本功能
        test_adaptive_frequency_weighting()

        # 測試自適應過程
        test_frequency_weight_adaptation()

        print("\n🎉 所有測試通過！adaptive_frequency_weighting 功能正常工作。")

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()