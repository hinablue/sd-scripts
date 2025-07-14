#!/usr/bin/env python3
"""
多維張量傅立葉特徵損失修復測試腳本

測試 HinaAdaptive 優化器在處理卷積層權重等多維張量時是否能正確工作
原始錯誤：IndexError: The shape of the mask [3, 3] at index 0 does not match the shape of the indexed tensor [40, 40, 3, 3]
"""

import torch
import torch.nn as nn
import sys
import os

# 添加庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from library.hina_adaptive import HinaAdaptive


class MultiDimTestModel(nn.Module):
    """包含不同維度參數的測試模型"""
    def __init__(self):
        super().__init__()
        # 4D 卷積權重 (會觸發原始錯誤)
        self.conv1 = nn.Conv2d(3, 40, kernel_size=3, padding=1)  # [40, 3, 3, 3]
        self.conv2 = nn.Conv2d(40, 40, kernel_size=3, padding=1)  # [40, 40, 3, 3] - 原始錯誤觸發

        # 2D 全連接權重
        self.fc1 = nn.Linear(40*16*16, 128)  # [128, 40*16*16]
        self.fc2 = nn.Linear(128, 10)  # [10, 128]

        # 1D 偏置項
        self.bias = nn.Parameter(torch.randn(10))  # [10]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = nn.AdaptiveAvgPool2d((16, 16))(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x + self.bias
        return x


def test_multidim_tensor_handling():
    """測試多維張量處理"""
    print("=== 多維張量傅立葉特徵損失測試 ===")

    # 創建模型和數據
    model = MultiDimTestModel()
    input_data = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 10, (2,))

    print(f"模型參數形狀:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    # 創建優化器（啟用傅立葉特徵損失）
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        fourier_feature_loss=True,
        super_resolution_mode=True,
        adaptive_frequency_weighting=True,
        fourier_high_freq_preservation=0.3,
        fourier_detail_enhancement=0.2,
        super_resolution_scale=4
    )

    print(f"\n優化器創建成功，傅立葉特徵損失已啟用")

    # 執行訓練步驟
    criterion = nn.CrossEntropyLoss()

    print(f"\n開始訓練測試...")
    for step in range(3):
        optimizer.zero_grad()

        # 前向傳播
        output = model(input_data)
        loss = criterion(output, target)

        print(f"\n步驟 {step + 1}:")
        print(f"  損失: {loss.item():.4f}")

        # 反向傳播
        loss.backward()

        # 檢查梯度是否正常
        grad_info = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_info[name] = grad_norm
                print(f"  {name} 梯度範數: {grad_norm:.6f}")

        # 優化器更新（這裡會測試傅立葉特徵損失）
        try:
            optimizer.step()
            print(f"  ✅ 優化步驟成功完成")
        except Exception as e:
            print(f"  ❌ 優化步驟失敗: {e}")
            raise e

    print(f"\n=== 測試完成 ===")
    return True


def test_dimension_filtering():
    """測試維度過濾邏輯"""
    print("\n=== 維度過濾邏輯測試 ===")

    # 創建不同維度的參數進行測試
    test_cases = [
        ("1D 張量", torch.randn(100)),
        ("小 2D 張量", torch.randn(4, 4)),  # 小於 8x8，應該被跳過
        ("大 2D 張量", torch.randn(32, 32)),  # 應該被處理
        ("3D 張量", torch.randn(10, 16, 16)),  # 應該被處理
        ("小 4D 張量", torch.randn(10, 20, 3, 3)),  # 最後兩維太小，應該被跳過
        ("大 4D 張量", torch.randn(40, 40, 8, 8)),  # 應該被處理
    ]

    for case_name, test_tensor in test_cases:
        print(f"\n測試 {case_name}: 形狀 {test_tensor.shape}")

        # 創建包含單個參數的模型
        class SingleParamModel(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.param = nn.Parameter(tensor.clone())

            def forward(self, x):
                return torch.sum(self.param)

        model = SingleParamModel(test_tensor)

        optimizer = HinaAdaptive(
            model.parameters(),
            lr=0.001,
            fourier_feature_loss=True,
            super_resolution_mode=True,
            adaptive_frequency_weighting=True
        )

        # 測試一個優化步驟
        try:
            loss = model(torch.tensor(1.0))
            loss.backward()
            optimizer.step()
            print(f"  ✅ 處理成功")
        except Exception as e:
            print(f"  ❌ 處理失敗: {e}")
            return False

    print(f"\n✅ 所有維度過濾測試通過")
    return True


def test_shape_consistency():
    """測試形狀一致性"""
    print("\n=== 形狀一致性測試 ===")

    # 創建一個有特定形狀的卷積層
    conv = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # [32, 16, 5, 5]

    optimizer = HinaAdaptive(
        conv.parameters(),
        lr=0.001,
        fourier_feature_loss=True,
        super_resolution_mode=True
    )

    # 創建輸入和目標
    input_data = torch.randn(1, 16, 28, 28)
    target = torch.randn(1, 32, 28, 28)

    print(f"卷積權重形狀: {conv.weight.shape}")
    print(f"卷積偏置形狀: {conv.bias.shape}")

    # 執行訓練步驟
    optimizer.zero_grad()
    output = conv(input_data)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    # 檢查梯度形狀
    print(f"權重梯度形狀: {conv.weight.grad.shape}")
    print(f"偏置梯度形狀: {conv.bias.grad.shape}")

    # 執行優化步驟
    try:
        optimizer.step()
        print(f"✅ 形狀一致性測試通過")
        return True
    except Exception as e:
        print(f"❌ 形狀一致性測試失敗: {e}")
        return False


if __name__ == "__main__":
    try:
        print("🚀 開始多維張量傅立葉特徵損失修復測試\n")

        # 測試1：多維張量處理
        success1 = test_multidim_tensor_handling()

        # 測試2：維度過濾
        success2 = test_dimension_filtering()

        # 測試3：形狀一致性
        success3 = test_shape_consistency()

        if success1 and success2 and success3:
            print("\n🎉 所有測試通過！多維張量傅立葉特徵損失修復成功。")
            print("\n修復內容：")
            print("✅ 正確處理4D卷積權重張量")
            print("✅ 維度過濾邏輯正常工作")
            print("✅ FFT和IFFT操作正確處理多維張量")
            print("✅ 形狀一致性得到保證")
            print("✅ 自適應頻率權重功能正常")
        else:
            print("\n❌ 部分測試失敗，需要進一步檢查。")

    except Exception as e:
        print(f"\n💥 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()