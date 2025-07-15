#!/usr/bin/env python3
"""
HinaAdaptive 正則化技術維度處理測試

測試不同形狀的張量是否能正確處理各種正則化技術，確保沒有維度不匹配錯誤。

注意：傅立葉特徵損失功能已被移除，因為它不適用於 SD-Scripts
的 latent space 訓練環境。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加庫路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from library.hina_adaptive import HinaAdaptive

def test_regularization_dimensions():
    """測試正則化技術的維度處理"""
    print("=" * 60)
    print("測試正則化技術維度處理")
    print("=" * 60)

    # 創建各種形狀的測試張量
    test_shapes = [
        (8, 8),           # 2D: 全連接層權重
        (1, 8, 8),        # 3D: 單通道卷積
        (64, 32, 3, 3),   # 4D: 2D卷積權重
        (128, 64, 5, 5),  # 4D: 較大的卷積核
        (256, 128, 7, 7), # 4D: 更大的卷積核
        (16, 32, 1, 1),   # 4D: 1x1卷積
        (32, 16, 3, 3, 3), # 5D: 3D卷積權重
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    all_tests_passed = True

    for i, shape in enumerate(test_shapes):
        print(f"\n測試 {i+1}/{len(test_shapes)}: 形狀 {shape}")
        print("-" * 40)

        try:
            # 創建測試參數
            param = torch.randn(shape, device=device, requires_grad=True)

            # 創建優化器，啟用各種正則化技術
            optimizer = HinaAdaptive(
                [param],
                lr=1e-3,
                # 各種正則化技術
                edge_suppression=True,
                edge_penalty=0.1,
                spatial_awareness=True,
                frequency_penalty=0.05,
                background_regularization=True,
                lora_rank_penalty=True,
                rank_penalty_strength=0.01,
                # 記憶體優化
                memory_efficient=True,
                vram_budget_gb=8.0
            )

            # 創建對應的目標張量
            target = torch.randn_like(param)

            # 模擬前向傳播
            output = param * 2.0  # 簡單操作
            loss = torch.nn.functional.mse_loss(output, target)

            # 測試反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"✅ 形狀 {shape} 處理成功")
            print(f"   - 參數元素數量: {param.numel()}")
            print(f"   - 梯度範數: {torch.norm(param.grad).item():.6f}")
            print(f"   - 損失值: {loss.item():.6f}")

            # 驗證優化器狀態
            info = optimizer.get_optimization_info()
            print(f"   - 啟用的正則化技術: {sum(info['features'].values())}")

        except Exception as e:
            print(f"❌ 形狀 {shape} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def test_edge_suppression_dimensions():
    """測試邊緣感知正則化的維度處理"""
    print("\n" + "=" * 60)
    print("測試邊緣感知正則化維度處理")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 測試不同維度的卷積層
    test_cases = [
        ("2D卷積", nn.Conv2d(3, 16, 3, padding=1)),
        ("大卷積核", nn.Conv2d(16, 32, 7, padding=3)),
        ("1x1卷積", nn.Conv2d(32, 64, 1)),
        ("深度卷積", nn.Conv2d(64, 128, 5, padding=2)),
    ]

    all_tests_passed = True

    for name, layer in test_cases:
        print(f"\n測試 {name}:")
        print("-" * 30)

        try:
            layer = layer.to(device)

            # 創建邊緣感知優化器
            optimizer = HinaAdaptive(
                layer.parameters(),
                lr=1e-3,
                edge_suppression=True,
                edge_penalty=0.1,
                edge_threshold=0.6,
                memory_efficient=True
            )

            # 創建測試數據
            if isinstance(layer, nn.Conv2d):
                x = torch.randn(2, layer.in_channels, 32, 32, device=device)
            else:
                x = torch.randn(2, layer.in_channels, 32, 32, device=device)

            # 前向傳播
            output = layer(x)
            loss = torch.mean(output ** 2)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"✅ {name} 處理成功")
            print(f"   - 權重形狀: {layer.weight.shape}")
            print(f"   - 梯度範數: {torch.norm(layer.weight.grad).item():.6f}")

        except Exception as e:
            print(f"❌ {name} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def test_spatial_awareness_dimensions():
    """測試空間感知正則化的維度處理"""
    print("\n" + "=" * 60)
    print("測試空間感知正則化維度處理")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 測試不同大小的特徵圖
    test_sizes = [
        (8, 8),     # 小特徵圖
        (32, 32),   # 中等特徵圖
        (64, 64),   # 大特徵圖
        (128, 128), # 很大特徵圖
    ]

    all_tests_passed = True

    for size in test_sizes:
        print(f"\n測試特徵圖大小 {size}:")
        print("-" * 30)

        try:
            # 創建對應大小的卷積層
            layer = nn.Conv2d(3, 16, 3, padding=1).to(device)

            # 創建空間感知優化器
            optimizer = HinaAdaptive(
                layer.parameters(),
                lr=1e-3,
                spatial_awareness=True,
                frequency_penalty=0.05,
                detail_preservation=0.8,
                memory_efficient=True
            )

            # 創建測試數據
            x = torch.randn(1, 3, size[0], size[1], device=device)

            # 前向傳播
            output = layer(x)
            loss = torch.mean(output ** 2)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"✅ 特徵圖大小 {size} 處理成功")
            print(f"   - 輸入形狀: {x.shape}")
            print(f"   - 輸出形狀: {output.shape}")
            print(f"   - 損失值: {loss.item():.6f}")

        except Exception as e:
            print(f"❌ 特徵圖大小 {size} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def test_lora_regularization_dimensions():
    """測試 LoRA 低秩正則化的維度處理"""
    print("\n" + "=" * 60)
    print("測試 LoRA 低秩正則化維度處理")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 測試不同大小的矩陣
    test_matrices = [
        (64, 32),    # 小矩陣
        (128, 64),   # 中等矩陣
        (256, 128),  # 大矩陣
        (512, 256),  # 很大矩陣
        (32, 128),   # 寬矩陣
        (128, 32),   # 窄矩陣
    ]

    all_tests_passed = True

    for rows, cols in test_matrices:
        print(f"\n測試矩陣大小 {rows}x{cols}:")
        print("-" * 30)

        try:
            # 創建線性層
            layer = nn.Linear(cols, rows).to(device)

            # 創建 LoRA 低秩正則化優化器
            optimizer = HinaAdaptive(
                layer.parameters(),
                lr=1e-3,
                lora_rank_penalty=True,
                rank_penalty_strength=0.01,
                low_rank_emphasis=1.2,
                memory_efficient=True
            )

            # 創建測試數據
            x = torch.randn(8, cols, device=device)

            # 前向傳播
            output = layer(x)
            loss = torch.mean(output ** 2)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"✅ 矩陣大小 {rows}x{cols} 處理成功")
            print(f"   - 權重形狀: {layer.weight.shape}")
            print(f"   - 權重秩估計: {torch.linalg.matrix_rank(layer.weight.data).item()}")
            print(f"   - 損失值: {loss.item():.6f}")

        except Exception as e:
            print(f"❌ 矩陣大小 {rows}x{cols} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def test_background_regularization_dimensions():
    """測試背景正則化的維度處理"""
    print("\n" + "=" * 60)
    print("測試背景正則化維度處理")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 測試不同形狀的張量
    test_shapes = [
        (1, 16, 16),      # 3D張量
        (3, 32, 32),      # RGB圖像
        (64, 64, 64),     # 體積數據
        (16, 128, 128),   # 高解析度
    ]

    all_tests_passed = True

    for shape in test_shapes:
        print(f"\n測試張量形狀 {shape}:")
        print("-" * 30)

        try:
            # 創建卷積層
            if len(shape) == 3:
                layer = nn.Conv2d(shape[0], 16, 3, padding=1).to(device)
            else:
                layer = nn.Conv2d(shape[0], 16, 3, padding=1).to(device)

            # 創建背景正則化優化器
            optimizer = HinaAdaptive(
                layer.parameters(),
                lr=1e-3,
                background_regularization=True,
                memory_efficient=True
            )

            # 創建測試數據
            x = torch.randn(1, *shape, device=device)

            # 前向傳播
            output = layer(x)
            loss = torch.mean(output ** 2)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"✅ 張量形狀 {shape} 處理成功")
            print(f"   - 輸入形狀: {x.shape}")
            print(f"   - 輸出形狀: {output.shape}")
            print(f"   - 損失值: {loss.item():.6f}")

        except Exception as e:
            print(f"❌ 張量形狀 {shape} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def test_combined_regularization_dimensions():
    """測試組合正則化技術的維度處理"""
    print("\n" + "=" * 60)
    print("測試組合正則化技術維度處理")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建複雜的模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.linear1 = nn.Linear(128 * 8 * 8, 256)
            self.linear2 = nn.Linear(256, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.adaptive_avg_pool2d(x, (8, 8))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    try:
        model = TestModel().to(device)

        # 創建組合正則化優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            # 組合所有正則化技術
            edge_suppression=True,
            edge_penalty=0.1,
            spatial_awareness=True,
            frequency_penalty=0.05,
            background_regularization=True,
            lora_rank_penalty=True,
            rank_penalty_strength=0.01,
            # 其他功能
            use_dynamic_adaptation=True,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 創建測試數據
        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randn(4, 10, device=device)

        # 訓練幾步
        for step in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

            print(f"步驟 {step+1}: 損失 = {loss.item():.6f}")

        print("✅ 組合正則化技術維度處理成功")
        return True

    except Exception as e:
        print(f"❌ 組合正則化技術維度處理失敗: {e}")
        return False

def main():
    """主函數"""
    print("🔍 HinaAdaptive 正則化技術維度處理測試")
    print("=" * 60)

    tests = [
        ("基本正則化維度處理", test_regularization_dimensions),
        ("邊緣感知正則化維度處理", test_edge_suppression_dimensions),
        ("空間感知正則化維度處理", test_spatial_awareness_dimensions),
        ("LoRA 低秩正則化維度處理", test_lora_regularization_dimensions),
        ("背景正則化維度處理", test_background_regularization_dimensions),
        ("組合正則化維度處理", test_combined_regularization_dimensions),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 執行 {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} 通過")
                passed += 1
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 異常: {e}")

    # 總結結果
    print("\n" + "=" * 60)
    print(f"📊 測試結果: {passed}/{total} 通過")

    if passed == total:
        print("🎉 所有維度處理測試通過！")
        print("✅ 各種正則化技術都能正確處理不同形狀的張量")
    else:
        print(f"⚠️  {total - passed} 個測試失敗")
        print("❌ 某些正則化技術可能存在維度處理問題")

    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 測試執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)