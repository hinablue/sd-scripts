#!/usr/bin/env python3
"""
多維張量正則化處理測試腳本

測試 HinaAdaptive 優化器對多維張量的正則化處理是否正常工作。

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

def test_multidim_regularization():
    """測試多維張量的正則化處理"""
    print("=" * 60)
    print("多維張量正則化處理測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 測試不同維度的張量
    test_cases = [
        ("1D 張量", torch.randn(128, device=device, requires_grad=True)),
        ("2D 張量", torch.randn(64, 32, device=device, requires_grad=True)),
        ("3D 張量", torch.randn(16, 8, 8, device=device, requires_grad=True)),
        ("4D 張量", torch.randn(32, 16, 3, 3, device=device, requires_grad=True)),
        ("5D 張量", torch.randn(8, 4, 2, 2, 2, device=device, requires_grad=True)),
    ]

    all_tests_passed = True

    for test_name, tensor in test_cases:
        print(f"\n測試 {test_name} (形狀: {tensor.shape}):")
        print("-" * 40)

        try:
            # 創建優化器，啟用多種正則化技術
            optimizer = HinaAdaptive(
                [tensor],
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

            # 創建目標張量
            target = torch.randn_like(tensor)

            # 執行前向和反向傳播
            optimizer.zero_grad()
            output = tensor * 2.0
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            print(f"✅ {test_name} 處理成功")
            print(f"   - 張量形狀: {tensor.shape}")
            print(f"   - 元素數量: {tensor.numel()}")
            print(f"   - 梯度範數: {torch.norm(tensor.grad).item():.6f}")
            print(f"   - 損失值: {loss.item():.6f}")

        except Exception as e:
            print(f"❌ {test_name} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def test_conv_layers_regularization():
    """測試卷積層的正則化處理"""
    print("\n" + "=" * 60)
    print("卷積層正則化處理測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 測試不同類型的卷積層
    test_layers = [
        ("Conv1d", nn.Conv1d(16, 32, 3, padding=1)),
        ("Conv2d", nn.Conv2d(16, 32, 3, padding=1)),
        ("Conv3d", nn.Conv3d(16, 32, 3, padding=1)),
        ("ConvTranspose2d", nn.ConvTranspose2d(32, 16, 3, padding=1)),
        ("DepthwiseConv2d", nn.Conv2d(16, 16, 3, padding=1, groups=16)),
    ]

    all_tests_passed = True

    for layer_name, layer in test_layers:
        print(f"\n測試 {layer_name}:")
        print("-" * 30)

        try:
            layer = layer.to(device)

            # 創建優化器
            optimizer = HinaAdaptive(
                layer.parameters(),
                lr=1e-3,
                edge_suppression=True,
                edge_penalty=0.1,
                spatial_awareness=True,
                frequency_penalty=0.05,
                memory_efficient=True
            )

            # 創建測試數據
            if isinstance(layer, nn.Conv1d):
                x = torch.randn(2, 16, 32, device=device)
            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                x = torch.randn(2, 16, 32, 32, device=device)
            elif isinstance(layer, nn.Conv3d):
                x = torch.randn(2, 16, 16, 16, 16, device=device)
            else:
                x = torch.randn(2, 16, 32, 32, device=device)

            # 執行訓練步驟
            optimizer.zero_grad()
            output = layer(x)
            loss = torch.mean(output ** 2)
            loss.backward()
            optimizer.step()

            print(f"✅ {layer_name} 處理成功")
            print(f"   - 權重形狀: {layer.weight.shape}")
            print(f"   - 輸入形狀: {x.shape}")
            print(f"   - 輸出形狀: {output.shape}")
            print(f"   - 損失值: {loss.item():.6f}")

        except Exception as e:
            print(f"❌ {layer_name} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def test_complex_model_regularization():
    """測試複雜模型的正則化處理"""
    print("\n" + "=" * 60)
    print("複雜模型正則化處理測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建複雜模型
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 多種類型的層
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            self.linear1 = nn.Linear(128 * 4 * 4, 256)
            self.linear2 = nn.Linear(256, 128)
            self.linear3 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.conv3(x))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.linear1(x))
            x = self.dropout(x)
            x = torch.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    try:
        model = ComplexModel().to(device)

        # 創建優化器，啟用所有正則化技術
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            # 所有正則化技術
            edge_suppression=True,
            edge_penalty=0.1,
            spatial_awareness=True,
            frequency_penalty=0.05,
            background_regularization=True,
            lora_rank_penalty=True,
            rank_penalty_strength=0.01,
            # 動態自適應
            use_dynamic_adaptation=True,
            # 記憶體優化
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 創建測試數據
        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)

        print("模型參數統計:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - 總參數數量: {total_params:,}")
        print(f"   - 可訓練參數數量: {trainable_params:,}")

        # 執行多步訓練
        print("\n執行訓練步驟:")
        for step in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

        # 檢查優化器狀態
        info = optimizer.get_optimization_info()
        print(f"\n優化器狀態:")
        print(f"   - 啟用的正則化技術: {sum(info['features'].values())}")
        print(f"   - 動態自適應: {info['features']['dynamic_adaptation']}")
        print(f"   - 記憶體優化: {info['memory_optimization']['memory_efficient']}")

        # 檢查記憶體使用
        memory_stats = optimizer.get_memory_stats()
        print(f"   - 記憶體壓力: {memory_stats['memory_pressure']:.2%}")

        print("✅ 複雜模型正則化處理測試成功")
        return True

    except Exception as e:
        print(f"❌ 複雜模型正則化處理測試失敗: {e}")
        return False

def test_batch_dimension_handling():
    """測試批次維度處理"""
    print("\n" + "=" * 60)
    print("批次維度處理測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 測試不同批次大小
    batch_sizes = [1, 2, 4, 8, 16]

    all_tests_passed = True

    for batch_size in batch_sizes:
        print(f"\n測試批次大小 {batch_size}:")
        print("-" * 30)

        try:
            # 創建卷積層
            layer = nn.Conv2d(3, 16, 3, padding=1).to(device)

            # 創建優化器
            optimizer = HinaAdaptive(
                layer.parameters(),
                lr=1e-3,
                edge_suppression=True,
                background_regularization=True,
                memory_efficient=True
            )

            # 創建測試數據
            x = torch.randn(batch_size, 3, 32, 32, device=device)
            y = torch.randn(batch_size, 16, 32, 32, device=device)

            # 執行訓練步驟
            optimizer.zero_grad()
            output = layer(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

            print(f"✅ 批次大小 {batch_size} 處理成功")
            print(f"   - 輸入形狀: {x.shape}")
            print(f"   - 輸出形狀: {output.shape}")
            print(f"   - 損失值: {loss.item():.6f}")

        except Exception as e:
            print(f"❌ 批次大小 {batch_size} 處理失敗: {e}")
            all_tests_passed = False

    return all_tests_passed

def main():
    """主函數"""
    print("🔍 多維張量正則化處理測試")
    print("=" * 60)

    tests = [
        ("多維張量正則化處理", test_multidim_regularization),
        ("卷積層正則化處理", test_conv_layers_regularization),
        ("複雜模型正則化處理", test_complex_model_regularization),
        ("批次維度處理", test_batch_dimension_handling),
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
        print("🎉 所有多維張量正則化處理測試通過！")
        print("✅ 各種正則化技術都能正確處理多維張量")
    else:
        print(f"⚠️  {total - passed} 個測試失敗")
        print("❌ 某些正則化技術可能存在多維張量處理問題")

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