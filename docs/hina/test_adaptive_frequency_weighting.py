#!/usr/bin/env python3
"""
動態自適應功能測試腳本

測試 HinaAdaptive 優化器的 dynamic_adaptation 功能是否正常工作

注意：傅立葉特徵損失功能已被移除，因為它不適用於 SD-Scripts
的 latent space 訓練環境。
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


def test_dynamic_adaptation():
    """測試動態自適應功能"""
    print("=== 動態自適應功能測試 ===")

    # 創建模型和數據
    model = SimpleTestModel()
    input_data = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 10, (2,))

    # 測試1：啟用動態自適應
    print("\n1. 測試啟用動態自適應的優化器")
    optimizer_adaptive = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        use_dynamic_adaptation=True,  # 啟用動態自適應
        adaptation_strength=1.0,
        relationship_discovery_interval=100,
        importance_decay=0.95,
        compatibility_threshold=0.3,
        memory_efficient=True
    )

    print(f"優化器創建成功，use_dynamic_adaptation = {optimizer_adaptive.use_dynamic_adaptation}")

    # 獲取優化器信息
    info = optimizer_adaptive.get_optimization_info()
    print(f"動態自適應已啟用：{info['features']['dynamic_adaptation']}")
    print(f"自適應強度：{info['adaptation_config']['adaptation_strength']}")
    print(f"關係發現間隔：{info['adaptation_config']['relationship_discovery_interval']}")
    print(f"重要性衰減：{info['adaptation_config']['importance_decay']}")
    print(f"相容性閾值：{info['adaptation_config']['compatibility_threshold']}")

    # 測試2：測試關閉動態自適應
    print("\n2. 測試關閉動態自適應的優化器")
    optimizer_static = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        use_dynamic_adaptation=False,  # 關閉動態自適應
        memory_efficient=True
    )

    print(f"優化器創建成功，use_dynamic_adaptation = {optimizer_static.use_dynamic_adaptation}")

    # 獲取優化器信息
    info_static = optimizer_static.get_optimization_info()
    print(f"動態自適應已關閉：{info_static['features']['dynamic_adaptation']}")

    return True


def test_lr_mask_functionality():
    """測試學習率遮罩功能"""
    print("\n=== 學習率遮罩功能測試 ===")

    model = SimpleTestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 創建優化器，啟用學習率遮罩
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        use_lr_mask=True,  # 啟用學習率遮罩
        lr_bump=3e-6,
        min_lr=1e-7,
        max_lr=1e-3,
        warmup_steps=500,
        memory_efficient=True
    )

    print(f"優化器創建成功，use_lr_mask = {optimizer.use_lr_mask}")

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"學習率遮罩已啟用：{info['features']['lr_mask']}")

    # 測試幾步訓練
    input_data = torch.randn(2, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (2,), device=device)

    print("\n進行幾步訓練測試...")
    for step in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    print("✅ 學習率遮罩功能測試完成")
    return True


def test_edge_suppression_functionality():
    """測試邊緣感知正則化功能"""
    print("\n=== 邊緣感知正則化功能測試 ===")

    model = SimpleTestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 創建優化器，啟用邊緣感知正則化
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        edge_suppression=True,  # 啟用邊緣感知正則化
        edge_penalty=0.1,
        edge_threshold=0.6,
        memory_efficient=True
    )

    print(f"優化器創建成功，edge_suppression = {optimizer.edge_suppression}")

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"邊緣感知正則化已啟用：{info['features']['edge_suppression']}")
    print(f"邊緣懲罰：{info['edge_overfitting_control']['edge_penalty']}")
    print(f"邊緣閾值：{info['edge_overfitting_control']['edge_threshold']}")

    # 測試幾步訓練
    input_data = torch.randn(2, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (2,), device=device)

    print("\n進行幾步訓練測試...")
    for step in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    print("✅ 邊緣感知正則化功能測試完成")
    return True


def test_memory_optimization_functionality():
    """測試記憶體優化功能"""
    print("\n=== 記憶體優化功能測試 ===")

    model = SimpleTestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 創建優化器，啟用記憶體優化
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        memory_efficient=True,  # 啟用記憶體優化
        vram_budget_gb=8.0,
        cpu_offload_states=True,
        reduce_precision=True,
        adaptive_features=True,
        max_buffer_memory_mb=500
    )

    print(f"優化器創建成功，memory_efficient = {optimizer.memory_efficient}")

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"記憶體優化已啟用：{info['memory_optimization']['memory_efficient']}")
    print(f"VRAM 預算：{info['memory_optimization']['vram_budget_gb']}GB")
    print(f"CPU 狀態卸載：{info['memory_optimization']['cpu_offload_states']}")
    print(f"精度降低：{info['memory_optimization']['reduce_precision']}")

    # 測試記憶體統計
    memory_stats = optimizer.get_memory_stats()
    print(f"\n記憶體統計：")
    print(f"記憶體壓力：{memory_stats['memory_pressure']:.2%}")
    print(f"緩衝池記憶體：{memory_stats['buffer_pool_stats']['current_memory_mb']:.2f}MB")

    # 測試記憶體優化
    optimizer.optimize_for_vram(target_vram_gb=6.0)
    print("✅ 記憶體優化功能測試完成")
    return True


def test_regularization_combination():
    """測試正則化技術組合"""
    print("\n=== 正則化技術組合測試 ===")

    model = SimpleTestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 創建優化器，組合多種正則化技術
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=0.001,
        # 組合多種正則化技術
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

    print("優化器創建成功，組合多種正則化技術")

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"啟用的正則化技術：")
    for feature, enabled in info['features'].items():
        if enabled:
            print(f"  ✅ {feature}")

    # 測試幾步訓練
    input_data = torch.randn(2, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (2,), device=device)

    print("\n進行幾步訓練測試...")
    for step in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    print("✅ 正則化技術組合測試完成")
    return True


def main():
    """主測試函數"""
    print("🧪 HinaAdaptive 動態自適應功能測試")
    print("=" * 50)

    tests = [
        ("動態自適應功能", test_dynamic_adaptation),
        ("學習率遮罩功能", test_lr_mask_functionality),
        ("邊緣感知正則化功能", test_edge_suppression_functionality),
        ("記憶體優化功能", test_memory_optimization_functionality),
        ("正則化技術組合", test_regularization_combination),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 執行 {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} 通過")
                passed += 1
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 異常: {e}")

    # 總結結果
    print("\n" + "=" * 50)
    print(f"📊 測試結果: {passed}/{total} 通過")

    if passed == total:
        print("🎉 所有測試通過！")
    else:
        print(f"⚠️  {total - passed} 個測試失敗")

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