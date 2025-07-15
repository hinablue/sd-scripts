#!/usr/bin/env python3
"""
HinaAdaptive 優化器功能測試腳本

測試 HinaAdaptive 優化器的各種功能是否正常工作。

注意：潛在空間檢測和傅立葉特徵損失功能已被移除，因為它們不適用於
SD-Scripts 的 latent space 訓練環境。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加庫路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from library.hina_adaptive import HinaAdaptive

def test_basic_functionality():
    """測試基本功能"""
    print("=" * 60)
    print("基本功能測試")
    print("=" * 60)

    # 創建簡單模型
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    # 創建優化器
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        memory_efficient=True,
        vram_budget_gb=8.0
    )

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"優化器類型: {info['optimizer_type']}")
    print(f"記憶體優化: {info['memory_optimization']['memory_efficient']}")
    print(f"VRAM 預算: {info['memory_optimization']['vram_budget_gb']}GB")

    # 測試簡單訓練
    x = torch.randn(8, 128)
    y = torch.randn(8, 10)

    for step in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    print("✅ 基本功能測試通過")
    return True

def test_regularization_features():
    """測試正則化功能"""
    print("\n" + "=" * 60)
    print("正則化功能測試")
    print("=" * 60)

    # 創建 CNN 模型
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 10)
    )

    # 創建優化器，啟用多種正則化技術
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        edge_suppression=True,
        edge_penalty=0.1,
        spatial_awareness=True,
        frequency_penalty=0.05,
        background_regularization=True,
        lora_rank_penalty=True,
        rank_penalty_strength=0.01,
        memory_efficient=True
    )

    # 檢查正則化功能狀態
    info = optimizer.get_optimization_info()
    print("啟用的正則化技術:")
    if info['features']['edge_suppression']:
        print("  ✅ 邊緣感知正則化")
    if info['features']['spatial_awareness']:
        print("  ✅ 空間感知正則化")
    if info['features']['background_regularization']:
        print("  ✅ 背景正則化")
    if info['features']['lora_rank_penalty']:
        print("  ✅ LoRA 低秩正則化")

    # 測試訓練
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    for step in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    print("✅ 正則化功能測試通過")
    return True

def test_memory_optimization():
    """測試記憶體優化"""
    print("\n" + "=" * 60)
    print("記憶體優化測試")
    print("=" * 60)

    # 創建較大的模型
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    # 創建優化器，啟用記憶體優化
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        memory_efficient=True,
        vram_budget_gb=4.0,
        reduce_precision=True,
        cpu_offload_states=True,
        max_buffer_memory_mb=200
    )

    # 檢查記憶體設置
    info = optimizer.get_optimization_info()
    print(f"記憶體優化配置:")
    print(f"  - 記憶體效率: {info['memory_optimization']['memory_efficient']}")
    print(f"  - VRAM 預算: {info['memory_optimization']['vram_budget_gb']}GB")
    print(f"  - 精度降低: {info['memory_optimization']['reduce_precision']}")
    print(f"  - CPU 狀態卸載: {info['memory_optimization']['cpu_offload_states']}")

    # 測試記憶體統計
    memory_stats = optimizer.get_memory_stats()
    print(f"\n記憶體統計:")
    print(f"  - 記憶體壓力: {memory_stats['memory_pressure']:.2%}")
    print(f"  - 緩衝池記憶體: {memory_stats['buffer_pool_stats']['current_memory_mb']:.2f}MB")

    # 測試訓練
    x = torch.randn(16, 512)
    y = torch.randn(16, 10)

    for step in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    print("✅ 記憶體優化測試通過")
    return True

def test_advanced_features():
    """測試高級功能"""
    print("\n" + "=" * 60)
    print("高級功能測試")
    print("=" * 60)

    # 創建模型
    model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    # 創建優化器，啟用高級功能
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        # 高級功能
        use_dynamic_adaptation=True,
        use_tam=True,
        use_cautious=True,
        use_spd=True,
        use_orthogonal_grad=True,
        use_lr_mask=True,
        dynamic_weight_decay=True,
        # 記憶體優化
        memory_efficient=True
    )

    # 檢查高級功能狀態
    info = optimizer.get_optimization_info()
    print("啟用的高級功能:")
    if info['features']['dynamic_adaptation']:
        print("  ✅ 動態自適應")
    if info['features']['tam']:
        print("  ✅ TAM 阻尼")
    if info['features']['cautious']:
        print("  ✅ 謹慎更新")
    if info['features']['spd']:
        print("  ✅ SPD 正則化")
    if info['features']['orthogonal_grad']:
        print("  ✅ 正交梯度投影")
    if info['features']['lr_mask']:
        print("  ✅ 學習率遮罩")
    if info['features']['dynamic_weight_decay']:
        print("  ✅ 動態權重衰減")

    # 測試訓練
    x = torch.randn(8, 256)
    y = torch.randn(8, 10)

    for step in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    print("✅ 高級功能測試通過")
    return True

def test_removed_features():
    """測試確認已移除的功能"""
    print("\n" + "=" * 60)
    print("已移除功能確認測試")
    print("=" * 60)

    # 創建模型
    model = nn.Linear(64, 32)

    # 測試已移除的 fourier_feature_loss 參數
    try:
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            fourier_feature_loss=True,  # 這個參數已被移除
            memory_efficient=True
        )
        print("❌ 應該引發錯誤但沒有，fourier_feature_loss 參數可能仍然存在")
        return False
    except TypeError as e:
        if "fourier_feature_loss" in str(e):
            print("✅ 確認 fourier_feature_loss 參數已被移除")
        else:
            print(f"❌ 意外的錯誤: {e}")
            return False

    # 測試已移除的 super_resolution_mode 參數
    try:
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            super_resolution_mode=True,  # 這個參數已被移除
            memory_efficient=True
        )
        print("❌ 應該引發錯誤但沒有，super_resolution_mode 參數可能仍然存在")
        return False
    except TypeError as e:
        if "super_resolution_mode" in str(e):
            print("✅ 確認 super_resolution_mode 參數已被移除")
        else:
            print(f"❌ 意外的錯誤: {e}")
            return False

    # 測試已移除的 adaptive_frequency_weighting 參數
    try:
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            adaptive_frequency_weighting=True,  # 這個參數已被移除
            memory_efficient=True
        )
        print("❌ 應該引發錯誤但沒有，adaptive_frequency_weighting 參數可能仍然存在")
        return False
    except TypeError as e:
        if "adaptive_frequency_weighting" in str(e):
            print("✅ 確認 adaptive_frequency_weighting 參數已被移除")
        else:
            print(f"❌ 意外的錯誤: {e}")
            return False

    print("✅ 已移除功能確認測試通過")
    return True

def test_optimization_performance():
    """測試優化器性能"""
    print("\n" + "=" * 60)
    print("優化器性能測試")
    print("=" * 60)

    # 創建模型
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    # 創建優化器
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        memory_efficient=True,
        vram_budget_gb=8.0
    )

    # 測試多次訓練步驟
    x = torch.randn(16, 128)
    y = torch.randn(16, 10)

    initial_loss = None
    final_loss = None

    for step in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if step == 9:
            final_loss = loss.item()

        if step % 3 == 0:
            print(f"步驟 {step+1}: 損失 = {loss.item():.4f}")

    # 計算改善率
    improvement = (initial_loss - final_loss) / initial_loss * 100
    print(f"\n性能結果:")
    print(f"  - 初始損失: {initial_loss:.4f}")
    print(f"  - 最終損失: {final_loss:.4f}")
    print(f"  - 改善率: {improvement:.1f}%")

    if improvement > 0:
        print("✅ 優化器性能測試通過")
        return True
    else:
        print("❌ 優化器性能測試失敗")
        return False

def main():
    """主函數"""
    print("🔍 HinaAdaptive 優化器功能測試")
    print("=" * 60)

    tests = [
        ("基本功能", test_basic_functionality),
        ("正則化功能", test_regularization_features),
        ("記憶體優化", test_memory_optimization),
        ("高級功能", test_advanced_features),
        ("已移除功能確認", test_removed_features),
        ("優化器性能", test_optimization_performance),
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
        print("🎉 所有測試通過！")
        print("✅ HinaAdaptive 優化器功能正常")
    else:
        print(f"⚠️  {total - passed} 個測試失敗")
        print("❌ 某些功能可能存在問題")

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