#!/usr/bin/env python3
"""
HinaAdaptive 正則化技術快速測試腳本

這是一個輕量級測試腳本，用於快速驗證 HinaAdaptive 優化器的
各種正則化技術是否正常工作。

注意：傅立葉特徵損失功能已被移除，因為它不適用於 SD-Scripts
的 latent space 訓練環境。

使用方法:
    python docs/hina/test_fourier_super_resolution.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time

# 添加庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """測試必要的導入是否正常"""
    print("🔍 測試導入...")
    try:
        from library.hina_adaptive import HinaAdaptive
        print("✅ HinaAdaptive 導入成功")
        return True
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        return False

def test_edge_suppression_optimizer():
    """測試邊緣感知正則化優化器創建"""
    print("\n🔍 測試邊緣感知正則化優化器...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建簡單模型
        model = nn.Conv2d(3, 64, 3, padding=1)

        # 創建帶邊緣感知正則化的優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            edge_suppression=True,
            edge_penalty=0.1,
            edge_threshold=0.6,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 驗證優化器配置
        info = optimizer.get_optimization_info()
        assert info['features']['edge_suppression'] == True
        assert info['edge_overfitting_control']['edge_penalty'] == 0.1
        assert info['edge_overfitting_control']['edge_threshold'] == 0.6

        print("✅ 邊緣感知正則化優化器創建成功")
        return True
    except Exception as e:
        print(f"❌ 邊緣感知正則化優化器創建失敗: {e}")
        return False

def test_spatial_awareness_optimizer():
    """測試空間感知正則化優化器創建"""
    print("\n🔍 測試空間感知正則化優化器...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建簡單模型
        model = nn.Conv2d(3, 64, 3, padding=1)

        # 創建帶空間感知正則化的優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            spatial_awareness=True,
            frequency_penalty=0.05,
            detail_preservation=0.8,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 驗證優化器配置
        info = optimizer.get_optimization_info()
        assert info['features']['spatial_awareness'] == True
        assert info['edge_overfitting_control']['frequency_penalty'] == 0.05
        assert info['edge_overfitting_control']['detail_preservation'] == 0.8

        print("✅ 空間感知正則化優化器創建成功")
        return True
    except Exception as e:
        print(f"❌ 空間感知正則化優化器創建失敗: {e}")
        return False

def test_background_regularization_optimizer():
    """測試背景正則化優化器創建"""
    print("\n🔍 測試背景正則化優化器...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建簡單模型
        model = nn.Conv2d(3, 64, 3, padding=1)

        # 創建帶背景正則化的優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            background_regularization=True,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 驗證優化器配置
        info = optimizer.get_optimization_info()
        assert info['features']['background_regularization'] == True

        print("✅ 背景正則化優化器創建成功")
        return True
    except Exception as e:
        print(f"❌ 背景正則化優化器創建失敗: {e}")
        return False

def test_lora_regularization_optimizer():
    """測試 LoRA 低秩正則化優化器創建"""
    print("\n🔍 測試 LoRA 低秩正則化優化器...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建線性模型（適合 LoRA）
        model = nn.Linear(128, 64)

        # 創建帶 LoRA 低秩正則化的優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            lora_rank_penalty=True,
            rank_penalty_strength=0.01,
            low_rank_emphasis=1.2,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 驗證優化器配置
        info = optimizer.get_optimization_info()
        assert info['features']['lora_rank_penalty'] == True
        assert info['edge_overfitting_control']['rank_penalty_strength'] == 0.01
        assert info['edge_overfitting_control']['low_rank_emphasis'] == 1.2

        print("✅ LoRA 低秩正則化優化器創建成功")
        return True
    except Exception as e:
        print(f"❌ LoRA 低秩正則化優化器創建失敗: {e}")
        return False

def test_combined_regularization_optimizer():
    """測試組合正則化優化器創建"""
    print("\n🔍 測試組合正則化優化器...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建簡單模型
        model = nn.Conv2d(3, 64, 3, padding=1)

        # 創建帶組合正則化的優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            # 組合多種正則化技術
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

        # 驗證優化器配置
        info = optimizer.get_optimization_info()
        assert info['features']['edge_suppression'] == True
        assert info['features']['spatial_awareness'] == True
        assert info['features']['background_regularization'] == True
        assert info['features']['lora_rank_penalty'] == True
        assert info['features']['dynamic_adaptation'] == True

        print("✅ 組合正則化優化器創建成功")
        return True
    except Exception as e:
        print(f"❌ 組合正則化優化器創建失敗: {e}")
        return False

def test_edge_suppression_training():
    """測試邊緣感知正則化訓練"""
    print("\n🔍 測試邊緣感知正則化訓練...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建簡單模型
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        ).to(device)

        # 創建邊緣感知正則化優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            edge_suppression=True,
            edge_penalty=0.1,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 創建測試數據
        x = torch.randn(2, 3, 32, 32, device=device)
        y = torch.randn(2, 3, 32, 32, device=device)

        # 訓練幾步
        model.train()
        for step in range(3):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        print("✅ 邊緣感知正則化訓練成功")
        return True
    except Exception as e:
        print(f"❌ 邊緣感知正則化訓練失敗: {e}")
        return False

def test_spatial_awareness_training():
    """測試空間感知正則化訓練"""
    print("\n🔍 測試空間感知正則化訓練...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建簡單模型
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        ).to(device)

        # 創建空間感知正則化優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            spatial_awareness=True,
            frequency_penalty=0.05,
            detail_preservation=0.8,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 創建測試數據
        x = torch.randn(2, 3, 32, 32, device=device)
        y = torch.randn(2, 3, 32, 32, device=device)

        # 訓練幾步
        model.train()
        for step in range(3):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        print("✅ 空間感知正則化訓練成功")
        return True
    except Exception as e:
        print(f"❌ 空間感知正則化訓練失敗: {e}")
        return False

def test_background_regularization_training():
    """測試背景正則化訓練"""
    print("\n🔍 測試背景正則化訓練...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建簡單模型
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        ).to(device)

        # 創建背景正則化優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            background_regularization=True,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 創建測試數據
        x = torch.randn(2, 3, 32, 32, device=device)
        y = torch.randn(2, 3, 32, 32, device=device)

        # 訓練幾步
        model.train()
        for step in range(3):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        print("✅ 背景正則化訓練成功")
        return True
    except Exception as e:
        print(f"❌ 背景正則化訓練失敗: {e}")
        return False

def test_lora_regularization_training():
    """測試 LoRA 低秩正則化訓練"""
    print("\n🔍 測試 LoRA 低秩正則化訓練...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建線性模型（適合 LoRA）
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        ).to(device)

        # 創建 LoRA 低秩正則化優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            lora_rank_penalty=True,
            rank_penalty_strength=0.01,
            memory_efficient=True,
            vram_budget_gb=8.0
        )

        # 創建測試數據
        x = torch.randn(8, 128, device=device)
        y = torch.randn(8, 32, device=device)

        # 訓練幾步
        model.train()
        for step in range(3):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        print("✅ LoRA 低秩正則化訓練成功")
        return True
    except Exception as e:
        print(f"❌ LoRA 低秩正則化訓練失敗: {e}")
        return False

def test_memory_optimization():
    """測試記憶體優化功能"""
    print("\n🔍 測試記憶體優化功能...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建較大的模型
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1)
        ).to(device)

        # 創建記憶體優化優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-3,
            memory_efficient=True,
            vram_budget_gb=4.0,
            reduce_precision=True,
            cpu_offload_states=True,
            max_buffer_memory_mb=200
        )

        # 測試記憶體統計
        memory_stats = optimizer.get_memory_stats()
        assert 'memory_pressure' in memory_stats
        assert 'buffer_pool_stats' in memory_stats

        # 測試記憶體優化設置
        optimizer.optimize_for_vram(target_vram_gb=6.0)

        print("✅ 記憶體優化功能測試成功")
        return True
    except Exception as e:
        print(f"❌ 記憶體優化功能測試失敗: {e}")
        return False

def test_optimizer_info():
    """測試優化器信息獲取"""
    print("\n🔍 測試優化器信息獲取...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建模型
        model = nn.Conv2d(3, 64, 3, padding=1)

        # 創建優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            edge_suppression=True,
            spatial_awareness=True,
            lora_rank_penalty=True,
            memory_efficient=True
        )

        # 獲取優化器信息
        info = optimizer.get_optimization_info()

        # 驗證信息結構
        assert 'optimizer_type' in info
        assert 'features' in info
        assert 'memory_optimization' in info
        assert 'edge_overfitting_control' in info

        # 驗證功能配置
        assert info['features']['edge_suppression'] == True
        assert info['features']['spatial_awareness'] == True
        assert info['features']['lora_rank_penalty'] == True
        assert info['memory_optimization']['memory_efficient'] == True

        print("✅ 優化器信息獲取成功")
        return True
    except Exception as e:
        print(f"❌ 優化器信息獲取失敗: {e}")
        return False

def test_removed_fourier_feature():
    """測試確認 Fourier 特徵損失已被移除"""
    print("\n🔍 測試確認 Fourier 特徵損失已被移除...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建模型
        model = nn.Conv2d(3, 64, 3, padding=1)

        # 嘗試使用已移除的 Fourier 參數，應該引發錯誤
        try:
            optimizer = HinaAdaptive(
                model.parameters(),
                lr=1e-4,
                fourier_feature_loss=True,  # 這個參數已被移除
                memory_efficient=True
            )
            print("❌ 應該引發錯誤但沒有，Fourier 參數可能仍然存在")
            return False
        except TypeError as e:
            if "fourier_feature_loss" in str(e):
                print("✅ 確認 Fourier 特徵損失已被移除")
                return True
            else:
                print(f"❌ 意外的錯誤: {e}")
                return False

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def run_all_tests():
    """運行所有測試"""
    print("🧪 開始運行 HinaAdaptive 正則化技術測試")
    print("=" * 60)

    tests = [
        test_imports,
        test_edge_suppression_optimizer,
        test_spatial_awareness_optimizer,
        test_background_regularization_optimizer,
        test_lora_regularization_optimizer,
        test_combined_regularization_optimizer,
        test_edge_suppression_training,
        test_spatial_awareness_training,
        test_background_regularization_training,
        test_lora_regularization_training,
        test_memory_optimization,
        test_optimizer_info,
        test_removed_fourier_feature,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 測試 {test.__name__} 異常: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 測試結果: {passed} 通過, {failed} 失敗")

    if failed == 0:
        print("🎉 所有測試通過！")
    else:
        print(f"⚠️  {failed} 個測試失敗，請檢查配置")

    return failed == 0

def main():
    """主函數"""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 測試執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()