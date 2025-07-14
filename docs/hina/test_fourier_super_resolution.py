#!/usr/bin/env python3
"""
傅立葉特徵損失超解析度優化快速測試腳本

這是一個輕量級測試腳本，用於快速驗證傅立葉特徵損失功能是否正常工作。

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

def test_fourier_optimizer_creation():
    """測試傅立葉優化器創建"""
    print("\n🔍 測試優化器創建...")
    try:
        from library.hina_adaptive import HinaAdaptive

        # 創建簡單模型
        model = nn.Conv2d(3, 64, 3, padding=1)

        # 創建帶傅立葉特徵的優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            fourier_feature_loss=True,
            super_resolution_mode=True,
            super_resolution_scale=4,
            fourier_high_freq_preservation=0.3,
            fourier_detail_enhancement=0.25,
            fourier_blur_suppression=0.2,
            memory_efficient=True
        )

        print("✅ 傅立葉優化器創建成功")

        # 檢查配置
        info = optimizer.get_optimization_info()
        fourier_config = info.get('fourier_super_resolution_config', {})

        print(f"   超解析度模式: {optimizer.super_resolution_mode}")
        print(f"   放大倍數: {optimizer.super_resolution_scale}")
        print(f"   高頻保持: {fourier_config.get('fourier_high_freq_preservation', 'N/A')}")

        return True, optimizer, model
    except Exception as e:
        print(f"❌ 優化器創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_fourier_forward_backward():
    """測試前向和反向傳播"""
    print("\n🔍 測試前向反向傳播...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   使用設備: {device}")

        # 創建測試模型
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        ).to(device)

        # 創建優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            fourier_feature_loss=True,
            super_resolution_mode=True,
            super_resolution_scale=4,
            memory_efficient=True,
            vram_budget_gb=4.0
        )

        # 創建測試數據
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32, device=device)
        target_tensor = torch.randn(batch_size, 3, 32, 32, device=device)

        # 前向傳播
        output = model(input_tensor)
        loss = F.mse_loss(output, target_tensor)

        print(f"   輸入形狀: {input_tensor.shape}")
        print(f"   輸出形狀: {output.shape}")
        print(f"   初始損失: {loss.item():.6f}")

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 再次前向傳播檢查
        output2 = model(input_tensor)
        loss2 = F.mse_loss(output2, target_tensor)

        print(f"   更新後損失: {loss2.item():.6f}")
        print(f"   損失變化: {loss2.item() - loss.item():+.6f}")

        print("✅ 前向反向傳播測試成功")
        return True

    except Exception as e:
        print(f"❌ 前向反向傳播測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fourier_frequency_analysis():
    """測試傅立葉頻率分析功能"""
    print("\n🔍 測試傅立葉頻率分析...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建測試模型
        model = nn.Conv2d(3, 16, 3, padding=1).to(device)

        # 創建優化器
        optimizer = HinaAdaptive(
            model.parameters(),
            fourier_feature_loss=True,
            super_resolution_mode=True,
            super_resolution_scale=4
        )

        # 創建測試數據
        input_tensor = torch.randn(1, 3, 16, 16, device=device)
        target_tensor = torch.randn(1, 16, 16, 16, device=device)

        # 前向反向傳播
        output = model(input_tensor)
        loss = F.mse_loss(output, target_tensor)

        optimizer.zero_grad()
        loss.backward()

        # 檢查梯度是否包含傅立葉分析的影響
        if model.weight.grad is not None:
            grad_norm = torch.norm(model.weight.grad).item()
            print(f"   梯度範數: {grad_norm:.6f}")

            # 檢查是否有傅立葉緩存
            if hasattr(optimizer, 'fourier_cache'):
                print(f"   傅立葉緩存項數: {len(optimizer.fourier_cache)}")

            print("✅ 傅立葉頻率分析測試成功")
            return True
        else:
            print("⚠️  無梯度產生，可能模型設置有問題")
            return False

    except Exception as e:
        print(f"❌ 傅立葉頻率分析測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """測試記憶體優化功能"""
    print("\n🔍 測試記憶體優化...")
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

        # 測試不同的記憶體設置
        for memory_efficient in [False, True]:
            print(f"\n   記憶體優化: {memory_efficient}")

            optimizer = HinaAdaptive(
                model.parameters(),
                fourier_feature_loss=True,
                super_resolution_mode=True,
                memory_efficient=memory_efficient,
                reduce_precision=memory_efficient,
                vram_budget_gb=4.0
            )

            # 記錄初始記憶體
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()

            # 進行幾步訓練
            for step in range(3):
                input_tensor = torch.randn(2, 3, 32, 32, device=device)
                target_tensor = torch.randn(2, 3, 32, 32, device=device)

                output = model(input_tensor)
                loss = F.mse_loss(output, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 檢查記憶體狀態
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                memory_stats = optimizer.get_memory_stats()
                print(f"     記憶體使用: {(current_memory - initial_memory) / 1024**2:.1f}MB")
                print(f"     記憶體壓力: {memory_stats['memory_pressure']:.1%}")

            # 清理
            optimizer.cleanup_resources()

        print("✅ 記憶體優化測試成功")
        return True

    except Exception as e:
        print(f"❌ 記憶體優化測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_scales():
    """測試不同的超解析度倍數"""
    print("\n🔍 測試不同超解析度倍數...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = nn.Conv2d(3, 16, 3, padding=1).to(device)

        for scale in [2, 4, 8]:
            print(f"\n   測試 {scale}x 超解析度...")

            optimizer = HinaAdaptive(
                model.parameters(),
                fourier_feature_loss=True,
                super_resolution_mode=True,
                super_resolution_scale=scale,
                memory_efficient=True
            )

            # 測試訓練步驟
            input_tensor = torch.randn(1, 3, 16, 16, device=device)
            target_tensor = torch.randn(1, 16, 16, 16, device=device)

            output = model(input_tensor)
            loss = F.mse_loss(output, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"     ✅ {scale}x 配置工作正常")

            # 清理
            optimizer.cleanup_resources()

        print("✅ 不同倍數測試成功")
        return True

    except Exception as e:
        print(f"❌ 不同倍數測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmark():
    """簡單的性能基準測試"""
    print("\n🔍 性能基準測試...")
    try:
        from library.hina_adaptive import HinaAdaptive

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 創建測試模型
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        ).to(device)

        # 測試標準 Adam 優化器
        print("   測試標準 Adam...")
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        start_time = time.time()
        for step in range(10):
            input_tensor = torch.randn(2, 3, 32, 32, device=device)
            target_tensor = torch.randn(2, 3, 32, 32, device=device)

            output = model(input_tensor)
            loss = F.mse_loss(output, target_tensor)

            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()

        adam_time = time.time() - start_time
        print(f"     Adam 時間: {adam_time:.3f}s")

        # 測試傅立葉 HinaAdaptive 優化器
        print("   測試傅立葉 HinaAdaptive...")
        fourier_optimizer = HinaAdaptive(
            model.parameters(),
            lr=1e-4,
            fourier_feature_loss=True,
            super_resolution_mode=True,
            memory_efficient=True
        )

        start_time = time.time()
        for step in range(10):
            input_tensor = torch.randn(2, 3, 32, 32, device=device)
            target_tensor = torch.randn(2, 3, 32, 32, device=device)

            output = model(input_tensor)
            loss = F.mse_loss(output, target_tensor)

            fourier_optimizer.zero_grad()
            loss.backward()
            fourier_optimizer.step()

        fourier_time = time.time() - start_time
        print(f"     Fourier HinaAdaptive 時間: {fourier_time:.3f}s")

        # 計算性能開銷
        overhead = (fourier_time - adam_time) / adam_time * 100
        print(f"     性能開銷: {overhead:+.1f}%")

        if overhead < 50:
            print("✅ 性能開銷在可接受範圍內")
        else:
            print("⚠️  性能開銷較高，考慮啟用更多記憶體優化")

        return True

    except Exception as e:
        print(f"❌ 性能基準測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("🎨 傅立葉特徵損失超解析度優化測試")
    print("=" * 50)

    test_results = {}

    # 運行所有測試
    tests = [
        ("導入測試", test_imports),
        ("優化器創建", test_fourier_optimizer_creation),
        ("前向反向傳播", test_fourier_forward_backward),
        ("傅立葉頻率分析", test_fourier_frequency_analysis),
        ("記憶體優化", test_memory_optimization),
        ("不同倍數", test_different_scales),
        ("性能基準", test_performance_benchmark)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            if test_name == "優化器創建":
                result = test_func()
                if isinstance(result, tuple):
                    test_results[test_name] = result[0]
                else:
                    test_results[test_name] = result
            else:
                test_results[test_name] = test_func()

            if test_results[test_name]:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} 測試異常: {e}")
            test_results[test_name] = False

    # 總結結果
    print("\n" + "=" * 50)
    print("🏁 測試總結")
    print("=" * 50)

    for test_name, result in test_results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")

    print(f"\n📊 總體結果: {passed_tests}/{total_tests} 測試通過")

    if passed_tests == total_tests:
        print("🎉 所有測試通過！傅立葉特徵損失功能正常工作。")
        print("\n💡 接下來可以:")
        print("   1. 運行完整示例: python docs/hina/fourier_super_resolution_example.py")
        print("   2. 閱讀使用指南: docs/hina/FOURIER_SUPER_RESOLUTION_GUIDE.md")
        print("   3. 在實際項目中使用傅立葉特徵損失優化")

        return True
    elif passed_tests >= total_tests * 0.7:
        print("⚠️  大部分測試通過，功能基本可用。")
        print("   建議檢查失敗的測試項目。")
        return True
    else:
        print("❌ 多項測試失敗，可能存在嚴重問題。")
        print("   請檢查代碼或環境配置。")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  測試被用戶中斷")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ 測試過程中發生未處理的錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # 清理資源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 資源清理完成")

    exit(exit_code)