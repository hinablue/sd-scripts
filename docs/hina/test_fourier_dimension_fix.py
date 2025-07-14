#!/usr/bin/env python3
"""
傅立葉特徵處理維度修復測試

測試不同形狀的張量是否能正確處理傅立葉特徵，確保沒有維度不匹配錯誤。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加庫路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from library.hina_adaptive import HinaAdaptive

def test_fourier_feature_dimensions():
    """測試傅立葉特徵處理的維度匹配"""
    print("=" * 60)
    print("測試傅立葉特徵處理維度修復")
    print("=" * 60)

    # 創建各種形狀的測試張量
    test_shapes = [
        (8, 8),           # 2D: 全連接層權重
        (1, 8, 8),        # 3D: 單通道卷積
        (64, 32, 3, 3),   # 4D: 2D卷積權重（原錯誤來源）
        (128, 64, 5, 5),  # 4D: 較大的卷積核
        (256, 128, 7, 7), # 4D: 更大的卷積核
        (16, 32, 1, 1),   # 4D: 1x1卷積
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

            # 創建優化器，啟用傅立葉特徵損失
            optimizer = HinaAdaptive(
                [param],
                lr=1e-3,
                fourier_feature_loss=True,
                super_resolution_mode=True,
                super_resolution_scale=4,
                adaptive_frequency_weighting=True,
                fourier_high_freq_preservation=0.3,
                fourier_detail_enhancement=0.2,
                fourier_blur_suppression=0.15,
                texture_coherence_penalty=0.1,
                frequency_domain_lr_scaling=True
            )

            # 創建模擬損失和梯度
            dummy_output = torch.sum(param ** 2)
            dummy_output.backward()

            print(f"  ✓ 參數形狀: {param.shape}")
            print(f"  ✓ 梯度形狀: {param.grad.shape}")
            print(f"  ✓ 梯度範數: {torch.norm(param.grad).item():.6f}")

            # 執行優化步驟（這裡會觸發傅立葉特徵處理）
            optimizer.step()

            print(f"  ✓ 優化步驟成功完成")
            print(f"  ✓ 更新後參數範數: {torch.norm(param).item():.6f}")

            # 檢查參數是否有效更新
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  ✗ 警告：參數包含 NaN 或 Inf 值")
                all_tests_passed = False
            else:
                print(f"  ✓ 參數值正常")

            # 清理梯度
            optimizer.zero_grad()

        except Exception as e:
            print(f"  ✗ 測試失敗: {e}")
            print(f"     錯誤類型: {type(e).__name__}")
            all_tests_passed = False

            # 打印詳細的錯誤堆棧
            import traceback
            print("     詳細錯誤信息:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 所有測試通過！傅立葉特徵處理維度修復成功！")
    else:
        print("❌ 部分測試失敗，請檢查修復是否完整。")
    print("=" * 60)

    return all_tests_passed

def test_frequency_mask_consistency():
    """測試頻率掩膜的一致性"""
    print("\n" + "=" * 60)
    print("測試頻率掩膜一致性")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建一個4D卷積權重（之前出錯的情況）
    param = torch.randn(64, 32, 8, 8, device=device, requires_grad=True)

    # 創建優化器實例
    optimizer = HinaAdaptive(
        [param],
        lr=1e-3,
        fourier_feature_loss=True,
        super_resolution_mode=True,
    )

    # 獲取參數ID和緊湊狀態
    param_id = id(param)
    group_metadata = optimizer.param_groups_metadata[0]
    compact_state = group_metadata['compact_states'][param_id]

    try:
        # 模擬梯度
        dummy_loss = torch.sum(param ** 2)
        dummy_loss.backward()

        grad = param.grad
        print(f"原始梯度形狀: {grad.shape}")

        # 直接調用傅立葉特徵計算方法
        fourier_features = optimizer._compute_fourier_features(grad, f"test_{param_id}", compact_state)

        print(f"傅立葉特徵檢查:")
        print(f"  - magnitude 形狀: {fourier_features['magnitude'].shape}")
        print(f"  - 頻率掩膜形狀:")
        print(f"    - low_freq_mask: {fourier_features['low_freq_mask'].shape}")
        print(f"    - mid_freq_mask: {fourier_features['mid_freq_mask'].shape}")
        print(f"    - high_freq_mask: {fourier_features['high_freq_mask'].shape}")
        print(f"  - 批次大小: {fourier_features['batch_size']}")
        print(f"  - 是否多維: {fourier_features['is_multidim']}")
        print(f"  - 原始形狀: {fourier_features['original_shape']}")

        # 測試各種傅立葉調整
        print(f"\n測試傅立葉調整方法:")

        # 高頻保持
        high_freq_adj = optimizer._compute_high_freq_preservation(grad, fourier_features, compact_state)
        print(f"  ✓ 高頻保持調整形狀: {high_freq_adj.shape}")

        # 模糊抑制
        blur_adj = optimizer._compute_blur_suppression(grad, fourier_features, compact_state)
        print(f"  ✓ 模糊抑制調整形狀: {blur_adj.shape}")

        # 紋理一致性懲罰
        texture_penalty = optimizer._compute_texture_coherence_penalty(grad, fourier_features, compact_state)
        print(f"  ✓ 紋理一致性懲罰形狀: {texture_penalty.shape}")

        # 超解析度調整
        sr_adj = optimizer._compute_super_resolution_adjustment(grad, fourier_features, compact_state)
        print(f"  ✓ 超解析度調整形狀: {sr_adj.shape}")

        # 檢查所有調整是否與原始梯度形狀匹配
        adjustments = [high_freq_adj, blur_adj, texture_penalty, sr_adj]
        for i, adj in enumerate(adjustments):
            if adj.shape != grad.shape:
                print(f"  ✗ 調整 {i} 形狀不匹配: {adj.shape} vs {grad.shape}")
                return False

        print(f"  ✓ 所有調整形狀都與原始梯度匹配")

        return True

    except Exception as e:
        print(f"✗ 頻率掩膜一致性測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_error_case():
    """測試具體的錯誤案例：[8, 8] mask vs [1, 8, 8] tensor"""
    print("\n" + "=" * 60)
    print("測試具體錯誤案例修復")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 重現原始錯誤的確切條件
    param = torch.randn(1, 8, 8, device=device, requires_grad=True)

    print(f"測試參數形狀: {param.shape}")

    try:
        # 創建優化器
        optimizer = HinaAdaptive(
            [param],
            lr=1e-3,
            fourier_feature_loss=True,
            super_resolution_mode=True,
            fourier_high_freq_preservation=0.3,
        )

        # 創建梯度
        loss = torch.sum(param ** 2)
        loss.backward()

        print(f"梯度形狀: {param.grad.shape}")

        # 這應該不會再產生 IndexError
        optimizer.step()

        print("✓ 具體錯誤案例修復成功！")
        return True

    except IndexError as e:
        print(f"✗ IndexError 仍然存在: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("傅立葉特徵處理維度修復測試開始...")

    test1_passed = test_fourier_feature_dimensions()
    test2_passed = test_frequency_mask_consistency()
    test3_passed = test_specific_error_case()

    print("\n" + "=" * 80)
    print("最終測試結果匯總:")
    print("=" * 80)
    print(f"基本傅立葉特徵處理: {'✓ 通過' if test1_passed else '✗ 失敗'}")
    print(f"頻率掩膜一致性測試: {'✓ 通過' if test2_passed else '✗ 失敗'}")
    print(f"具體錯誤案例修復: {'✓ 通過' if test3_passed else '✗ 失敗'}")

    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 所有測試都通過！維度不匹配問題已完全修復！")
        exit(0)
    else:
        print("\n❌ 部分測試失敗，需要進一步調試。")
        exit(1)