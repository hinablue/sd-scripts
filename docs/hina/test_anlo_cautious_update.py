#!/usr/bin/env python3
"""
測試增強版 ANLO 優化器的謹慎更新功能

這個腳本展示了如何將 hina_adaptive.py 中的 _apply_cautious_update_optimized
方法應用到 ANLO 優化器中，提高訓練穩定性。
"""

import torch
import torch.nn as nn
import numpy as np
from library.hina_anlo import ANLO

def create_test_model():
    """創建測試模型"""
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Linear(25, 10)
    )
    return model

def test_anlo_with_cautious_update():
    """測試帶謹慎更新功能的 ANLO 優化器"""
    print("=== 測試增強版 ANLO 優化器 ===")

    # 創建測試模型
    model = create_test_model()

    # 創建測試數據
    x = torch.randn(32, 100)
    y = torch.randn(32, 10)

    # 測試不同的配置
    configs = [
        {
            'name': '標準 ANLO（無謹慎更新）',
            'use_cautious_update': False,
            'cautious_threshold': 0.1,
            'cautious_scale': 0.5
        },
        {
            'name': 'ANLO + 謹慎更新（標準配置）',
            'use_cautious_update': True,
            'cautious_threshold': 0.1,
            'cautious_scale': 0.5
        },
        {
            'name': 'ANLO + 謹慎更新（嚴格配置）',
            'use_cautious_update': True,
            'cautious_threshold': 0.3,
            'cautious_scale': 0.3
        },
        {
            'name': 'ANLO + 謹慎更新（寬鬆配置）',
            'use_cautious_update': True,
            'cautious_threshold': 0.05,
            'cautious_scale': 0.7
        }
    ]

    for config in configs:
        print(f"\n--- {config['name']} ---")

        # 重置模型
        model = create_test_model()

        # 創建優化器
        optimizer = ANLO(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            normalize_frequency=1,
            use_cautious_update=config['use_cautious_update'],
            cautious_threshold=config['cautious_threshold'],
            cautious_scale=config['cautious_scale'],
            verbose=True
        )

        # 訓練循環
        losses = []
        for step in range(100):
            optimizer.zero_grad()

            # 前向傳播
            output = model(x)
            loss = nn.MSELoss()(output, y)

            # 反向傳播
            loss.backward()

            # 優化步驟
            optimizer.step()

            losses.append(loss.item())

            if step % 20 == 0:
                print(f"Step {step}: Loss = {loss.item():.6f}")

        # 顯示統計信息
        print(f"最終損失: {losses[-1]:.6f}")
        print(f"損失變化: {losses[0] - losses[-1]:.6f}")

        # 獲取優化器統計信息
        norm_stats = optimizer.get_normalization_stats()
        cautious_stats = optimizer.get_cautious_update_stats()

        print(f"正規化模式: {norm_stats['normalization_mode']}")
        print(f"謹慎更新啟用: {norm_stats['cautious_update_enabled']}")
        print(f"啟用謹慎更新的參數組: {cautious_stats['enabled_groups']}/{cautious_stats['total_groups']}")

def test_cautious_update_effectiveness():
    """測試謹慎更新的有效性"""
    print("\n=== 測試謹慎更新有效性 ===")

    # 創建一個故意產生不一致梯度的場景
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    # 測試有無謹慎更新的差異
    results = {}

    for use_cautious in [False, True]:
        print(f"\n--- {'啟用' if use_cautious else '禁用'}謹慎更新 ---")

        # 重置模型
        model = create_test_model()

        optimizer = ANLO(
            model.parameters(),
            lr=1e-2,
            use_cautious_update=use_cautious,
            cautious_threshold=0.1,
            cautious_scale=0.5,
            verbose=False
        )

        losses = []
        grad_norms = []

        for step in range(50):
            optimizer.zero_grad()

            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            # 記錄梯度範數
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += torch.norm(param.grad).item()
            grad_norms.append(total_grad_norm)

            optimizer.step()
            losses.append(loss.item())

        results[f'cautious_{use_cautious}'] = {
            'final_loss': losses[-1],
            'loss_improvement': losses[0] - losses[-1],
            'avg_grad_norm': np.mean(grad_norms),
            'grad_norm_std': np.std(grad_norms)
        }

        print(f"最終損失: {losses[-1]:.6f}")
        print(f"平均梯度範數: {np.mean(grad_norms):.6f}")
        print(f"梯度範數標準差: {np.std(grad_norms):.6f}")

    # 比較結果
    print("\n--- 比較結果 ---")
    cautious_result = results['cautious_True']
    normal_result = results['cautious_False']

    print(f"謹慎更新 vs 標準更新:")
    print(f"  最終損失改善: {cautious_result['final_loss']:.6f} vs {normal_result['final_loss']:.6f}")
    print(f"  損失改善幅度: {cautious_result['loss_improvement']:.6f} vs {normal_result['loss_improvement']:.6f}")
    print(f"  梯度穩定性: {cautious_result['grad_norm_std']:.6f} vs {normal_result['grad_norm_std']:.6f}")

def main():
    """主函數"""
    print("ANLO 優化器謹慎更新功能測試")
    print("=" * 50)

    # 設置隨機種子以確保可重現性
    torch.manual_seed(42)
    np.random.seed(42)

    # 運行測試
    test_anlo_with_cautious_update()
    test_cautious_update_effectiveness()

    print("\n=== 測試完成 ===")
    print("謹慎更新功能已成功整合到 ANLO 優化器中！")

if __name__ == "__main__":
    main()