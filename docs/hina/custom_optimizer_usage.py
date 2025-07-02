#!/usr/bin/env python3
"""
Custom AdamW Optimizer 使用示例
Example usage of the Custom AdamW Optimizer with enhanced features

這個腳本展示如何在 sd-scripts 中使用新的 CustomAdamWOptimizer
This script demonstrates how to use the new CustomAdamWOptimizer in sd-scripts
"""

import torch
import torch.nn as nn
import sys
import os
import argparse

# 添加 library 路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'library'))

from custom_adamw_optimizer import CustomAdamWOptimizer, get_custom_adamw_optimizer


def create_test_model():
    """創建一個測試模型來模擬 LoRA 結構"""
    class TestLoRAModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 模擬 LoRA 結構
            self.lora_down = nn.Linear(512, 32, bias=False)  # LoRA A 矩陣
            self.lora_up = nn.Linear(32, 512, bias=False)    # LoRA B 矩陣

            # 其他參數
            self.regular_layer = nn.Linear(512, 256)
            self.output_layer = nn.Linear(256, 10)

            # 為參數添加名稱（模擬真實情況）
            self.lora_down.weight.param_name = "test.lora_down.weight"
            self.lora_up.weight.param_name = "test.lora_up.weight"
            self.regular_layer.weight.param_name = "test.regular_layer.weight"
            self.output_layer.weight.param_name = "test.output_layer.weight"

        def forward(self, x):
            # LoRA 前向傳播
            lora_out = self.lora_up(self.lora_down(x))

            # 正常前向傳播
            x = self.regular_layer(x)
            x = torch.relu(x)
            x = self.output_layer(x)

            return x + lora_out  # 添加 LoRA 輸出

    return TestLoRAModel()


def test_basic_functionality():
    """測試優化器基本功能"""
    print("=== 測試基本功能 ===")

    model = create_test_model()

    # 基本配置
    optimizer_kwargs = {
        'use_spd': True,
        'use_cautious': True,
        'use_adopt_stability': True,
        'use_grams': True,
        'use_agr': True,
        'use_tam': True,
        'use_alora': True,
        'dynamic_weight_decay': True
    }

    optimizer = get_custom_adamw_optimizer(
        model.parameters(),
        lr=1e-3,
        optimizer_kwargs=optimizer_kwargs
    )

    # 顯示優化器信息
    opt_info = optimizer.get_optimization_info()
    print(f"優化器類型: {opt_info['optimizer_type']}")
    print(f"總參數數量: {opt_info['total_params']}")
    print(f"啟用功能: {opt_info['features']}")
    print(f"LoRA 統計: {opt_info['lora_stats']}")

    # 測試一個簡單的訓練步驟
    x = torch.randn(32, 512)
    target = torch.randn(32, 10)

    output = model(x)
    loss = nn.MSELoss()(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"測試訓練步驟完成，損失: {loss.item():.6f}")
    print("✓ 基本功能測試通過\n")


def test_feature_combinations():
    """測試不同功能組合"""
    print("=== 測試功能組合 ===")

    model = create_test_model()

    # 測試不同的功能組合
    test_configs = [
        {
            'name': '僅 SPD + ALoRA',
            'config': {
                'use_spd': True,
                'use_cautious': False,
                'use_adopt_stability': False,
                'use_grams': False,
                'use_agr': False,
                'use_tam': False,
                'use_alora': True,
                'dynamic_weight_decay': False
            }
        },
        {
            'name': 'ADOPT + TAM + AGR',
            'config': {
                'use_spd': False,
                'use_cautious': False,
                'use_adopt_stability': True,
                'use_grams': False,
                'use_agr': True,
                'use_tam': True,
                'use_alora': False,
                'dynamic_weight_decay': False
            }
        },
        {
            'name': '全部功能',
            'config': {
                'use_spd': True,
                'use_cautious': True,
                'use_adopt_stability': True,
                'use_grams': True,
                'use_agr': True,
                'use_tam': True,
                'use_alora': True,
                'dynamic_weight_decay': True
            }
        }
    ]

    for test_config in test_configs:
        print(f"測試配置: {test_config['name']}")

        optimizer = CustomAdamWOptimizer(
            model.parameters(),
            lr=1e-3,
            **test_config['config']
        )

        # 運行幾個訓練步驟
        losses = []
        for step in range(5):
            x = torch.randn(16, 512)
            target = torch.randn(16, 10)

            output = model(x)
            loss = nn.MSELoss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"  平均損失: {sum(losses)/len(losses):.6f}")
        print(f"  ✓ 配置測試通過\n")


def benchmark_performance():
    """性能基準測試"""
    print("=== 性能基準測試 ===")

    model = create_test_model()

    # 比較不同優化器的性能
    optimizers = {
        'PyTorch AdamW': torch.optim.AdamW(model.parameters(), lr=1e-3),
        'CustomAdamW (最小功能)': CustomAdamWOptimizer(
            model.parameters(), lr=1e-3,
            use_spd=False, use_cautious=False, use_orthogonal_grad=False,
            use_adopt_stability=False, use_grams=False, use_agr=False,
            use_tam=False, use_alora=False, dynamic_weight_decay=False
        ),
        'CustomAdamW (全功能)': CustomAdamWOptimizer(
            model.parameters(), lr=1e-3,
            use_spd=True, use_cautious=True, use_orthogonal_grad=True,
            use_adopt_stability=True, use_grams=True, use_agr=True,
            use_tam=True, use_alora=True, dynamic_weight_decay=True
        ),
    }

    import time

    for opt_name, optimizer in optimizers.items():
        print(f"測試 {opt_name}...")

        # 重置模型
        model = create_test_model()

        start_time = time.time()

        for step in range(100):
            x = torch.randn(32, 512)
            target = torch.randn(32, 10)

            output = model(x)
            loss = nn.MSELoss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        print(f"  100 步訓練時間: {elapsed_time:.3f} 秒")
        print(f"  平均每步時間: {elapsed_time/100*1000:.2f} 毫秒\n")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Custom AdamW Optimizer 測試腳本")
    parser.add_argument('--test', choices=['basic', 'features', 'benchmark', 'all'],
                       default='all', help='要運行的測試類型')

    args = parser.parse_args()

    print("Custom AdamW Optimizer 測試腳本")
    print("=" * 50)

    if args.test in ['basic', 'all']:
        test_basic_functionality()

    if args.test in ['features', 'all']:
        test_feature_combinations()

    if args.test in ['benchmark', 'all']:
        benchmark_performance()

    print("所有測試完成！")


if __name__ == "__main__":
    main()