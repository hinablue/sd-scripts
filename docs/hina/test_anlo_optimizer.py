#!/usr/bin/env python3
"""
ANLO 優化器測試腳本

這個腳本用於測試和驗證 ANLO (Alternating Norm LoRA Optimizer) 的功能，
包括記憶體使用、正規化效果、訓練穩定性等。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# 添加項目根目錄到路徑
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from library.hina_ano import ANLO


class MockLoRAModel(nn.Module):
    """
    模擬 LoRA 模型，用於測試優化器
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, num_layers: int = 12):
        super().__init__()

        # 模擬 LoRA 參數
        self.lora_layers = nn.ModuleList()
        for i in range(num_layers):
            # 模擬 lora_up 和 lora_down 參數
            lora_up = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
            lora_down = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.02)

            layer = nn.ModuleDict({
                'lora_up': lora_up,
                'lora_down': lora_down
            })
            self.lora_layers.append(layer)

        # 模擬其他參數
        self.norm_weight = nn.Parameter(torch.ones(input_dim))
        self.norm_bias = nn.Parameter(torch.zeros(input_dim))

        # 輸出層
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # 模擬 LoRA 前向傳播
        for layer in self.lora_layers:
            # LoRA 計算: x + alpha * (x @ lora_down @ lora_up)
            lora_output = x @ layer['lora_down'] @ layer['lora_up']
            x = x + 0.1 * lora_output  # alpha = 0.1

        # 應用 norm
        x = x * self.norm_weight + self.norm_bias

        # 輸出投影
        x = self.output_proj(x)

        return x


def get_memory_usage() -> Dict[str, float]:
    """
    獲取當前記憶體使用情況
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # 物理記憶體
        'vms_mb': memory_info.vms / 1024 / 1024,  # 虛擬記憶體
    }


def create_optimizer_params(model: MockLoRAModel) -> List[Dict[str, Any]]:
    """
    創建模擬的優化器參數組，類似於 LoRA 網絡的 prepare_optimizer_params
    """
    param_groups = []

    # 第一組：LoRA 參數
    lora_params = []
    lora_names = []
    for i, layer in enumerate(model.lora_layers):
        lora_params.extend([layer['lora_up'], layer['lora_down']])
        lora_names.extend([f'lora_layer_{i}_up', f'lora_layer_{i}_down'])

    param_groups.append({
        'params': lora_params,
        'named': lora_names,
        'lr': 1e-4
    })

    # 第二組：Norm 參數
    norm_params = [model.norm_weight, model.norm_bias]
    norm_names = ['norm_weight', 'norm_bias']

    param_groups.append({
        'params': norm_params,
        'named': norm_names,
        'lr': 1e-5
    })

    # 第三組：輸出層參數
    output_params = list(model.output_proj.parameters())
    output_names = ['output_proj.weight', 'output_proj.bias']

    param_groups.append({
        'params': output_params,
        'named': output_names,
        'lr': 1e-4
    })

    return param_groups


def test_memory_efficiency():
    """
    測試記憶體效率
    """
    print("=== 記憶體效率測試 ===")

    # 創建模型
    model = MockLoRAModel()
    model.train()

    # 創建不同的優化器
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=1e-4),
        'AdamW': optim.AdamW(model.parameters(), lr=1e-4),
        'ANLO': ANLO(create_optimizer_params(model), lr=1e-4, verbose=False)
    }

    # 記錄記憶體使用
    memory_results = {}

    for name, optimizer in optimizers.items():
        print(f"\n測試 {name} 優化器...")

        # 清理記憶體
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 記錄初始記憶體
        initial_memory = get_memory_usage()

        # 執行幾步優化
        for step in range(10):
            # 前向傳播
            x = torch.randn(32, 768)
            output = model(x)
            loss = output.mean()

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 記錄最終記憶體
        final_memory = get_memory_usage()

        memory_results[name] = {
            'initial': initial_memory,
            'final': final_memory,
            'increase': {
                'rss_mb': final_memory['rss_mb'] - initial_memory['rss_mb'],
                'vms_mb': final_memory['vms_mb'] - initial_memory['vms_mb']
            }
        }

        print(f"  記憶體增加: {memory_results[name]['increase']['rss_mb']:.2f} MB (RSS)")

        # 如果是 ANLO，獲取詳細記憶體信息
        if name == 'ANLO':
            memory_info = optimizer.get_memory_usage()
            print(f"  優化器記憶體信息:")
            print(f"    總參數: {memory_info['total_parameters']}")
            print(f"    參數記憶體: {memory_info['parameter_memory_mb']:.2f} MB")
            print(f"    梯度記憶體: {memory_info['gradient_memory_mb']:.2f} MB")
            print(f"    優化器狀態記憶體: {memory_info['optimizer_state_memory_mb']:.2f} MB")

    return memory_results


def test_normalization_effect():
    """
    測試正規化效果
    """
    print("\n=== 正規化效果測試 ===")

    # 創建模型和優化器
    model = MockLoRAModel()
    model.train()

    optimizer = ANLO(
        create_optimizer_params(model),
        lr=1e-4,
        normalize_frequency=1,
        verbose=True
    )

    # 記錄梯度範數變化
    grad_norms = []
    normalization_modes = []

    for step in range(20):
        # 前向傳播
        x = torch.randn(16, 768)
        output = model(x)
        loss = output.mean()

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()

        # 記錄梯度範數
        total_grad_norm = 0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += torch.norm(param.grad).item() ** 2
                param_count += 1

        if param_count > 0:
            avg_grad_norm = (total_grad_norm / param_count) ** 0.5
            grad_norms.append(avg_grad_norm)
        else:
            grad_norms.append(0.0)

        # 執行優化步驟
        optimizer.step()

        # 記錄正規化模式
        stats = optimizer.get_normalization_stats()
        normalization_modes.append(stats['normalization_mode'])

        if step % 5 == 0:
            print(f"Step {step}: 梯度範數 = {grad_norms[-1]:.6f}, 正規化模式 = {normalization_modes[-1]}")

    return grad_norms, normalization_modes


def test_training_stability():
    """
    測試訓練穩定性
    """
    print("\n=== 訓練穩定性測試 ===")

    # 創建模型
    model = MockLoRAModel()
    model.train()

    # 創建不同的優化器
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=1e-4),
        'AdamW': optim.AdamW(model.parameters(), lr=1e-4),
        'ANLO': ANLO(create_optimizer_params(model), lr=1e-4, verbose=False)
    }

    # 記錄損失變化
    loss_history = {}

    for name, optimizer in optimizers.items():
        print(f"\n測試 {name} 優化器穩定性...")

        # 重置模型參數
        for param in model.parameters():
            param.data.normal_(0, 0.02)

        losses = []

        for step in range(100):
            # 前向傳播
            x = torch.randn(16, 768)
            output = model(x)
            loss = output.mean()

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % 20 == 0:
                print(f"  Step {step}: Loss = {loss.item():.6f}")

        loss_history[name] = losses

        # 計算穩定性指標
        losses_array = np.array(losses)
        stability_metrics = {
            'final_loss': losses[-1],
            'loss_std': np.std(losses),
            'loss_range': np.max(losses) - np.min(losses),
            'convergence_rate': (losses[0] - losses[-1]) / losses[0] if losses[0] != 0 else 0
        }

        print(f"  穩定性指標:")
        print(f"    最終損失: {stability_metrics['final_loss']:.6f}")
        print(f"    損失標準差: {stability_metrics['loss_std']:.6f}")
        print(f"    損失範圍: {stability_metrics['loss_range']:.6f}")
        print(f"    收斂率: {stability_metrics['convergence_rate']:.2%}")

    return loss_history


def test_adaptive_eps():
    """
    測試自適應 eps 功能
    """
    print("\n=== 自適應 eps 測試 ===")

    model = MockLoRAModel()
    model.train()

    # 測試不同的 eps 設置
    eps_configs = [
        {'adaptive_eps': False, 'eps': 1e-8, 'name': '固定 eps'},
        {'adaptive_eps': True, 'eps': 1e-8, 'name': '自適應 eps'}
    ]

    for config in eps_configs:
        print(f"\n測試 {config['name']}...")

        optimizer = ANLO(
            create_optimizer_params(model),
            lr=1e-4,
            adaptive_eps=config['adaptive_eps'],
            eps=config['eps'],
            verbose=False
        )

        # 記錄 eps 變化
        eps_values = []

        for step in range(50):
            # 前向傳播
            x = torch.randn(16, 768)
            output = model(x)
            loss = output.mean()

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 獲取當前 eps 值
            stats = optimizer.get_normalization_stats()
            current_eps = stats['param_groups']['group_0']['eps']
            eps_values.append(current_eps)

            if step % 10 == 0:
                print(f"  Step {step}: eps = {current_eps:.2e}")

        print(f"  eps 範圍: {min(eps_values):.2e} - {max(eps_values):.2e}")


def plot_results(memory_results, loss_history, grad_norms):
    """
    繪製測試結果
    """
    print("\n=== 生成結果圖表 ===")

    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 記憶體使用比較
    ax1 = axes[0, 0]
    names = list(memory_results.keys())
    memory_increases = [memory_results[name]['increase']['rss_mb'] for name in names]

    bars = ax1.bar(names, memory_increases, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_title('記憶體使用比較')
    ax1.set_ylabel('記憶體增加 (MB)')
    ax1.set_ylim(0, max(memory_increases) * 1.2)

    # 添加數值標籤
    for bar, value in zip(bars, memory_increases):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom')

    # 2. 損失收斂比較
    ax2 = axes[0, 1]
    for name, losses in loss_history.items():
        ax2.plot(losses, label=name, linewidth=2)

    ax2.set_title('損失收斂比較')
    ax2.set_xlabel('訓練步數')
    ax2.set_ylabel('損失值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 梯度範數變化
    ax3 = axes[1, 0]
    ax3.plot(grad_norms, 'b-', linewidth=2, label='梯度範數')
    ax3.set_title('梯度範數變化')
    ax3.set_xlabel('訓練步數')
    ax3.set_ylabel('平均梯度範數')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 正規化模式
    ax4 = axes[1, 1]
    # 創建正規化模式的數值表示
    norm_modes = ['global' if mode == 'global' else 'layer' for mode in grad_norms]
    norm_values = [1 if mode == 'global' else 0 for mode in norm_modes]

    ax4.plot(norm_values, 'r-', linewidth=2, label='正規化模式')
    ax4.set_title('正規化模式變化')
    ax4.set_xlabel('訓練步數')
    ax4.set_ylabel('正規化模式 (1=全局, 0=層級)')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['層級', '全局'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('anlo_optimizer_test_results.png', dpi=300, bbox_inches='tight')
    print("結果圖表已保存為 'anlo_optimizer_test_results.png'")


def main():
    """
    主測試函數
    """
    print("ANLO 優化器測試開始...")
    print("=" * 50)

    # 設置隨機種子以確保可重現性
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 執行各種測試
        memory_results = test_memory_efficiency()
        loss_history = test_training_stability()
        grad_norms, normalization_modes = test_normalization_effect()
        test_adaptive_eps()

        # 生成結果圖表
        plot_results(memory_results, loss_history, grad_norms)

        print("\n" + "=" * 50)
        print("所有測試完成！")

        # 總結
        print("\n=== 測試總結 ===")
        print("✓ 記憶體效率測試完成")
        print("✓ 訓練穩定性測試完成")
        print("✓ 正規化效果測試完成")
        print("✓ 自適應 eps 測試完成")
        print("✓ 結果圖表已生成")

    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()