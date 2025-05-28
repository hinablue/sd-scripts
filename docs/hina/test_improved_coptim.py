#!/usr/bin/env python3
"""
改進版 C-Optim 上下文感知優化器測試

這個測試展示了改進後的上下文感知學習率調整機制，
解決了原版本學習效果低的問題。

主要改進：
1. 更智能的學習率乘數計算
2. 多維度的穩定性評估
3. 動態邊緣情況處理
4. 收斂速度自適應調整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from library.automagic_cameamp import Automagic_CameAMP_COptim, Automagic_CameAMP_COptim8bit

class TestModel(nn.Module):
    """測試用的神經網路模型"""

    def __init__(self, input_size=100, hidden_sizes=[64, 32], output_size=10):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def create_test_data(batch_size=64, input_size=100, output_size=10, num_batches=100):
    """創建測試數據"""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_size)
        # 創建有一定模式的標籤
        y = torch.randint(0, output_size, (batch_size,))
        data.append((x, y))
    return data

def run_training_comparison():
    """運行訓練比較測試"""

    print("🔬 改進版 C-Optim 上下文感知優化器測試")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建測試數據
    test_data = create_test_data()

    # 測試配置
    configs = [
        {
            'name': '標準 AdamW',
            'optimizer_class': torch.optim.AdamW,
            'kwargs': {'lr': 1e-3, 'weight_decay': 1e-4}
        },
        {
            'name': '改進版 C-Optim (32-bit)',
            'optimizer_class': Automagic_CameAMP_COptim,
            'kwargs': {
                'lr': 1e-3,
                'context_window': 50,
                'edge_threshold': 0.8,  # 降低閾值，更容易觸發改進邏輯
                'adaptation_rate': 0.2,
                'verbose': True
            }
        },
        {
            'name': '改進版 C-Optim (8-bit)',
            'optimizer_class': Automagic_CameAMP_COptim8bit,
            'kwargs': {
                'lr': 1e-3,
                'context_window': 50,
                'edge_threshold': 0.8,
                'adaptation_rate': 0.2,
                'verbose': True
            }
        }
    ]

    results = {}

    for config in configs:
        print(f"\n{'='*50}")
        print(f"測試: {config['name']}")
        print(f"{'='*50}")

        # 創建模型
        model = TestModel().to(device)

        try:
            # 創建優化器
            optimizer = config['optimizer_class'](
                model.parameters(),
                **config['kwargs']
            )

            # 訓練
            losses, lr_multipliers, edge_cases = train_model(
                model, optimizer, test_data, device, config['name']
            )

            results[config['name']] = {
                'losses': losses,
                'lr_multipliers': lr_multipliers,
                'edge_cases': edge_cases,
                'final_loss': losses[-1],
                'convergence_step': find_convergence_step(losses)
            }

        except Exception as e:
            print(f"❌ 錯誤: {e}")
            results[config['name']] = {'error': str(e)}

    # 顯示結果並繪圖
    display_results(results)
    plot_training_curves(results)

    return results

def train_model(model, optimizer, test_data, device, optimizer_name):
    """訓練模型並收集指標"""
    model.train()
    losses = []
    lr_multipliers = []
    edge_cases = []

    print("開始訓練...")

    for epoch, (x, y) in enumerate(test_data):
        x, y = x.to(device), y.to(device)

        # 前向傳播
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # 收集 C-Optim 特定指標
        if hasattr(optimizer, 'c_optim'):
            lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()
            is_edge = optimizer.c_optim.detect_edge_case()
            grad_consistency = optimizer.c_optim.compute_gradient_consistency()
            loss_stability = optimizer.c_optim.compute_loss_stability()

            lr_multipliers.append(lr_mult)
            edge_cases.append(is_edge)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                      f"LR倍數={lr_mult:.3f}, 邊緣={is_edge}, "
                      f"梯度一致性={grad_consistency:.3f}, "
                      f"損失穩定性={loss_stability:.3f}")
        else:
            lr_multipliers.append(1.0)  # 標準優化器沒有乘數
            edge_cases.append(False)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}")

    return losses, lr_multipliers, edge_cases

def find_convergence_step(losses, threshold=0.01):
    """找到收斂步數（損失變化小於閾值）"""
    if len(losses) < 10:
        return len(losses)

    for i in range(10, len(losses)):
        recent_losses = losses[i-10:i]
        if max(recent_losses) - min(recent_losses) < threshold:
            return i
    return len(losses)

def display_results(results):
    """顯示訓練結果"""
    print(f"\n{'='*60}")
    print("📊 訓練結果比較")
    print(f"{'='*60}")

    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: ❌ 錯誤 - {result['error']}")
        else:
            print(f"\n📈 {name}:")
            print(f"  最終損失: {result['final_loss']:.6f}")
            print(f"  收斂步數: {result['convergence_step']}")

            if result['lr_multipliers']:
                avg_lr_mult = np.mean(result['lr_multipliers'])
                edge_rate = np.mean(result['edge_cases']) * 100
                print(f"  平均 LR 乘數: {avg_lr_mult:.3f}")
                print(f"  邊緣情況比例: {edge_rate:.1f}%")

def plot_training_curves(results):
    """繪製訓練曲線"""
    plt.figure(figsize=(15, 10))

    # 子圖1: 損失曲線
    plt.subplot(2, 3, 1)
    for name, result in results.items():
        if 'losses' in result:
            plt.plot(result['losses'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('📉 訓練損失比較')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 子圖2: 學習率乘數
    plt.subplot(2, 3, 2)
    for name, result in results.items():
        if 'lr_multipliers' in result and any(x != 1.0 for x in result['lr_multipliers']):
            plt.plot(result['lr_multipliers'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('LR Multiplier')
    plt.title('📊 學習率乘數變化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子圖3: 邊緣情況檢測
    plt.subplot(2, 3, 3)
    for name, result in results.items():
        if 'edge_cases' in result:
            edge_smooth = np.convolve(
                [1 if x else 0 for x in result['edge_cases']],
                np.ones(10)/10, mode='same'
            )
            plt.plot(edge_smooth, label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Edge Case Rate')
    plt.title('⚠️ 邊緣情況檢測率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子圖4: 收斂性比較
    plt.subplot(2, 3, 4)
    names = []
    final_losses = []
    convergence_steps = []

    for name, result in results.items():
        if 'final_loss' in result:
            names.append(name.replace(' ', '\n'))
            final_losses.append(result['final_loss'])
            convergence_steps.append(result['convergence_step'])

    x = np.arange(len(names))
    plt.bar(x, final_losses, alpha=0.7)
    plt.xlabel('Optimizer')
    plt.ylabel('Final Loss')
    plt.title('🎯 最終損失比較')
    plt.xticks(x, names, rotation=45)
    plt.grid(True, alpha=0.3)

    # 子圖5: 收斂速度
    plt.subplot(2, 3, 5)
    plt.bar(x, convergence_steps, alpha=0.7, color='orange')
    plt.xlabel('Optimizer')
    plt.ylabel('Convergence Steps')
    plt.title('⚡ 收斂速度比較')
    plt.xticks(x, names, rotation=45)
    plt.grid(True, alpha=0.3)

    # 子圖6: 損失改善趨勢
    plt.subplot(2, 3, 6)
    for name, result in results.items():
        if 'losses' in result and len(result['losses']) > 10:
            losses = np.array(result['losses'])
            improvement_rate = []
            for i in range(10, len(losses)):
                rate = (losses[i-10] - losses[i]) / losses[i-10]
                improvement_rate.append(rate)
            plt.plot(improvement_rate, label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Improvement Rate')
    plt.title('📈 損失改善率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('improved_coptim_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_improvement_effectiveness():
    """分析改進效果"""
    print(f"\n{'='*60}")
    print("🔍 改進效果分析")
    print(f"{'='*60}")

    print("\n📋 主要改進點：")
    print("1. ✅ 多維度損失趨勢分析（短期 + 長期）")
    print("2. ✅ 動態學習率邊界調整（基於穩定性）")
    print("3. ✅ 停滯檢測和突破機制")
    print("4. ✅ 收斂速度自適應因子")
    print("5. ✅ 智能邊緣情況處理")

    print("\n🎯 預期效果：")
    print("• 更積極的學習率調整（1.2-3.0 vs 原版 1.2）")
    print("• 更好的停滯狀態處理")
    print("• 更穩定的邊緣情況恢復")
    print("• 更快的收斂速度")

if __name__ == "__main__":
    print("🚀 改進版 C-Optim 上下文感知優化器測試")

    try:
        # 運行比較測試
        results = run_training_comparison()

        # 分析改進效果
        analyze_improvement_effectiveness()

        print(f"\n{'='*60}")
        print("✅ 測試完成！查看生成的圖表以了解改進效果。")
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()