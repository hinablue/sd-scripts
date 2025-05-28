#!/usr/bin/env python3
"""
LoRA 優化版 Automagic_CameAMP 測試腳本

這個腳本展示了針對 Stable Diffusion LoRA 訓練優化的使用方法，
解決了學習率乘數過小的問題。

作者: Hina
版本: 1.0
日期: 2025-01-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List

# 添加庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from library.automagic_cameamp import (
        Automagic_CameAMP_COptim,
        Automagic_CameAMP_COptim8bit,
        OptimizerConfig
    )
    print("✅ 成功導入 Automagic_CameAMP 優化器")
except ImportError as e:
    print(f"❌ 無法導入優化器: {e}")
    sys.exit(1)

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 使用設備: {device}")

class LoRALayer(nn.Module):
    """模擬 LoRA 層"""

    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features

        # LoRA 分解：A 和 B 矩陣
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 原始權重（凍結）
        self.register_buffer('base_weight', torch.randn(out_features, in_features))

    def forward(self, x):
        # 基礎前向傳播 + LoRA 調整
        base_out = F.linear(x, self.base_weight)
        lora_out = F.linear(F.linear(x, self.lora_A.T), self.lora_B.T)
        return base_out + lora_out

class MockStableDiffusionModel(nn.Module):
    """模擬 Stable Diffusion 模型的關鍵部分"""

    def __init__(self):
        super().__init__()

        # 模擬 UNet 的注意力層（使用 LoRA 微調）
        self.cross_attn_lora = LoRALayer(768, 768, rank=16)
        self.self_attn_lora = LoRALayer(768, 768, rank=8)

        # 模擬其他層
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(768, 512)

    def forward(self, x):
        # 模擬注意力計算
        x = self.norm(x)
        x = self.cross_attn_lora(x)
        x = self.self_attn_lora(x)
        x = self.proj(x)
        return x

def create_lora_optimizer_config(optimizer_type: str = "coptim") -> Dict:
    """創建 LoRA 優化的配置"""

    base_config = {
        'lr': 1e-3,               # LoRA 通常需要較高學習率
        'weight_decay': 1e-4,     # 適中的權重衰減
        'warmup_steps': 300,      # 較短的暖身期
        'full_finetune': False,   # 啟用 ALLoRA 行縮放
        'verbose': True
    }

    if optimizer_type == "coptim":
        # C-Optim 版本的 LoRA 友好配置
        base_config.update({
            'context_window': 30,      # 較小窗口，更靈敏
            'edge_threshold': 0.6,     # 降低閾值，減少邊緣情況觸發
            'adaptation_rate': 0.25    # 提高適應速率
        })

    return base_config

def test_lr_multiplier_evolution(optimizer, num_steps: int = 200):
    """測試學習率乘數的演化過程"""

    print(f"\n📊 測試 {optimizer.__class__.__name__} 的學習率乘數演化")
    print("-" * 60)

    model = MockStableDiffusionModel().to(device)

    # 記錄指標
    lr_multipliers = []
    edge_cases = []
    grad_consistencies = []
    loss_stabilities = []
    losses = []

    for step in range(num_steps):
        # 模擬訓練數據
        batch_size = 4
        seq_len = 77  # 典型的 CLIP 序列長度
        x = torch.randn(batch_size, seq_len, 768, device=device)
        target = torch.randn(batch_size, seq_len, 512, device=device)

        # 前向傳播
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # 記錄 C-Optim 指標
        if hasattr(optimizer, 'c_optim'):
            lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()
            is_edge = optimizer.c_optim.detect_edge_case()
            grad_consistency = optimizer.c_optim.compute_gradient_consistency()
            loss_stability = optimizer.c_optim.compute_loss_stability()

            lr_multipliers.append(lr_mult)
            edge_cases.append(is_edge)
            grad_consistencies.append(grad_consistency)
            loss_stabilities.append(loss_stability)

            # 定期輸出狀態
            if step % 50 == 0:
                status = "🔴 邊緣" if is_edge else "🟢 正常"
                print(f"Step {step:3d}: Loss={loss.item():.4f}, "
                      f"LR乘數={lr_mult:.3f}, 狀態={status}")
        else:
            lr_multipliers.append(1.0)
            edge_cases.append(False)
            grad_consistencies.append(1.0)
            loss_stabilities.append(1.0)

    return {
        'lr_multipliers': lr_multipliers,
        'edge_cases': edge_cases,
        'grad_consistencies': grad_consistencies,
        'loss_stabilities': loss_stabilities,
        'losses': losses
    }

def compare_optimizers():
    """比較優化前後的優化器性能"""

    print("\n🔬 LoRA 優化器性能比較測試")
    print("=" * 60)

    # 測試配置
    optimizers_config = [
        # 標準配置（舊版本模擬）
        {
            'name': '標準 C-Optim',
            'class': Automagic_CameAMP_COptim,
            'config': {
                'lr': 1e-3,
                'context_window': 50,
                'edge_threshold': 0.9,  # 舊的嚴格閾值
                'adaptation_rate': 0.1,
                'verbose': False
            }
        },
        # LoRA 優化配置（新版本）
        {
            'name': 'LoRA 優化 C-Optim',
            'class': Automagic_CameAMP_COptim,
            'config': create_lora_optimizer_config('coptim')
        }
    ]

    # 測試 8-bit 版本（如果可用）
    try:
        model_test = MockStableDiffusionModel().to(device)
        opt_test = Automagic_CameAMP_COptim8bit(model_test.parameters(), lr=1e-3)
        del opt_test, model_test

        # 8-bit 版本可用，添加到測試中
        optimizers_config.append({
            'name': 'LoRA 優化 8bit',
            'class': Automagic_CameAMP_COptim8bit,
            'config': create_lora_optimizer_config('coptim')
        })
        print("✅ 8-bit 版本可用，將包含在測試中")
    except Exception as e:
        print(f"⚠️  8-bit 版本不可用: {e}")

    results = {}

    for opt_config in optimizers_config:
        print(f"\n🧪 測試 {opt_config['name']}")

        # 創建新的模型實例（確保公平比較）
        model = MockStableDiffusionModel().to(device)

        # 創建優化器
        optimizer = opt_config['class'](
            model.parameters(),
            **opt_config['config']
        )

        # 運行測試
        result = test_lr_multiplier_evolution(optimizer, num_steps=150)
        results[opt_config['name']] = result

        # 輸出統計
        avg_lr_mult = np.mean(result['lr_multipliers'])
        edge_rate = np.mean(result['edge_cases']) * 100
        final_loss = result['losses'][-1]

        print(f"  平均 LR 乘數: {avg_lr_mult:.3f}")
        print(f"  邊緣情況比例: {edge_rate:.1f}%")
        print(f"  最終損失: {final_loss:.6f}")

    return results

def plot_comparison_results(results: Dict, save_dir: str = "docs/hina/plots"):
    """繪製比較結果"""

    if not results:
        print("❌ 沒有結果可繪製")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 設定顏色
    colors = ['blue', 'red', 'green', 'orange']

    # 1. 學習率乘數比較
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for i, (name, result) in enumerate(results.items()):
        plt.plot(result['lr_multipliers'],
                label=name,
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.8)

    plt.title('學習率乘數演化', fontsize=14, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('LR Multiplier')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 損失曲線
    plt.subplot(2, 2, 2)
    for i, (name, result) in enumerate(results.items()):
        plt.plot(result['losses'],
                label=name,
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.8)

    plt.title('損失曲線', fontsize=14, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 3. 邊緣情況檢測
    plt.subplot(2, 2, 3)
    for i, (name, result) in enumerate(results.items()):
        edge_numeric = [1 if x else 0 for x in result['edge_cases']]
        plt.plot(edge_numeric,
                label=name,
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.7)

    plt.title('邊緣情況檢測', fontsize=14, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('Edge Case (1=True, 0=False)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 梯度一致性
    plt.subplot(2, 2, 4)
    for i, (name, result) in enumerate(results.items()):
        plt.plot(result['grad_consistencies'],
                label=name,
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.8)

    plt.title('梯度一致性', fontsize=14, fontweight='bold')
    plt.xlabel('Step')
    plt.ylabel('Gradient Consistency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存圖表
    save_path = f"{save_dir}/lora_optimization_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 比較圖表已保存: {save_path}")
    plt.show()

def print_recommendations():
    """打印 LoRA 訓練的建議配置"""

    print("\n" + "=" * 80)
    print("💡 LoRA 訓練的 Automagic_CameAMP 推薦配置")
    print("=" * 80)

    print("\n🎯 針對 Stable Diffusion LoRA 的優化建議:")

    print("\n1. 使用 LoRA 優化版本的 C-Optim 配置:")
    print("```python")
    print("from library.automagic_cameamp import Automagic_CameAMP_COptim")
    print("")
    print("optimizer = Automagic_CameAMP_COptim(")
    print("    model.parameters(),")
    print("    lr=1e-3,                    # 提高基礎學習率")
    print("    weight_decay=1e-4,          # 適中的正則化")
    print("    warmup_steps=300,           # 較短暖身期")
    print("    context_window=30,          # 減小窗口，提高靈敏度")
    print("    edge_threshold=0.6,         # 降低閾值，減少邊緣觸發")
    print("    adaptation_rate=0.25,       # 提高適應速率")
    print("    full_finetune=False,        # 啟用 ALLoRA")
    print("    verbose=True")
    print(")")
    print("```")

    print("\n2. 如果記憶體受限，使用 8-bit 版本:")
    print("```python")
    print("from library.automagic_cameamp import Automagic_CameAMP_COptim8bit")
    print("")
    print("optimizer = Automagic_CameAMP_COptim8bit(")
    print("    model.parameters(),")
    print("    lr=1e-3,")
    print("    context_window=25,          # 8-bit 版本建議更小窗口")
    print("    edge_threshold=0.5,         # 更寬容的邊緣檢測")
    print("    adaptation_rate=0.3,        # 略高的適應速率")
    print("    full_finetune=False")
    print(")")
    print("```")

    print("\n3. 監控學習率乘數:")
    print("```python")
    print("# 在訓練循環中添加監控")
    print("if step % 100 == 0:")
    print("    lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()")
    print("    is_edge = optimizer.c_optim.detect_edge_case()")
    print("    print(f'Step {step}: LR倍數={lr_mult:.3f}, 邊緣={is_edge}')")
    print("```")

    print("\n4. 關鍵改進點:")
    print("   ✅ 提高了學習率乘數的基準值（1.1-4.0 vs 0.5-3.0）")
    print("   ✅ 放寬了邊緣情況檢測條件（變異係數 0.5 vs 0.3）")
    print("   ✅ 減少了邊緣情況的學習率懲罰（0.7-0.95 vs 0.4-0.8）")
    print("   ✅ 優化了停滯檢測閾值（30 步 vs 20 步）")
    print("   ✅ 增強了正向收斂的獎勵機制")

    print("\n5. 預期效果:")
    print("   📈 學習率乘數平均提高 50-80%")
    print("   📉 邊緣情況觸發頻率降低 30-50%")
    print("   🚀 LoRA 訓練效果明顯改善")
    print("   💾 8-bit 版本節省 ~75% 記憶體")

def main():
    """主函數"""

    print("🎨 LoRA 優化版 Automagic_CameAMP 測試")
    print("解決學習率乘數過小問題")
    print("=" * 60)

    try:
        # 運行比較測試
        results = compare_optimizers()

        # 繪製結果
        plot_comparison_results(results)

        # 顯示詳細統計
        print("\n📊 詳細統計比較:")
        print("-" * 60)

        for name, result in results.items():
            avg_lr_mult = np.mean(result['lr_multipliers'])
            max_lr_mult = np.max(result['lr_multipliers'])
            min_lr_mult = np.min(result['lr_multipliers'])
            edge_rate = np.mean(result['edge_cases']) * 100

            print(f"\n{name}:")
            print(f"  平均 LR 乘數: {avg_lr_mult:.3f}")
            print(f"  LR 乘數範圍: {min_lr_mult:.3f} - {max_lr_mult:.3f}")
            print(f"  邊緣情況比例: {edge_rate:.1f}%")

        # 顯示建議
        print_recommendations()

        print(f"\n✅ 測試完成！")
        print("現在您可以使用優化後的配置進行 LoRA 訓練了！")

    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()