#!/usr/bin/env python3
"""
Automagic_CameAMP 優化器完整測試套件

這個測試文件演示了所有四個版本的 Automagic_CameAMP 優化器的使用方法：
1. Automagic_CameAMP - 基礎版本
2. Automagic_CameAMP8bit - 8-bit 量化版本
3. Automagic_CameAMP_COptim - 上下文感知版本
4. Automagic_CameAMP_COptim8bit - 全功能版本

作者: Hina
版本: 2.0
日期: 2025-01-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 加入庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from library.automagic_cameamp import (
        Automagic_CameAMP,
        Automagic_CameAMP8bit,
        Automagic_CameAMP_COptim,
        Automagic_CameAMP_COptim8bit,
        OptimizerConfig
    )
except ImportError as e:
    print(f"❌ 無法導入優化器: {e}")
    print("請確保 library/automagic_cameamp.py 存在且可用")
    sys.exit(1)

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 使用設備: {device}")

@dataclass
class TestConfig:
    """測試配置"""
    batch_size: int = 32
    input_size: int = 128
    hidden_sizes: List[int] = None
    output_size: int = 10
    num_epochs: int = 20
    num_batches: int = 50
    save_plots: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128, 64]

class TestModel(nn.Module):
    """測試用的神經網路模型"""

    def __init__(self, config: TestConfig):
        super().__init__()
        self.config = config

        layers = []
        prev_size = config.input_size

        # 隱藏層
        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size

        # 輸出層
        layers.append(nn.Linear(prev_size, config.output_size))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化權重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

class OptimizerTester:
    """優化器測試器"""

    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.results = {}

    def create_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成合成測試數據"""
        # 生成分類數據
        X = torch.randn(
            self.config.batch_size,
            self.config.input_size,
            device=device
        )
        y = torch.randint(
            0,
            self.config.output_size,
            (self.config.batch_size,),
            device=device
        )
        return X, y

    def test_optimizer(self,
                      optimizer_class,
                      optimizer_name: str,
                      optimizer_kwargs: Optional[Dict] = None) -> Dict[str, List[float]]:
        """測試單個優化器"""
        print(f"\n🧪 測試 {optimizer_name}")
        print("=" * 50)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        # 創建模型
        model = TestModel(self.config).to(device)

        # 創建優化器
        try:
            optimizer = optimizer_class(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                warmup_steps=100,
                verbose=False,
                **optimizer_kwargs
            )
            print(f"✅ {optimizer_name} 初始化成功")
        except Exception as e:
            print(f"❌ {optimizer_name} 初始化失敗: {e}")
            return None

        # 測試指標
        losses = []
        learning_rates = []
        gradient_norms = []
        times = []

        # 額外指標 (C-Optim 版本)
        contextual_multipliers = []
        edge_cases = []

        model.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            epoch_start = time.time()

            for batch_idx in range(self.config.num_batches):
                # 生成數據
                X, y = self.create_synthetic_data()

                # 前向傳播
                optimizer.zero_grad()
                outputs = model(X)
                loss = F.cross_entropy(outputs, y)

                # 反向傳播
                loss.backward()

                # 計算梯度範數
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                # 優化器步驟
                step_loss = optimizer.step(closure=lambda: loss)
                if step_loss is not None:
                    loss = step_loss

                # 記錄指標
                epoch_losses.append(loss.item())
                gradient_norms.append(total_norm)

                # 記錄學習率
                if hasattr(optimizer, '_get_group_lr'):
                    current_lr = optimizer._get_group_lr(optimizer.param_groups[0])
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)

                # C-Optim 特殊指標
                if hasattr(optimizer, 'c_optim'):
                    lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()
                    is_edge = optimizer.c_optim.detect_edge_case()
                    contextual_multipliers.append(lr_mult)
                    edge_cases.append(is_edge)

            # 記錄時間和損失
            epoch_time = time.time() - epoch_start
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            times.append(epoch_time)

            if self.config.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch:2d}: Loss = {avg_loss:.6f}, "
                      f"LR = {current_lr:.6f}, Time = {epoch_time:.2f}s")

                # C-Optim 額外信息
                if hasattr(optimizer, 'c_optim'):
                    if contextual_multipliers:
                        print(f"          上下文乘數 = {contextual_multipliers[-1]:.3f}, "
                              f"邊緣情況 = {edge_cases[-1]}")

        # 計算最終統計
        final_loss = losses[-1]
        min_loss = min(losses)
        avg_time = np.mean(times)
        convergence_speed = len(losses) - np.argmin(losses)

        print(f"📊 {optimizer_name} 結果:")
        print(f"   最終損失: {final_loss:.6f}")
        print(f"   最佳損失: {min_loss:.6f}")
        print(f"   平均時間: {avg_time:.2f}s/epoch")
        print(f"   收斂速度: {convergence_speed} epochs to min")

        # 返回結果
        results = {
            'losses': losses,
            'learning_rates': learning_rates,
            'gradient_norms': gradient_norms,
            'times': times,
            'final_loss': final_loss,
            'min_loss': min_loss,
            'avg_time': avg_time,
            'convergence_speed': convergence_speed
        }

        # C-Optim 版本的額外結果
        if contextual_multipliers:
            results['contextual_multipliers'] = contextual_multipliers
            results['edge_cases'] = edge_cases

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """運行所有優化器測試"""
        print("🚀 開始 Automagic_CameAMP 優化器套件測試")
        print("=" * 60)

        optimizers_to_test = [
            (Automagic_CameAMP, "Automagic_CameAMP (基礎版)", {}),
            (Automagic_CameAMP_COptim, "Automagic_CameAMP_COptim (上下文感知)", {
                'context_window': 50,
                'edge_threshold': 0.8,
                'adaptation_rate': 0.15
            })
        ]

        # 測試 8-bit 版本（如果可用）
        try:
            # 測試 bitsandbytes 可用性
            test_model = TestModel(self.config).to(device)
            test_optimizer = Automagic_CameAMP8bit(test_model.parameters(), lr=1e-3)
            del test_optimizer, test_model

            optimizers_to_test.extend([
                (Automagic_CameAMP8bit, "Automagic_CameAMP8bit (8-bit)", {}),
                (Automagic_CameAMP_COptim8bit, "Automagic_CameAMP_COptim8bit (全功能)", {
                    'context_window': 30,
                    'edge_threshold': 0.7,
                    'adaptation_rate': 0.2
                })
            ])
            print("✅ 8-bit 版本可用，將包含在測試中")
        except Exception as e:
            print(f"⚠️  8-bit 版本不可用: {e}")
            print("   跳過 8-bit 版本測試")

        # 運行測試
        for optimizer_class, optimizer_name, kwargs in optimizers_to_test:
            result = self.test_optimizer(optimizer_class, optimizer_name, kwargs)
            if result is not None:
                self.results[optimizer_name] = result

        return self.results

    def plot_results(self, save_dir: str = "docs/hina/plots"):
        """繪製測試結果"""
        if not self.results:
            print("❌ 沒有結果可繪製")
            return

        # 創建保存目錄
        os.makedirs(save_dir, exist_ok=True)

        # 設定繪圖樣式
        plt.style.use('default')
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        # 1. 損失曲線
        plt.figure(figsize=(12, 8))
        for i, (name, results) in enumerate(self.results.items()):
            plt.plot(results['losses'],
                    label=name,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8)

        plt.title('Automagic_CameAMP 優化器損失比較', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        if self.config.save_plots:
            plt.savefig(f"{save_dir}/loss_comparison.png", dpi=300, bbox_inches='tight')
            print(f"📊 損失比較圖已保存: {save_dir}/loss_comparison.png")
        plt.show()

        # 2. 學習率變化
        plt.figure(figsize=(12, 8))
        for i, (name, results) in enumerate(self.results.items()):
            # 只顯示前 200 個點避免過於密集
            lr_data = results['learning_rates'][:200] if len(results['learning_rates']) > 200 else results['learning_rates']
            plt.plot(lr_data,
                    label=name,
                    color=colors[i % len(colors)],
                    linewidth=1.5,
                    alpha=0.7)

        plt.title('學習率變化對比', fontsize=16, fontweight='bold')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        if self.config.save_plots:
            plt.savefig(f"{save_dir}/learning_rate_comparison.png", dpi=300, bbox_inches='tight')
            print(f"📊 學習率比較圖已保存: {save_dir}/learning_rate_comparison.png")
        plt.show()

        # 3. 梯度範數
        plt.figure(figsize=(12, 8))
        for i, (name, results) in enumerate(self.results.items()):
            grad_data = results['gradient_norms'][:200] if len(results['gradient_norms']) > 200 else results['gradient_norms']
            plt.plot(grad_data,
                    label=name,
                    color=colors[i % len(colors)],
                    linewidth=1.5,
                    alpha=0.7)

        plt.title('梯度範數變化', fontsize=16, fontweight='bold')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Gradient Norm', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        if self.config.save_plots:
            plt.savefig(f"{save_dir}/gradient_norm_comparison.png", dpi=300, bbox_inches='tight')
            print(f"📊 梯度範數比較圖已保存: {save_dir}/gradient_norm_comparison.png")
        plt.show()

        # 4. C-Optim 特殊指標 (如果有的話)
        c_optim_results = {name: results for name, results in self.results.items()
                          if 'contextual_multipliers' in results}

        if c_optim_results:
            plt.figure(figsize=(12, 10))

            # 上下文乘數
            plt.subplot(2, 1, 1)
            for i, (name, results) in enumerate(c_optim_results.items()):
                mult_data = results['contextual_multipliers'][:200] if len(results['contextual_multipliers']) > 200 else results['contextual_multipliers']
                plt.plot(mult_data,
                        label=name,
                        color=colors[i % len(colors)],
                        linewidth=1.5)

            plt.title('上下文學習率乘數', fontsize=14, fontweight='bold')
            plt.ylabel('LR Multiplier', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

            # 邊緣情況
            plt.subplot(2, 1, 2)
            for i, (name, results) in enumerate(c_optim_results.items()):
                edge_data = results['edge_cases'][:200] if len(results['edge_cases']) > 200 else results['edge_cases']
                edge_numeric = [1 if x else 0 for x in edge_data]
                plt.plot(edge_numeric,
                        label=name,
                        color=colors[i % len(colors)],
                        linewidth=1.5,
                        alpha=0.7)

            plt.title('邊緣情況檢測', fontsize=14, fontweight='bold')
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Edge Case (1=True, 0=False)', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            if self.config.save_plots:
                plt.savefig(f"{save_dir}/c_optim_metrics.png", dpi=300, bbox_inches='tight')
                print(f"📊 C-Optim 指標圖已保存: {save_dir}/c_optim_metrics.png")
            plt.show()

    def print_summary(self):
        """打印測試總結"""
        if not self.results:
            print("❌ 沒有測試結果")
            return

        print("\n" + "=" * 80)
        print("📋 Automagic_CameAMP 優化器測試總結")
        print("=" * 80)

        # 創建結果表格
        headers = ["優化器", "最終損失", "最佳損失", "平均時間", "收斂速度"]
        data = []

        for name, results in self.results.items():
            data.append([
                name.split('(')[0].strip(),  # 簡化名稱
                f"{results['final_loss']:.6f}",
                f"{results['min_loss']:.6f}",
                f"{results['avg_time']:.2f}s",
                f"{results['convergence_speed']} epochs"
            ])

        # 打印表格
        col_widths = [max(len(str(row[i])) for row in [headers] + data) + 2
                     for i in range(len(headers))]

        def print_row(row):
            print("|" + "|".join(f" {str(item).ljust(col_widths[i])} "
                                 for i, item in enumerate(row)) + "|")

        # 表格標題
        print_row(headers)
        print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")

        # 表格數據
        for row in data:
            print_row(row)

        # 推薦
        print("\n💡 使用建議:")
        print("=" * 40)

        best_final = min(self.results.items(), key=lambda x: x[1]['final_loss'])
        fastest = min(self.results.items(), key=lambda x: x[1]['avg_time'])
        best_convergence = min(self.results.items(), key=lambda x: x[1]['convergence_speed'])

        print(f"🏆 最佳最終損失: {best_final[0]}")
        print(f"⚡ 最快訓練速度: {fastest[0]}")
        print(f"🎯 最快收斂速度: {best_convergence[0]}")

        # 記憶體使用建議
        print("\n💾 記憶體使用指南:")
        has_8bit = any('8bit' in name for name in self.results.keys())
        if has_8bit:
            print("✅ 8-bit 版本可用 - 推薦用於大模型 (節省 ~75% 記憶體)")
        else:
            print("⚠️  8-bit 版本不可用 - 需要安裝 bitsandbytes")

        has_coptim = any('COptim' in name for name in self.results.keys())
        if has_coptim:
            print("✅ C-Optim 版本可用 - 推薦用於需要智能調整的場景")

def run_comprehensive_test():
    """運行完整測試"""
    print("🧪 Automagic_CameAMP 優化器測試套件")
    print("作者: Hina")
    print("版本: 2.0")
    print("日期: 2025-01-27")
    print("=" * 60)

    # 測試配置
    test_config = TestConfig(
        batch_size=64,
        input_size=128,
        hidden_sizes=[256, 128, 64],
        output_size=10,
        num_epochs=15,
        num_batches=30,
        save_plots=True,
        verbose=True
    )

    print(f"📋 測試配置:")
    print(f"   批次大小: {test_config.batch_size}")
    print(f"   輸入維度: {test_config.input_size}")
    print(f"   隱藏層: {test_config.hidden_sizes}")
    print(f"   輸出類別: {test_config.output_size}")
    print(f"   訓練輪數: {test_config.num_epochs}")
    print(f"   每輪批次: {test_config.num_batches}")

    # 創建測試器
    tester = OptimizerTester(test_config)

    # 運行測試
    results = tester.run_all_tests()

    if results:
        # 繪製結果
        tester.plot_results()

        # 打印總結
        tester.print_summary()

        print(f"\n✅ 測試完成！共測試了 {len(results)} 個優化器")
    else:
        print("❌ 測試失敗，沒有成功的結果")

def demo_basic_usage():
    """演示基本使用方法"""
    print("\n" + "=" * 60)
    print("🎯 基本使用方法演示")
    print("=" * 60)

    # 創建簡單模型
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)

    print("📦 模型結構:")
    print(model)

    # 演示各種優化器創建
    optimizers = {}

    # 1. 基礎版本
    try:
        optimizers['基礎版'] = Automagic_CameAMP(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            warmup_steps=100
        )
        print("✅ 基礎版本創建成功")
    except Exception as e:
        print(f"❌ 基礎版本創建失敗: {e}")

    # 2. C-Optim 版本
    try:
        optimizers['上下文感知版'] = Automagic_CameAMP_COptim(
            model.parameters(),
            lr=1e-3,
            context_window=50,
            edge_threshold=0.8
        )
        print("✅ 上下文感知版本創建成功")
    except Exception as e:
        print(f"❌ 上下文感知版本創建失敗: {e}")

    # 3. 8-bit 版本（如果可用）
    try:
        optimizers['8-bit版'] = Automagic_CameAMP8bit(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        print("✅ 8-bit 版本創建成功")
    except Exception as e:
        print(f"⚠️  8-bit 版本不可用: {e}")

    # 4. 全功能版本
    try:
        optimizers['全功能版'] = Automagic_CameAMP_COptim8bit(
            model.parameters(),
            lr=1e-3,
            context_window=30
        )
        print("✅ 全功能版本創建成功")
    except Exception as e:
        print(f"⚠️  全功能版本不可用: {e}")

    print(f"\n🎉 成功創建 {len(optimizers)} 個優化器")

    # 簡單訓練演示
    if optimizers:
        print("\n🚀 簡單訓練演示 (基礎版本):")

        optimizer_name, optimizer = next(iter(optimizers.items()))
        print(f"使用優化器: {optimizer_name}")

        # 訓練數據
        X = torch.randn(32, 100, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        # 訓練幾步
        for step in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()

            print(f"Step {step + 1}: Loss = {loss.item():.6f}")

        print("✅ 訓練演示完成")

if __name__ == "__main__":
    try:
        # 基本使用演示
        demo_basic_usage()

        # 完整測試
        run_comprehensive_test()

    except KeyboardInterrupt:
        print("\n⚠️  測試被用戶中斷")
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🎉 Automagic_CameAMP 測試完成！")
    print("感謝使用 Automagic_CameAMP 優化器套件 🚀")