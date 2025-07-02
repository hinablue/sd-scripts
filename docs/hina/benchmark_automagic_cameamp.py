#!/usr/bin/env python3
"""
Automagic_CameAMP 優化器性能基準測試

這個測試腳本比較 Automagic_CameAMP 與其他常見優化器的性能。

作者: Hina
版本: 1.0
日期: 2025-01-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as torch_optim
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 加入庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from library.automagic_cameamp import (
        Automagic_CameAMP,
        Automagic_CameAMP8bit,
        Automagic_CameAMP_COptim,
        Automagic_CameAMP_COptim8bit
    )
    AUTOMAGIC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  無法導入 Automagic 優化器: {e}")
    AUTOMAGIC_AVAILABLE = False

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 使用設備: {device}")

@dataclass
class BenchmarkConfig:
    """基準測試配置"""
    model_sizes: List[str] = None
    batch_size: int = 64
    input_size: int = 784
    num_classes: int = 10
    num_epochs: int = 50
    num_warmup: int = 5
    measure_memory: bool = True
    save_plots: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.model_sizes is None:
            self.model_sizes = ['small', 'medium', 'large']

class BenchmarkModel(nn.Module):
    """可配置大小的測試模型"""

    def __init__(self, input_size: int, num_classes: int, size: str = 'medium'):
        super().__init__()

        if size == 'small':
            hidden_sizes = [128, 64]
        elif size == 'medium':
            hidden_sizes = [512, 256, 128]
        elif size == 'large':
            hidden_sizes = [1024, 512, 256, 128]
        else:
            raise ValueError(f"未知模型大小: {size}")

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

        # 計算參數數量
        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.network(x)

class MemoryTracker:
    """記憶體使用追蹤器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.peak_memory = 0
        self.current_memory = 0

    def update(self):
        if device.type == 'cuda':
            self.current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            self.peak_memory = max(self.peak_memory, self.current_memory)
        else:
            process = psutil.Process()
            self.current_memory = process.memory_info().rss / 1024**2  # MB
            self.peak_memory = max(self.peak_memory, self.current_memory)

    def get_memory_mb(self) -> float:
        return self.current_memory

    def get_peak_memory_mb(self) -> float:
        return self.peak_memory

class OptimizersComparison:
    """優化器比較測試"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}

    def create_optimizers(self, model_params) -> Dict[str, torch.optim.Optimizer]:
        """創建所有要測試的優化器"""
        optimizers = {}

        # PyTorch 內建優化器
        optimizers['Adam'] = torch_optim.Adam(model_params, lr=1e-3, weight_decay=1e-4)
        optimizers['AdamW'] = torch_optim.AdamW(model_params, lr=1e-3, weight_decay=1e-4)
        optimizers['SGD'] = torch_optim.SGD(model_params, lr=1e-3, momentum=0.9, weight_decay=1e-4)
        optimizers['RMSprop'] = torch_optim.RMSprop(model_params, lr=1e-3, weight_decay=1e-4)

        # Automagic 優化器 (如果可用)
        if AUTOMAGIC_AVAILABLE:
            try:
                optimizers['Automagic_CameAMP'] = Automagic_CameAMP(
                    model_params, lr=1e-3, weight_decay=1e-4, warmup_steps=50
                )
            except Exception as e:
                print(f"⚠️  無法創建 Automagic_CameAMP: {e}")

            try:
                optimizers['Automagic_CameAMP_COptim'] = Automagic_CameAMP_COptim(
                    model_params, lr=1e-3, weight_decay=1e-4, warmup_steps=50,
                    context_window=30, edge_threshold=0.8
                )
            except Exception as e:
                print(f"⚠️  無法創建 Automagic_CameAMP_COptim: {e}")

            # 8-bit 版本 (如果支援)
            try:
                optimizers['Automagic_CameAMP8bit'] = Automagic_CameAMP8bit(
                    model_params, lr=1e-3, weight_decay=1e-4, warmup_steps=50
                )
            except Exception as e:
                print(f"⚠️  無法創建 8-bit 版本: {e}")

        return optimizers

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成測試數據"""
        x = torch.randn(self.config.batch_size, self.config.input_size, device=device)
        y = torch.randint(0, self.config.num_classes, (self.config.batch_size,), device=device)
        return x, y

    def benchmark_optimizer(self,
                          optimizer_name: str,
                          optimizer: torch.optim.Optimizer,
                          model: nn.Module) -> Dict[str, float]:
        """測試單個優化器"""
        print(f"  🧪 測試 {optimizer_name}")

        model.train()
        memory_tracker = MemoryTracker()

        # 記錄指標
        losses = []
        times = []
        memory_usage = []

        # 清理記憶體
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        memory_tracker.update()
        initial_memory = memory_tracker.get_memory_mb()

        try:
            for epoch in range(self.config.num_epochs):
                epoch_start = time.time()
                epoch_losses = []

                for batch_idx in range(20):  # 每個 epoch 20 個批次
                    x, y = self.generate_data()

                    optimizer.zero_grad()
                    output = model(x)
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())

                    # 記錄記憶體使用
                    if self.config.measure_memory and batch_idx == 0:
                        memory_tracker.update()
                        memory_usage.append(memory_tracker.get_memory_mb())

                epoch_time = time.time() - epoch_start
                avg_loss = np.mean(epoch_losses)

                losses.append(avg_loss)
                times.append(epoch_time)

                # 暖身後才開始記錄
                if epoch >= self.config.num_warmup and self.config.verbose:
                    if epoch % 10 == 0:
                        print(f"    Epoch {epoch}: Loss = {avg_loss:.6f}, Time = {epoch_time:.3f}s")

        except Exception as e:
            print(f"    ❌ {optimizer_name} 訓練失敗: {e}")
            return None

        total_time = time.time() - start_time
        memory_tracker.update()
        peak_memory = memory_tracker.get_peak_memory_mb()
        memory_overhead = peak_memory - initial_memory

        # 計算指標 (跳過暖身)
        valid_losses = losses[self.config.num_warmup:]
        valid_times = times[self.config.num_warmup:]

        results = {
            'final_loss': valid_losses[-1] if valid_losses else float('inf'),
            'best_loss': min(valid_losses) if valid_losses else float('inf'),
            'avg_time_per_epoch': np.mean(valid_times) if valid_times else float('inf'),
            'total_time': total_time,
            'memory_overhead_mb': memory_overhead,
            'peak_memory_mb': peak_memory,
            'convergence_rate': len(valid_losses) - np.argmin(valid_losses) if valid_losses else 0,
            'losses': losses,
            'times': times,
            'memory_usage': memory_usage
        }

        print(f"    ✅ 完成 - 最終損失: {results['final_loss']:.6f}, "
              f"平均時間: {results['avg_time_per_epoch']:.3f}s, "
              f"記憶體: {results['memory_overhead_mb']:.1f}MB")

        return results

    def run_benchmark(self, model_size: str) -> Dict[str, Dict]:
        """運行特定模型大小的基準測試"""
        print(f"\n🏃 運行 {model_size} 模型基準測試")
        print("=" * 50)

        # 創建模型
        model = BenchmarkModel(
            self.config.input_size,
            self.config.num_classes,
            model_size
        ).to(device)

        print(f"📊 模型參數: {model.param_count:,}")

        # 創建優化器
        optimizers = self.create_optimizers(model.parameters())
        print(f"🔧 測試 {len(optimizers)} 個優化器")

        results = {}

        for opt_name, optimizer in optimizers.items():
            # 重新初始化模型權重以確保公平比較
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

            result = self.benchmark_optimizer(opt_name, optimizer, model)
            if result is not None:
                results[opt_name] = result

        return results

    def run_all_benchmarks(self) -> Dict[str, Dict]:
        """運行所有基準測試"""
        print("🚀 開始 Automagic_CameAMP 性能基準測試")
        print("作者: Hina")
        print("版本: 1.0")
        print("=" * 60)

        all_results = {}

        for model_size in self.config.model_sizes:
            results = self.run_benchmark(model_size)
            if results:
                all_results[model_size] = results

        self.results = all_results
        return all_results

    def plot_results(self, save_dir: str = "docs/hina/plots"):
        """繪製基準測試結果"""
        if not self.results:
            print("❌ 沒有結果可繪製")
            return

        os.makedirs(save_dir, exist_ok=True)

        # 顏色映射
        color_map = {
            'Adam': 'blue',
            'AdamW': 'green',
            'SGD': 'red',
            'RMSprop': 'orange',
            'Automagic_CameAMP': 'purple',
            'Automagic_CameAMP_COptim': 'brown',
            'Automagic_CameAMP8bit': 'pink'
        }

        # 1. 最終損失比較
        fig, axes = plt.subplots(1, len(self.config.model_sizes), figsize=(15, 5))
        if len(self.config.model_sizes) == 1:
            axes = [axes]

        for i, model_size in enumerate(self.config.model_sizes):
            if model_size not in self.results:
                continue

            results = self.results[model_size]
            names = list(results.keys())
            final_losses = [results[name]['final_loss'] for name in names]
            colors = [color_map.get(name, 'gray') for name in names]

            bars = axes[i].bar(names, final_losses, color=colors, alpha=0.7)
            axes[i].set_title(f'{model_size.title()} Model - Final Loss')
            axes[i].set_ylabel('Loss')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_yscale('log')

            # 添加數值標籤
            for bar, loss in zip(bars, final_losses):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{loss:.4f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if self.config.save_plots:
            plt.savefig(f"{save_dir}/benchmark_final_loss.png", dpi=300, bbox_inches='tight')
            print(f"📊 最終損失圖已保存: {save_dir}/benchmark_final_loss.png")
        plt.show()

        # 2. 訓練時間比較
        fig, axes = plt.subplots(1, len(self.config.model_sizes), figsize=(15, 5))
        if len(self.config.model_sizes) == 1:
            axes = [axes]

        for i, model_size in enumerate(self.config.model_sizes):
            if model_size not in self.results:
                continue

            results = self.results[model_size]
            names = list(results.keys())
            avg_times = [results[name]['avg_time_per_epoch'] for name in names]
            colors = [color_map.get(name, 'gray') for name in names]

            bars = axes[i].bar(names, avg_times, color=colors, alpha=0.7)
            axes[i].set_title(f'{model_size.title()} Model - Training Speed')
            axes[i].set_ylabel('Time per Epoch (s)')
            axes[i].tick_params(axis='x', rotation=45)

            # 添加數值標籤
            for bar, time_val in zip(bars, avg_times):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if self.config.save_plots:
            plt.savefig(f"{save_dir}/benchmark_training_time.png", dpi=300, bbox_inches='tight')
            print(f"📊 訓練時間圖已保存: {save_dir}/benchmark_training_time.png")
        plt.show()

        # 3. 記憶體使用比較
        if self.config.measure_memory:
            fig, axes = plt.subplots(1, len(self.config.model_sizes), figsize=(15, 5))
            if len(self.config.model_sizes) == 1:
                axes = [axes]

            for i, model_size in enumerate(self.config.model_sizes):
                if model_size not in self.results:
                    continue

                results = self.results[model_size]
                names = list(results.keys())
                memory_usage = [results[name]['memory_overhead_mb'] for name in names]
                colors = [color_map.get(name, 'gray') for name in names]

                bars = axes[i].bar(names, memory_usage, color=colors, alpha=0.7)
                axes[i].set_title(f'{model_size.title()} Model - Memory Usage')
                axes[i].set_ylabel('Memory Overhead (MB)')
                axes[i].tick_params(axis='x', rotation=45)

                # 添加數值標籤
                for bar, memory in zip(bars, memory_usage):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{memory:.1f}MB', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            if self.config.save_plots:
                plt.savefig(f"{save_dir}/benchmark_memory_usage.png", dpi=300, bbox_inches='tight')
                print(f"📊 記憶體使用圖已保存: {save_dir}/benchmark_memory_usage.png")
            plt.show()

        # 4. 損失曲線 (選擇一個模型大小)
        if self.config.model_sizes:
            model_size = self.config.model_sizes[0]  # 使用第一個模型大小
            if model_size in self.results:
                plt.figure(figsize=(12, 8))

                for name, result in self.results[model_size].items():
                    if 'losses' in result:
                        losses = result['losses'][self.config.num_warmup:]  # 跳過暖身
                        color = color_map.get(name, 'gray')
                        plt.plot(losses, label=name, color=color, linewidth=2, alpha=0.8)

                plt.title(f'{model_size.title()} Model - Loss Curves', fontsize=16, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log')

                if self.config.save_plots:
                    plt.savefig(f"{save_dir}/benchmark_loss_curves.png", dpi=300, bbox_inches='tight')
                    print(f"📊 損失曲線圖已保存: {save_dir}/benchmark_loss_curves.png")
                plt.show()

    def print_summary(self):
        """打印基準測試總結"""
        if not self.results:
            print("❌ 沒有基準測試結果")
            return

        print("\n" + "=" * 80)
        print("📋 Automagic_CameAMP 基準測試總結")
        print("=" * 80)

        for model_size, results in self.results.items():
            print(f"\n🔍 {model_size.upper()} 模型結果:")
            print("-" * 40)

            # 創建結果表格
            headers = ["優化器", "最終損失", "最佳損失", "平均時間(s)", "記憶體(MB)"]

            # 收集數據
            data = []
            for name, result in results.items():
                data.append([
                    name,
                    f"{result['final_loss']:.6f}",
                    f"{result['best_loss']:.6f}",
                    f"{result['avg_time_per_epoch']:.3f}",
                    f"{result['memory_overhead_mb']:.1f}"
                ])

            # 打印表格
            col_widths = [max(len(str(row[i])) for row in [headers] + data) + 2
                         for i in range(len(headers))]

            def print_row(row):
                print("|" + "|".join(f" {str(item).ljust(col_widths[i])} "
                                     for i, item in enumerate(row)) + "|")

            print_row(headers)
            print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
            for row in data:
                print_row(row)

            # 分析最佳表現
            if results:
                best_loss = min(results.items(), key=lambda x: x[1]['final_loss'])
                fastest = min(results.items(), key=lambda x: x[1]['avg_time_per_epoch'])
                most_memory_efficient = min(results.items(), key=lambda x: x[1]['memory_overhead_mb'])

                print(f"\n💡 {model_size.title()} 模型分析:")
                print(f"   🏆 最佳損失: {best_loss[0]} ({best_loss[1]['final_loss']:.6f})")
                print(f"   ⚡ 最快速度: {fastest[0]} ({fastest[1]['avg_time_per_epoch']:.3f}s/epoch)")
                print(f"   💾 最省記憶體: {most_memory_efficient[0]} ({most_memory_efficient[1]['memory_overhead_mb']:.1f}MB)")

        # 整體推薦
        print(f"\n🎯 總體推薦:")
        print("=" * 40)

        # 檢查 Automagic 優化器是否在測試中
        automagic_optimizers = []
        for model_size, results in self.results.items():
            for name in results.keys():
                if 'Automagic' in name and name not in automagic_optimizers:
                    automagic_optimizers.append(name)

        if automagic_optimizers:
            print("✅ Automagic 優化器可用:")
            for opt in automagic_optimizers:
                print(f"   - {opt}")
            print("\n📈 使用建議:")
            print("   • 一般訓練: Automagic_CameAMP")
            print("   • 大模型: Automagic_CameAMP8bit")
            print("   • 高級功能: Automagic_CameAMP_COptim")
            print("   • 生產環境: Automagic_CameAMP_COptim8bit")
        else:
            print("⚠️  沒有測試 Automagic 優化器")
            print("   請確保 library/automagic_cameamp.py 可用")

def main():
    """主測試函數"""
    if not AUTOMAGIC_AVAILABLE:
        print("❌ Automagic 優化器不可用，僅測試 PyTorch 內建優化器")

    # 測試配置
    config = BenchmarkConfig(
        model_sizes=['small', 'medium'],  # 減少測試時間
        num_epochs=30,
        num_warmup=5,
        batch_size=64,
        measure_memory=True,
        save_plots=True,
        verbose=True
    )

    print(f"📋 基準測試配置:")
    print(f"   模型大小: {config.model_sizes}")
    print(f"   訓練輪數: {config.num_epochs}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   暖身輪數: {config.num_warmup}")

    # 運行基準測試
    comparison = OptimizersComparison(config)
    results = comparison.run_all_benchmarks()

    if results:
        # 繪製結果
        comparison.plot_results()

        # 打印總結
        comparison.print_summary()

        print(f"\n✅ 基準測試完成！")
        print(f"   測試了 {len(config.model_sizes)} 種模型大小")
        print(f"   總共 {sum(len(r) for r in results.values())} 個優化器測試")
    else:
        print("❌ 基準測試失敗，沒有成功的結果")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  基準測試被用戶中斷")
    except Exception as e:
        print(f"\n❌ 基準測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Automagic_CameAMP 基準測試完成！")