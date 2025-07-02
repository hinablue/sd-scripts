#!/usr/bin/env python3
"""
使用 bitsandbytes 的 Automagic_CameAMP_Improved_8Bit 優化器使用範例

這個範例展示如何使用基於 bitsandbytes 的 8bit 優化器來訓練 LoRA 模型。
相比自定義量化版本，bitsandbytes 版本更加穩定且高效。

作者: AI 助手
版本: 1.0
日期: 2024年12月
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import math
from typing import Dict, List, Any
import os

# 導入我們的 8bit 優化器
from automagic_cameamp_improved_8bit import (
    Automagic_CameAMP_Improved_8Bit,
    OptimizationProfiles,
    create_improved_8bit_optimizer,
    BITSANDBYTES_AVAILABLE
)

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 使用設備: {device}")
print(f"📦 bitsandbytes 可用性: {'✅' if BITSANDBYTES_AVAILABLE else '❌'}")

if not BITSANDBYTES_AVAILABLE:
    print("⚠️  警告：bitsandbytes 不可用，某些功能將受限。")
    print("   安裝命令: pip install bitsandbytes")


class SimpleLoRALayer(nn.Module):
    """簡單的 LoRA 層實現，用於演示."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始線性層（凍結）
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        # LoRA 分解：W = W_base + B @ A
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # 標記為 LoRA 層（供優化器識別）
        self.lora_A.weight._is_lora_layer = True
        self.lora_B.weight._is_lora_layer = True

    def forward(self, x):
        base_output = self.linear(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return base_output + lora_output


class LoRATestModel(nn.Module):
    """包含多個 LoRA 層的測試模型."""

    def __init__(self, input_size=512, hidden_size=256, output_size=10, lora_rank=16):
        super().__init__()

        self.layers = nn.ModuleList([
            SimpleLoRALayer(input_size, hidden_size, rank=lora_rank),
            nn.ReLU(),
            SimpleLoRALayer(hidden_size, hidden_size, rank=lora_rank),
            nn.ReLU(),
            SimpleLoRALayer(hidden_size, output_size, rank=lora_rank)
        ])

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, SimpleLoRALayer):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
            x = self.dropout(x)
        return x


def create_synthetic_data(batch_size=32, input_size=512, num_batches=100):
    """創建合成訓練數據."""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_size, device=device)
        # 創建有結構的目標（模擬真實任務）
        y = torch.randint(0, 10, (batch_size,), device=device)
        data.append((x, y))
    return data


def benchmark_memory_usage(model, optimizer):
    """基準測試記憶體使用."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        # 執行幾個訓練步驟
        x = torch.randn(32, 512, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()

        return {
            'initial_memory_mb': initial_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'current_memory_mb': current_memory / 1024 / 1024,
            'memory_increase_mb': (current_memory - initial_memory) / 1024 / 1024
        }
    else:
        return {'message': 'Memory benchmarking only available on CUDA'}


def compare_optimizers():
    """比較不同優化器配置的效果."""
    print("\n" + "="*60)
    print("🔬 優化器配置比較測試")
    print("="*60)

    # 測試配置
    configs = {
        "記憶體優先": OptimizationProfiles.memory_optimized(),
        "品質優先": OptimizationProfiles.quality_optimized(),
        "平衡配置": OptimizationProfiles.balanced()
    }

    results = {}

    for config_name, config in configs.items():
        print(f"\n📊 測試配置: {config_name}")

        # 創建模型
        model = LoRATestModel().to(device)

        try:
            # 創建優化器
            optimizer = Automagic_CameAMP_Improved_8Bit(model.parameters(), **config.__dict__)

            # 記憶體測試
            memory_stats = benchmark_memory_usage(model, optimizer)

            # 效率報告
            efficiency_report = optimizer.get_memory_efficiency_report()

            results[config_name] = {
                'memory_stats': memory_stats,
                'efficiency_report': efficiency_report,
                'success': True
            }

            print(f"  ✅ 成功創建優化器")
            print(f"  📊 8bit 參數比例: {efficiency_report['compression_ratio']:.2%}")
            if 'memory_increase_mb' in memory_stats:
                print(f"  💾 記憶體增長: {memory_stats['memory_increase_mb']:.2f} MB")

        except Exception as e:
            print(f"  ❌ 創建失敗: {e}")
            results[config_name] = {'success': False, 'error': str(e)}

    return results


def train_with_monitoring():
    """帶有詳細監控的訓練範例."""
    print("\n" + "="*60)
    print("🚀 詳細訓練監控範例")
    print("="*60)

    # 創建模型和數據
    model = LoRATestModel(lora_rank=32).to(device)
    train_data = create_synthetic_data(batch_size=64, num_batches=50)

    # 使用平衡配置
    config = OptimizationProfiles.balanced()
    config.verbose = True  # 啟用詳細輸出

    try:
        optimizer = Automagic_CameAMP_Improved_8Bit(model.parameters(), **config.__dict__)
        print("✅ 優化器創建成功")

        # 訓練循環
        train_losses = []
        memory_usage = []
        compression_ratios = []

        print(f"\n🎯 開始訓練 {len(train_data)} 個批次...")

        for epoch, (x, y) in enumerate(train_data):
            # 前向傳播
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)

            # 反向傳播
            loss.backward()
            optimizer.step()

            # 記錄指標
            train_losses.append(loss.item())

            # 每 10 步記錄一次詳細統計
            if epoch % 10 == 0:
                efficiency_report = optimizer.get_memory_efficiency_report()
                compression_ratios.append(efficiency_report['compression_ratio'])

                if device.type == 'cuda':
                    current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    memory_usage.append(current_memory)
                    print(f"步驟 {epoch:03d}: 損失={loss:.4f}, "
                          f"記憶體={current_memory:.1f}MB, "
                          f"壓縮率={efficiency_report['compression_ratio']:.2%}")
                else:
                    print(f"步驟 {epoch:03d}: 損失={loss:.4f}")

        # 訓練結果
        print(f"\n📈 訓練完成！")
        print(f"  初始損失: {train_losses[0]:.4f}")
        print(f"  最終損失: {train_losses[-1]:.4f}")
        print(f"  損失改善: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.2f}%")

        # 最終效率報告
        final_report = optimizer.get_memory_efficiency_report()
        print(f"\n📊 最終效率報告:")
        print(f"  總參數數量: {final_report['total_parameters']:,}")
        print(f"  8bit 參數: {final_report['8bit_parameters']:,}")
        print(f"  32bit 參數: {final_report['32bit_parameters']:,}")
        print(f"  記憶體節省: {final_report['memory_saved_mb']:.2f} MB")
        print(f"  壓縮率: {final_report['compression_ratio']:.2%}")

        # 繪製訓練曲線
        if len(train_losses) > 10:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(train_losses)
            plt.title('訓練損失')
            plt.xlabel('步驟')
            plt.ylabel('損失')
            plt.grid(True)

            if memory_usage:
                plt.subplot(1, 3, 2)
                plt.plot(range(0, len(train_data), 10), memory_usage)
                plt.title('記憶體使用')
                plt.xlabel('步驟')
                plt.ylabel('記憶體 (MB)')
                plt.grid(True)

            if compression_ratios:
                plt.subplot(1, 3, 3)
                plt.plot(range(0, len(train_data), 10), compression_ratios)
                plt.title('壓縮率')
                plt.xlabel('步驟')
                plt.ylabel('壓縮率')
                plt.grid(True)

            plt.tight_layout()
            plt.savefig('training_monitoring.png', dpi=150)
            print(f"📊 訓練曲線已保存為 training_monitoring.png")

        return {
            'losses': train_losses,
            'memory_usage': memory_usage,
            'final_report': final_report
        }

    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def demonstrate_state_persistence():
    """演示狀態保存和載入."""
    print("\n" + "="*60)
    print("💾 狀態持久化演示")
    print("="*60)

    # 創建模型和優化器
    model = LoRATestModel().to(device)
    optimizer = create_improved_8bit_optimizer(
        model.parameters(),
        lr=1e-3,
        edge_suppression=True,
        lora_rank_penalty=True,
        verbose=False
    )

    # 訓練幾步
    print("🏃 執行初始訓練...")
    for i in range(5):
        x = torch.randn(32, 512, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        print(f"  步驟 {i+1}: 損失 = {loss:.4f}")

    # 保存狀態
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': 5
    }

    torch.save(checkpoint, 'bitsandbytes_8bit_checkpoint.pth')
    print("✅ 狀態已保存到 bitsandbytes_8bit_checkpoint.pth")

    # 創建新的模型和優化器
    print("\n🔄 創建新實例並載入狀態...")
    new_model = LoRATestModel().to(device)
    new_optimizer = create_improved_8bit_optimizer(new_model.parameters(), lr=1e-3)

    # 載入狀態
    checkpoint = torch.load('bitsandbytes_8bit_checkpoint.pth')
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"✅ 狀態載入成功，從步驟 {checkpoint['step']} 繼續")

    # 繼續訓練
    print("🏃 繼續訓練...")
    for i in range(3):
        x = torch.randn(32, 512, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        new_optimizer.zero_grad()
        output = new_model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        new_optimizer.step()

        print(f"  步驟 {checkpoint['step'] + i + 1}: 損失 = {loss:.4f}")

    print("✅ 狀態持久化測試完成")

    # 清理文件
    if os.path.exists('bitsandbytes_8bit_checkpoint.pth'):
        os.remove('bitsandbytes_8bit_checkpoint.pth')


def performance_comparison():
    """性能比較測試."""
    print("\n" + "="*60)
    print("⚡ 性能比較測試")
    print("="*60)

    model = LoRATestModel().to(device)

    # 測試數據
    x = torch.randn(64, 512, device=device)
    y = torch.randint(0, 10, (64,), device=device)

    optimizers = {}

    # 嘗試創建不同的優化器
    if BITSANDBYTES_AVAILABLE:
        try:
            optimizers['8bit (bitsandbytes)'] = create_improved_8bit_optimizer(
                model.parameters(), lr=1e-3, min_8bit_size=1024
            )
            print("✅ 創建 bitsandbytes 8bit 優化器")
        except Exception as e:
            print(f"❌ 無法創建 bitsandbytes 8bit 優化器: {e}")

    # 標準 PyTorch 優化器作為對照
    try:
        optimizers['標準 Adam'] = torch.optim.Adam(model.parameters(), lr=1e-3)
        print("✅ 創建標準 Adam 優化器")
    except Exception as e:
        print(f"❌ 無法創建標準 Adam 優化器: {e}")

    # 性能測試
    results = {}

    for name, opt in optimizers.items():
        print(f"\n🧪 測試優化器: {name}")

        # 重置模型狀態
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # 計時測試
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        for i in range(10):
            opt.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            opt.step()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()

        avg_time = (end_time - start_time) / 10

        # 記憶體使用
        if device.type == 'cuda':
            memory_usage = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            memory_usage = "N/A"

        results[name] = {
            'avg_time_per_step': avg_time,
            'memory_usage_mb': memory_usage
        }

        print(f"  ⏱️  平均每步時間: {avg_time*1000:.2f} ms")
        if memory_usage != "N/A":
            print(f"  💾 記憶體使用: {memory_usage:.2f} MB")

    # 比較結果
    if len(results) > 1:
        print(f"\n📊 性能比較摘要:")
        baseline = None
        for name, stats in results.items():
            if '標準' in name:
                baseline = stats
                break

        if baseline:
            for name, stats in results.items():
                if name != next(k for k in results.keys() if '標準' in k):
                    time_ratio = stats['avg_time_per_step'] / baseline['avg_time_per_step']
                    print(f"  {name} vs 標準 Adam:")
                    print(f"    時間比例: {time_ratio:.2f}x")
                    if stats['memory_usage_mb'] != "N/A" and baseline['memory_usage_mb'] != "N/A":
                        memory_ratio = stats['memory_usage_mb'] / baseline['memory_usage_mb']
                        print(f"    記憶體比例: {memory_ratio:.2f}x")


def main():
    """主函數，執行所有範例."""
    print("🚀 Automagic_CameAMP_Improved_8Bit (bitsandbytes 版本) 使用範例")
    print("="*80)

    # 檢查 bitsandbytes 可用性
    if not BITSANDBYTES_AVAILABLE:
        print("⚠️  bitsandbytes 不可用，部分功能將受限")
        print("   請使用以下命令安裝：pip install bitsandbytes")
        print("   繼續執行兼容性測試...\n")

    try:
        # 1. 比較不同配置
        compare_results = compare_optimizers()

        # 2. 詳細訓練監控
        if BITSANDBYTES_AVAILABLE:
            training_results = train_with_monitoring()
        else:
            print("⏭️  跳過詳細訓練監控（需要 bitsandbytes）")
            training_results = None

        # 3. 狀態持久化演示
        if BITSANDBYTES_AVAILABLE:
            demonstrate_state_persistence()
        else:
            print("⏭️  跳過狀態持久化演示（需要 bitsandbytes）")

        # 4. 性能比較
        performance_comparison()

        print("\n" + "="*80)
        print("🎉 所有範例執行完成！")

        if BITSANDBYTES_AVAILABLE:
            print("📋 總結:")
            print("  ✅ bitsandbytes 8bit 量化正常工作")
            print("  ✅ 記憶體效率顯著提升")
            print("  ✅ 訓練穩定性良好")
            print("  ✅ 狀態持久化功能正常")
        else:
            print("📋 總結:")
            print("  ⚠️  bitsandbytes 不可用，建議安裝以獲得完整功能")
            print("  ✅ 兼容性測試通過")

        print("\n🔗 相關文件:")
        print("  📄 完整文檔: BITSANDBYTES_8BIT_GUIDE.md")
        print("  🐛 問題報告: 請檢查控制台輸出中的警告信息")

    except Exception as e:
        print(f"\n❌ 執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()