#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
記憶體優化版本 AdaptiveHinaAdamW 優化器使用示例

這個示例展示如何在 Flux 模型訓練中使用記憶體優化的優化器，
特別針對 16GB VRAM 的限制進行了優化。
"""

import torch
import torch.nn as nn
from library.custom_hina_adaptive_adamw_memory_optimized import MemoryOptimizedAdaptiveHinaAdamW


def create_example_model():
    """創建一個示例模型（類似 LoRA 結構）"""
    class SimpleLoRALayer(nn.Module):
        def __init__(self, in_features: int, out_features: int, rank: int = 64):
            super().__init__()
            self.down_proj = nn.Linear(in_features, rank, bias=False)
            self.up_proj = nn.Linear(rank, out_features, bias=False)
            self.scaling = 1.0 / rank

        def forward(self, x):
            return self.up_proj(self.down_proj(x)) * self.scaling

    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 模擬大型模型的層
            self.layers = nn.ModuleList([
                SimpleLoRALayer(4096, 4096, rank=128) for _ in range(24)
            ])
            self.output_proj = nn.Linear(4096, 1000)

        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x)  # 殘差連接
            return self.output_proj(x)

    return ExampleModel()


def demonstrate_basic_usage():
    """基本使用示例"""
    print("=== 記憶體優化版本 AdaptiveHinaAdamW 基本使用示例 ===")

    # 創建模型
    model = create_example_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"模型設備: {device}")
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")

    # 創建記憶體優化的優化器
    optimizer = MemoryOptimizedAdaptiveHinaAdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        # 記憶體優化設置
        memory_efficient=True,
        vram_budget_gb=16.0,  # 16GB VRAM 預算
        cpu_offload_states=True,  # 將部分狀態移到 CPU
        reduce_precision=True,    # 使用低精度
        adaptive_features=True,   # 啟用自適應功能
        emergency_simplify=True,  # 啟用緊急簡化
        max_buffer_memory_mb=300, # 限制緩衝區記憶體
        # 功能開關
        use_spd=True,
        use_cautious=True,
        use_orthogonal_grad=False,  # 在記憶體緊張時可以關閉
        use_adopt_stability=True,
        use_agr=True,
        use_tam=True,
        use_dynamic_adaptation=True,
        use_lr_mask=True,
        dynamic_weight_decay=True,
        # 間隔設置（降低計算頻率）
        relationship_discovery_interval=200,  # 每 200 步重新發現關係
        warmup_steps=1000
    )

    # 顯示優化器信息
    info = optimizer.get_optimization_info()
    print(f"\n優化器類型: {info['optimizer_type']}")
    print(f"版本: {info['version']}")
    print(f"記憶體預算: {info['memory_optimization']['vram_budget_gb']}GB")
    print(f"啟用的功能: {list(info['features'].keys())}")

    return model, optimizer


def demonstrate_training_with_memory_monitoring():
    """展示帶記憶體監控的訓練過程"""
    print("\n=== 記憶體監控訓練示例 ===")

    model, optimizer = demonstrate_basic_usage()
    device = next(model.parameters()).device

    # 模擬訓練循環
    batch_size = 4
    sequence_length = 512
    hidden_size = 4096

    for step in range(10):
        # 創建模擬輸入
        x = torch.randn(batch_size, sequence_length, hidden_size, device=device)
        target = torch.randn(batch_size, sequence_length, 1000, device=device)

        # 前向傳播
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)

        # 反向傳播
        loss.backward()

        # 優化器步驟（包含記憶體監控）
        optimizer.step()

        # 每幾步檢查記憶體狀態
        if step % 3 == 0:
            memory_stats = optimizer.get_memory_stats()
            print(f"\n步驟 {step}:")
            print(f"  記憶體壓力: {memory_stats['memory_pressure']:.2%}")

            if torch.cuda.is_available():
                cuda_stats = memory_stats['cuda_memory']
                print(f"  GPU 記憶體分配: {cuda_stats['allocated_gb']:.2f}GB")
                print(f"  GPU 記憶體保留: {cuda_stats['reserved_gb']:.2f}GB")

            # 檢查緩衝區使用情況
            buffer_stats = memory_stats['buffer_pool_stats']
            print(f"  緩衝區記憶體: {buffer_stats['current_memory_mb']:.1f}MB / {buffer_stats['max_memory_mb']:.1f}MB")

            print(f"  損失: {loss.item():.4f}")


def demonstrate_dynamic_optimization():
    """展示動態記憶體優化"""
    print("\n=== 動態記憶體優化示例 ===")

    model, optimizer = demonstrate_basic_usage()

    # 模擬不同的記憶體預算場景
    scenarios = [
        ("低記憶體場景", 8.0),
        ("標準記憶體場景", 16.0),
        ("高記憶體場景", 24.0)
    ]

    for scenario_name, vram_budget in scenarios:
        print(f"\n{scenario_name} (預算: {vram_budget}GB):")

        # 自動調整設置
        optimizer.optimize_for_vram(vram_budget)

        # 檢查調整後的設置
        info = optimizer.get_optimization_info()
        memory_config = info['memory_optimization']

        print(f"  CPU 卸載: {memory_config['cpu_offload_states']}")
        print(f"  精度降低: {memory_config['reduce_precision']}")
        print(f"  緊急簡化: {memory_config['emergency_simplify']}")
        print(f"  當前記憶體壓力: {info['current_memory_pressure']:.2%}")


def demonstrate_advanced_features():
    """展示進階功能"""
    print("\n=== 進階功能示例 ===")

    model, optimizer = demonstrate_basic_usage()
    device = next(model.parameters()).device

    # 模擬幾步訓練以累積統計數據
    for step in range(5):
        x = torch.randn(2, 256, 4096, device=device)
        target = torch.randn(2, 256, 1000, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

    # 獲取詳細信息
    if hasattr(optimizer, 'global_step') and optimizer.global_step > 0:
        info = optimizer.get_optimization_info()

        if 'training_stats' in info:
            stats = info['training_stats']
            print(f"訓練統計:")
            print(f"  全局步數: {stats['global_step']}")
            print(f"  參數關係數: {stats['total_relationships']}")
            print(f"  重要性分數數: {stats['total_importance_scores']}")
            print(f"  平均重要性: {stats['avg_importance_score']:.4f}")
            print(f"  待處理異步任務: {stats['pending_async_tasks']}")


def demonstrate_resource_cleanup():
    """展示資源清理"""
    print("\n=== 資源清理示例 ===")

    model, optimizer = demonstrate_basic_usage()

    # 獲取清理前的記憶體統計
    memory_stats_before = optimizer.get_memory_stats()
    print("清理前:")
    print(f"  緩衝區類型數: {memory_stats_before['buffer_pool_stats']['total_buffer_types']}")
    print(f"  緩衝區記憶體: {memory_stats_before['buffer_pool_stats']['current_memory_mb']:.1f}MB")

    # 清理資源
    optimizer.cleanup_resources()

    # 獲取清理後的記憶體統計
    memory_stats_after = optimizer.get_memory_stats()
    print("\n清理後:")
    print(f"  緩衝區類型數: {memory_stats_after['buffer_pool_stats']['total_buffer_types']}")
    print(f"  緩衝區記憶體: {memory_stats_after['buffer_pool_stats']['current_memory_mb']:.1f}MB")

    print("資源清理完成！")


def flux_training_example_config():
    """提供 Flux 模型訓練的建議配置"""
    print("\n=== Flux 模型訓練建議配置 ===")

    recommended_config = {
        # 基本設置
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2,
        'eps': 1e-8,

        # 記憶體優化設置
        'memory_efficient': True,
        'vram_budget_gb': 16.0,
        'cpu_offload_states': True,
        'reduce_precision': True,
        'adaptive_features': True,
        'emergency_simplify': True,
        'max_buffer_memory_mb': 200,

        # 功能開關（針對大型模型優化）
        'use_spd': True,
        'use_cautious': True,
        'use_orthogonal_grad': False,  # 關閉以節省記憶體
        'use_adopt_stability': True,
        'use_agr': True,
        'use_tam': False,  # 關閉以節省計算
        'use_dynamic_adaptation': True,
        'use_lr_mask': True,
        'dynamic_weight_decay': True,

        # 間隔設置（降低計算頻率）
        'relationship_discovery_interval': 500,
        'warmup_steps': 2000,

        # 自適應設置
        'adaptation_strength': 0.8,
        'importance_decay': 0.98,
        'compatibility_threshold': 0.4,

        # lr_mask 設置
        'lr_bump': 2e-6,
        'min_lr': 5e-8,
        'max_lr': 5e-4,

        # 動態權重衰減
        'wd_transition_steps': 1500,
        'wd_decay_factor': 0.8,
        'wd_min_ratio': 0.2
    }

    print("建議的 Flux 訓練配置:")
    for key, value in recommended_config.items():
        print(f"  {key}: {value}")

    print("\n使用方式:")
    print("```python")
    print("from library.custom_hina_adaptive_adamw_memory_optimized import MemoryOptimizedAdaptiveHinaAdamW")
    print("")
    print("optimizer = MemoryOptimizedAdaptiveHinaAdamW(")
    print("    model.parameters(),")
    for key, value in list(recommended_config.items())[:5]:
        print(f"    {key}={repr(value)},")
    print("    # ... 其他配置")
    print(")")
    print("```")


if __name__ == "__main__":
    print("記憶體優化版本 AdaptiveHinaAdamW 優化器示例")
    print("=" * 60)

    try:
        # 基本使用示例
        demonstrate_basic_usage()

        # 帶記憶體監控的訓練
        demonstrate_training_with_memory_monitoring()

        # 動態優化
        demonstrate_dynamic_optimization()

        # 進階功能
        demonstrate_advanced_features()

        # 資源清理
        demonstrate_resource_cleanup()

        # Flux 訓練配置建議
        flux_training_example_config()

    except Exception as e:
        print(f"示例運行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    print("\n示例運行完成！")