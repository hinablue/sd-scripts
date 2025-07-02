#!/usr/bin/env python3
"""
優化器配置檔案使用範例

展示如何在 optimizer_kwargs 中使用 profile 參數來載入不同的
Automagic_CameAMP_Improved_8Bit 優化器配置。

用法範例：
    # 在命令列中使用
    python script.py --optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=memory_optimized

    # 或組合多個參數
    python script.py --optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=quality_optimized lr=2e-4 verbose=True
"""

import torch
import torch.nn as nn
from library.automagic_cameamp_improved_8bit import (
    Automagic_CameAMP_Improved_8Bit,
    create_improved_8bit_optimizer,
    OptimizationProfiles
)


def example_1_basic_profile_usage():
    """範例1：基本 profile 使用"""
    print("=" * 60)
    print("範例1：基本 profile 使用")
    print("=" * 60)

    # 建立一個簡單的模型
    model = nn.Linear(10, 1)

    # 使用不同的配置檔案
    profiles = ['memory_optimized', 'quality_optimized', 'balanced']

    for profile in profiles:
        print(f"\n📋 測試配置檔案: {profile}")

        # 模擬 optimizer_kwargs 的方式
        optimizer_kwargs = {'profile': profile, 'verbose': True}

        optimizer = Automagic_CameAMP_Improved_8Bit(
            model.parameters(),
            **optimizer_kwargs
        )

        print(f"✅ 成功創建 {profile} 優化器")

        # 顯示記憶體效率報告
        report = optimizer.get_memory_efficiency_report()
        print(f"   bitsandbytes 可用: {report['bitsandbytes_available']}")


def example_2_profile_with_custom_params():
    """範例2：profile 配合自定義參數"""
    print("=" * 60)
    print("範例2：profile 配合自定義參數")
    print("=" * 60)

    model = nn.Linear(1000, 100)  # 更大的模型以測試 8bit

    # 從品質優化配置開始，但自定義學習率和 verbose
    optimizer_kwargs = {
        'profile': 'quality_optimized',
        'lr': 2e-4,  # 自定義學習率
        'verbose': True,
        'min_8bit_size': 2048  # 自定義最小 8bit 大小
    }

    print("🔧 配置參數:")
    for key, value in optimizer_kwargs.items():
        print(f"   {key}: {value}")

    optimizer = Automagic_CameAMP_Improved_8Bit(
        model.parameters(),
        **optimizer_kwargs
    )

    print("✅ 成功創建自定義優化器")


def example_3_simulate_train_util_usage():
    """範例3：模擬 train_util.py 中的使用方式"""
    print("=" * 60)
    print("範例3：模擬 train_util.py 中的使用方式")
    print("=" * 60)

    model = nn.Linear(500, 200)

    # 模擬 train_util.py 中解析 optimizer_args 的過程
    # 假設命令列參數為：--optimizer_args profile=balanced lr=1e-4 verbose=True

    # 這是 train_util.py 中構建 optimizer_kwargs 的方式
    optimizer_kwargs = {}

    # 模擬解析的參數
    parsed_args = [
        "profile=balanced",
        "lr=1e-4",
        "verbose=True",
        "edge_penalty=0.12"
    ]

    # 模擬 train_util.py 中的解析邏輯
    import ast
    for arg in parsed_args:
        key, value = arg.split("=")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # 如果不能解析，保持字串
            pass
        optimizer_kwargs[key] = value

    print("🔧 解析得到的 optimizer_kwargs:")
    for key, value in optimizer_kwargs.items():
        print(f"   {key}: {value} (type: {type(value).__name__})")

    # 創建優化器（模擬 train_util.py 中的邏輯）
    lr = optimizer_kwargs.pop('lr', 1e-6)  # 提取學習率
    optimizer = Automagic_CameAMP_Improved_8Bit(
        model.parameters(),
        lr=lr,
        **optimizer_kwargs
    )

    print("✅ 成功創建優化器，模擬 train_util.py 使用方式")


def example_4_profile_comparison():
    """範例4：不同 profile 的比較"""
    print("=" * 60)
    print("範例4：不同 profile 的比較")
    print("=" * 60)

    model = nn.Linear(2048, 512)  # 較大模型

    profiles_info = {
        'memory_optimized': '記憶體優化 - 最大記憶體節省',
        'quality_optimized': '品質優化 - 最佳訓練效果',
        'balanced': '平衡配置 - 記憶體與品質兼顧'
    }

    optimizers = {}

    for profile, description in profiles_info.items():
        print(f"\n📋 {profile}: {description}")

        optimizer = create_improved_8bit_optimizer(
            model.parameters(),
            profile=profile,
            verbose=False  # 關閉詳細輸出以便比較
        )

        optimizers[profile] = optimizer

        # 顯示關鍵配置差異
        config = optimizer.config
        print(f"   邊緣抑制: {config.edge_suppression}")
        print(f"   空間感知: {config.spatial_awareness}")
        print(f"   LoRA 優化: {config.lora_rank_penalty}")
        print(f"   最小 8bit 大小: {config.min_8bit_size}")
        print(f"   強制 8bit: {config.force_8bit}")


def example_5_available_profiles():
    """範例5：顯示所有可用的預定義配置"""
    print("=" * 60)
    print("範例5：可用的預定義配置")
    print("=" * 60)

    profiles = {
        'memory_optimized': OptimizationProfiles.memory_optimized(),
        'quality_optimized': OptimizationProfiles.quality_optimized(),
        'balanced': OptimizationProfiles.balanced()
    }

    for name, config in profiles.items():
        print(f"\n📋 {name}:")
        print(f"   說明: {config.__class__.__name__}")

        # 顯示主要配置參數
        key_params = [
            'lr', 'edge_suppression', 'edge_penalty', 'background_regularization',
            'spatial_awareness', 'lora_rank_penalty', 'min_8bit_size', 'force_8bit'
        ]

        for param in key_params:
            if hasattr(config, param):
                value = getattr(config, param)
                print(f"   {param}: {value}")


def command_line_examples():
    """命令列使用範例"""
    print("=" * 60)
    print("命令列使用範例")
    print("=" * 60)

    examples = [
        {
            "description": "使用記憶體優化配置",
            "command": "--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=memory_optimized"
        },
        {
            "description": "使用品質優化配置並自定義學習率",
            "command": "--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=quality_optimized lr=2e-4"
        },
        {
            "description": "使用平衡配置並啟用詳細輸出",
            "command": "--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=balanced verbose=True"
        },
        {
            "description": "不使用 profile，完全自定義配置",
            "command": "--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args lr=1e-4 edge_suppression=True min_8bit_size=2048"
        },
        {
            "description": "組合使用：基於 balanced 配置但覆蓋部分參數",
            "command": "--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=balanced lr=3e-4 edge_penalty=0.15 verbose=True"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n範例 {i}: {example['description']}")
        print(f"命令: {example['command']}")


if __name__ == "__main__":
    print("🚀 Automagic_CameAMP_Improved_8Bit 優化器配置檔案使用範例")
    print(f"🔧 bitsandbytes 可用: {torch.cuda.is_available()}")

    try:
        # 執行所有範例
        example_1_basic_profile_usage()
        example_2_profile_with_custom_params()
        example_3_simulate_train_util_usage()
        example_4_profile_comparison()
        example_5_available_profiles()
        command_line_examples()

        print("\n" + "=" * 60)
        print("✅ 所有範例執行完成！")
        print("📚 現在您可以在 sd-scripts 訓練腳本中使用這些配置了")

    except Exception as e:
        print(f"\n❌ 執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()