#!/usr/bin/env python3
"""
測試 HinaAdaptive 背景正則化不同模式的性能
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加 library 路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from library.hina_adaptive import HinaAdaptive


def create_test_model():
    """創建一個簡單的測試模型"""
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model


def test_background_regularization_mode(mode: str, num_steps: int = 100):
    """測試指定模式的背景正則化性能"""
    print(f"\n=== 測試 {mode} 模式 ===")

    # 創建模型和優化器
    model = create_test_model()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        background_regularization=True,
        background_regularization_mode=mode
    )

    # 創建假數據
    batch_size = 32
    input_size = 128
    if torch.cuda.is_available():
        data = torch.randn(batch_size, input_size).cuda()
        target = torch.randint(0, 10, (batch_size,)).cuda()
    else:
        data = torch.randn(batch_size, input_size)
        target = torch.randint(0, 10, (batch_size,))

    criterion = nn.CrossEntropyLoss()

    # 計時測試
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()

        # 前向傳播
        output = model(data)
        loss = criterion(output, target)

        # 反向傳播
        loss.backward()

        # 優化器步驟
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"總耗時: {total_time:.2f} 秒")
    print(f"平均每步耗時: {total_time/num_steps*1000:.2f} ms")

    return {'total_time': total_time, 'avg_time_per_step': total_time/num_steps}


def main():
    """主函數"""
    print("HinaAdaptive 背景正則化模式性能測試")
    print("=" * 50)

    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"設備: {device_info}")

    # 測試不同模式
    modes = ["simple", "fast"]
    results = {}

    for mode in modes:
        try:
            stats = test_background_regularization_mode(mode, num_steps=50)
            results[mode] = stats
        except Exception as e:
            print(f"測試 {mode} 模式時出錯: {e}")
            continue

    # 比較結果
    print("\n" + "=" * 50)
    print("性能比較摘要")
    print("=" * 50)

    for mode, stats in results.items():
        print(f"\n{mode.upper()} 模式:")
        print(f"  總耗時: {stats['total_time']:.2f} 秒")
        print(f"  平均每步耗時: {stats['avg_time_per_step']*1000:.2f} ms")

    # 性能建議
    if len(results) > 1:
        simple_time = results.get('simple', {}).get('avg_time_per_step', float('inf'))
        fast_time = results.get('fast', {}).get('avg_time_per_step', float('inf'))

        print(f"\n💡 性能建議:")
        if simple_time < fast_time:
            speedup = fast_time / simple_time if simple_time > 0 else 1
            print(f"  - Simple 模式比 Fast 模式快 {speedup:.1f}x")
            print(f"  - 建議使用 Simple 模式（默認）")
        else:
            speedup = simple_time / fast_time if fast_time > 0 else 1
            print(f"  - Fast 模式比 Simple 模式快 {speedup:.1f}x")
            print(f"  - 在此場景下 Fast 模式表現更好")


if __name__ == "__main__":
    main()