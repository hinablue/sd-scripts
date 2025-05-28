#!/usr/bin/env python3
"""
Automagic_CameAMP_COptim8bit 優化器使用範例

這個範例展示了如何使用結合 C-Optim 上下文優化和 8-bit 量化的優化器。
包含：
1. 基本初始化和配置
2. 訓練循環中的使用
3. 記憶體使用比較
4. 邊緣情況檢測和處理
5. 狀態保存和載入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
from library.automagic_cameamp import Automagic_CameAMP_COptim, Automagic_CameAMP_COptim8bit

class AdvancedModel(nn.Module):
    """更複雜的模型用於測試優化器性能"""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_memory_usage():
    """獲取當前記憶體使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_synthetic_data(batch_size=64, input_size=784, num_classes=10):
    """創建合成數據用於測試"""
    x = torch.randn(batch_size, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y

def train_with_optimizer(model, optimizer, num_epochs=5, batch_size=64):
    """使用指定優化器訓練模型"""
    model.train()
    device = next(model.parameters()).device

    print(f"\n開始訓練 - 優化器: {optimizer.__class__.__name__}")
    print(f"設備: {device}")

    start_memory = get_memory_usage()
    start_time = time.time()

    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for step in range(20):  # 每個 epoch 20 步
            # 創建批次數據
            x, y = create_synthetic_data(batch_size)
            x, y = x.to(device), y.to(device)

            # 前向傳播
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()

            # 優化器步驟
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)

        # 檢查 C-Optim 狀態（如果可用）
        if hasattr(optimizer, 'c_optim'):
            edge_case = optimizer.c_optim.detect_edge_case()
            lr_multiplier = optimizer.c_optim.compute_contextual_lr_multiplier()
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                  f"Edge Case={edge_case}, LR Multiplier={lr_multiplier:.4f}")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

    end_time = time.time()
    end_memory = get_memory_usage()

    print(f"訓練完成！")
    print(f"訓練時間: {end_time - start_time:.2f} 秒")
    print(f"記憶體使用: {end_memory - start_memory:.2f} MB")
    print(f"最終損失: {losses[-1]:.4f}")

    return losses, end_memory - start_memory

def compare_optimizers():
    """比較不同優化器的性能"""
    print("=" * 60)
    print("Automagic_CameAMP_COptim vs Automagic_CameAMP_COptim8bit 比較")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 測試配置
    configs = [
        {
            'name': '32-bit C-Optim',
            'optimizer_class': Automagic_CameAMP_COptim,
            'kwargs': {
                'lr': 1e-3,
                'context_window': 50,
                'edge_threshold': 0.9,
                'adaptation_rate': 0.15,
                'momentum_scales': [1, 3, 7, 15]
            }
        },
        {
            'name': '8-bit C-Optim',
            'optimizer_class': Automagic_CameAMP_COptim8bit,
            'kwargs': {
                'lr': 1e-3,
                'context_window': 50,
                'edge_threshold': 0.9,
                'adaptation_rate': 0.15,
                'momentum_scales': [1, 3, 7, 15]
            }
        }
    ]

    results = {}

    for config in configs:
        print(f"\n{'='*40}")
        print(f"測試: {config['name']}")
        print(f"{'='*40}")

        # 創建新模型
        model = AdvancedModel().to(device)

        # 創建優化器
        try:
            optimizer = config['optimizer_class'](
                model.parameters(),
                **config['kwargs']
            )

            # 訓練
            losses, memory_usage = train_with_optimizer(model, optimizer)

            results[config['name']] = {
                'final_loss': losses[-1],
                'memory_usage': memory_usage,
                'losses': losses
            }

        except Exception as e:
            print(f"錯誤: {e}")
            results[config['name']] = {
                'error': str(e)
            }

    # 顯示比較結果
    print(f"\n{'='*60}")
    print("比較結果")
    print(f"{'='*60}")

    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: 錯誤 - {result['error']}")
        else:
            print(f"{name}:")
            print(f"  最終損失: {result['final_loss']:.4f}")
            print(f"  記憶體使用: {result['memory_usage']:.2f} MB")

    return results

def test_state_save_load():
    """測試狀態保存和載入功能"""
    print(f"\n{'='*40}")
    print("測試狀態保存和載入")
    print(f"{'='*40}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedModel().to(device)

    # 創建 8-bit C-Optim 優化器
    optimizer = Automagic_CameAMP_COptim8bit(
        model.parameters(),
        lr=1e-3,
        context_window=30,
        edge_threshold=0.85
    )

    # 訓練幾步
    print("訓練幾步以建立狀態...")
    for step in range(5):
        x, y = create_synthetic_data()
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = F.cross_entropy(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}: Loss={loss.item():.4f}")

    # 保存狀態
    print("\n保存優化器狀態...")
    state_dict = optimizer.state_dict()
    torch.save(state_dict, 'optimizer_state.pth')

    # 創建新的優化器並載入狀態
    print("創建新優化器並載入狀態...")
    new_optimizer = Automagic_CameAMP_COptim8bit(
        model.parameters(),
        lr=1e-3,
        context_window=30,
        edge_threshold=0.85
    )

    loaded_state = torch.load('optimizer_state.pth')
    new_optimizer.load_state_dict(loaded_state)

    print("狀態載入成功！")

    # 清理
    if os.path.exists('optimizer_state.pth'):
        os.remove('optimizer_state.pth')

def demonstrate_edge_case_handling():
    """展示邊緣情況檢測和處理"""
    print(f"\n{'='*40}")
    print("邊緣情況檢測演示")
    print(f"{'='*40}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedModel().to(device)

    optimizer = Automagic_CameAMP_COptim8bit(
        model.parameters(),
        lr=1e-3,
        edge_threshold=0.7,  # 較低的閾值，更容易觸發邊緣情況
        adaptation_rate=0.2
    )

    print("監控邊緣情況檢測...")

    for step in range(15):
        # 創建一些"困難"的數據來觸發邊緣情況
        if step > 5:
            # 增加噪音來模擬困難的優化情況
            x, y = create_synthetic_data()
            x += torch.randn_like(x) * 0.5  # 添加噪音
        else:
            x, y = create_synthetic_data()

        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = F.cross_entropy(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 檢查邊緣情況
        is_edge = optimizer.c_optim.detect_edge_case()
        lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()

        status = "🔴 邊緣情況" if is_edge else "🟢 正常"
        print(f"Step {step+1:2d}: Loss={loss.item():.4f}, "
              f"LR倍數={lr_mult:.3f}, 狀態={status}")

if __name__ == "__main__":
    print("Automagic_CameAMP_COptim8bit 優化器測試")
    print("=" * 60)

    try:
        # 1. 比較優化器性能
        results = compare_optimizers()

        # 2. 測試狀態保存載入
        test_state_save_load()

        # 3. 展示邊緣情況處理
        demonstrate_edge_case_handling()

        print(f"\n{'='*60}")
        print("所有測試完成！")
        print(f"{'='*60}")

    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()