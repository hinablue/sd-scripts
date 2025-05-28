# Automagic_CameAMP 快速入門指南

## 🚀 5分鐘上手

### 1. 基本安裝

```bash
# 確保已安裝 PyTorch
pip install torch torchvision

# 可選：安裝 8-bit 支援
pip install bitsandbytes

# 可選：安裝繪圖支援
pip install matplotlib
```

### 2. 最簡單的使用

```python
from library.automagic_cameamp import Automagic_CameAMP
import torch
import torch.nn as nn

# 創建模型
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# 創建優化器
optimizer = Automagic_CameAMP(
    model.parameters(),
    lr=1e-3,          # 學習率
    weight_decay=1e-4  # 權重衰減
)

# 訓練循環
for epoch in range(100):
    # 你的數據
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    # 訓練步驟
    optimizer.zero_grad()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## 🎯 選擇合適的版本

### 快速決策樹

```
需要節省記憶體嗎？
├─ 是 → 需要智能調整嗎？
│      ├─ 是 → Automagic_CameAMP_COptim8bit
│      └─ 否 → Automagic_CameAMP8bit
└─ 否 → 需要智能調整嗎？
       ├─ 是 → Automagic_CameAMP_COptim
       └─ 否 → Automagic_CameAMP
```

### 場景推薦

| 場景 | 推薦版本 | 理由 |
|------|----------|------|
| 初學者、小模型 | `Automagic_CameAMP` | 穩定可靠 |
| 大模型訓練 | `Automagic_CameAMP8bit` | 節省記憶體 |
| 研究實驗 | `Automagic_CameAMP_COptim` | 智能調整 |
| 生產環境 | `Automagic_CameAMP_COptim8bit` | 全功能 |

## 🛠️ 常用配置

### 1. 小模型 (< 100M 參數)

```python
optimizer = Automagic_CameAMP(
    model.parameters(),
    lr=1e-3,
    warmup_steps=500,
    weight_decay=1e-4
)
```

### 2. 中等模型 (100M - 1B 參數)

```python
optimizer = Automagic_CameAMP8bit(
    model.parameters(),
    lr=5e-4,
    warmup_steps=1000,
    weight_decay=5e-4
)
```

### 3. 大模型 (> 1B 參數)

```python
optimizer = Automagic_CameAMP_COptim8bit(
    model.parameters(),
    lr=1e-4,
    warmup_steps=2000,
    weight_decay=1e-3,
    context_window=50,
    edge_threshold=0.8
)
```

### 4. LoRA 微調

```python
optimizer = Automagic_CameAMP(
    model.parameters(),
    lr=1e-3,
    full_finetune=False,  # 啟用 ALLoRA
    eta=2.0,
    warmup_steps=100
)
```

## 📊 監控訓練狀態

### 基礎監控

```python
# 在訓練循環中添加
if step % 100 == 0:
    # 獲取當前學習率
    current_lr = optimizer.param_groups[0]['lr']

    # 如果是 Automagic 優化器，獲取平均學習率
    if hasattr(optimizer, '_get_group_lr'):
        avg_lr = optimizer._get_group_lr(optimizer.param_groups[0])
        print(f"Step {step}: Loss = {loss:.4f}, LR = {avg_lr:.6f}")
    else:
        print(f"Step {step}: Loss = {loss:.4f}, LR = {current_lr:.6f}")
```

### 高級監控 (C-Optim 版本)

```python
# C-Optim 版本的額外監控
if hasattr(optimizer, 'c_optim'):
    lr_multiplier = optimizer.c_optim.compute_contextual_lr_multiplier()
    is_edge_case = optimizer.c_optim.detect_edge_case()
    grad_consistency = optimizer.c_optim.compute_gradient_consistency()

    print(f"上下文乘數: {lr_multiplier:.3f}")
    print(f"邊緣情況: {is_edge_case}")
    print(f"梯度一致性: {grad_consistency:.3f}")
```

## 🔧 常見問題解決

### 1. 8-bit 初始化失敗

```python
try:
    optimizer = Automagic_CameAMP8bit(model.parameters(), lr=1e-3)
except RuntimeError as e:
    print(f"8-bit 不可用: {e}")
    # 回退到 32-bit 版本
    optimizer = Automagic_CameAMP(model.parameters(), lr=1e-3)
```

### 2. 學習率太小

```python
# 檢查學習率遮罩
for group in optimizer.param_groups:
    for p in group['params']:
        if p.grad is not None:
            state = optimizer.state[p]
            if 'lr_mask' in state:
                lr_mask = state['lr_mask']
                print(f"LR 範圍: {lr_mask.min():.6f} - {lr_mask.max():.6f}")
```

### 3. 記憶體不足

```python
# 使用 8-bit 版本
optimizer = Automagic_CameAMP8bit(model.parameters(), lr=1e-3)

# 或者減少批次大小
batch_size = 16  # 從 64 減少到 16
```

### 4. 訓練不穩定

```python
# 降低學習率
optimizer = Automagic_CameAMP(
    model.parameters(),
    lr=5e-4,  # 從 1e-3 降低到 5e-4
    weight_decay=1e-3  # 增加正則化
)

# 或使用 C-Optim 版本的自動調整
optimizer = Automagic_CameAMP_COptim(
    model.parameters(),
    lr=1e-3,
    edge_threshold=0.8,  # 更敏感的邊緣檢測
    adaptation_rate=0.2   # 更積極的調整
)
```

## 🎯 實用技巧

### 1. 學習率調度器配合

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = Automagic_CameAMP(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    # 訓練...
    scheduler.step()
```

### 2. 梯度裁剪

```python
import torch.nn.utils as utils

for epoch in range(100):
    optimizer.zero_grad()
    loss = model(x, y)
    loss.backward()

    # 梯度裁剪
    utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

### 3. 混合精度訓練

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = Automagic_CameAMP(model.parameters(), lr=1e-3)

for epoch in range(100):
    optimizer.zero_grad()

    with autocast():
        output = model(x)
        loss = criterion(output, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4. 檢查點保存

```python
# 保存
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, 'checkpoint.pth')

# 載入
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## ⚡ 性能調優

### 1. 暖身步數調整

```python
# 小模型：快速暖身
warmup_steps = 500

# 大模型：較長暖身
warmup_steps = 2000

# 微調：短暖身
warmup_steps = 100
```

### 2. C-Optim 參數調優

```python
# 穩定訓練
optimizer = Automagic_CameAMP_COptim(
    model.parameters(),
    context_window=100,   # 較大窗口，更穩定
    edge_threshold=0.9,   # 較高閾值，保守調整
    adaptation_rate=0.1   # 較低速率，緩慢調整
)

# 快速調整
optimizer = Automagic_CameAMP_COptim(
    model.parameters(),
    context_window=30,    # 較小窗口，更靈敏
    edge_threshold=0.7,   # 較低閾值，積極調整
    adaptation_rate=0.3   # 較高速率，快速調整
)
```

## 📝 完整示例

```python
#!/usr/bin/env python3
"""Automagic_CameAMP 完整訓練示例"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from library.automagic_cameamp import Automagic_CameAMP_COptim

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 創建模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)

# 初始化
model = SimpleModel().to(device)
optimizer = Automagic_CameAMP_COptim(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    warmup_steps=1000,
    context_window=50,
    edge_threshold=0.8,
    verbose=True
)

# 訓練循環
for epoch in range(100):
    model.train()

    # 假設有 DataLoader
    for batch_idx in range(100):  # 模擬 100 個批次
        # 生成假數據
        data = torch.randn(64, 784, device=device)
        target = torch.randint(0, 10, (64,), device=device)

        # 訓練步驟
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # 監控
        if batch_idx % 20 == 0:
            avg_lr = optimizer._get_group_lr(optimizer.param_groups[0])
            lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()
            is_edge = optimizer.c_optim.detect_edge_case()

            print(f'Epoch {epoch}, Batch {batch_idx}: '
                  f'Loss={loss:.4f}, LR={avg_lr:.6f}, '
                  f'Mult={lr_mult:.3f}, Edge={is_edge}')

print("✅ 訓練完成！")
```

## 🎉 恭喜！

您已經掌握了 Automagic_CameAMP 的基本使用方法。更多詳細信息請參考：

- 📖 完整說明文件：`README_Automagic_CameAMP.md`
- 🧪 測試範例：`test_automagic_cameamp.py`
- 🔬 進階功能：`README_COptim_Improvements.md`

開始您的高效訓練之旅吧！🚀