# Automagic_CameAMP 優化器完整說明

## 📋 概述

`Automagic_CameAMP` 是一個先進的深度學習優化器，整合了多種最新的優化技術，提供高效且穩定的模型訓練體驗。該優化器系列包含四個版本，從基礎到高級功能逐步增強。

## 🚀 優化器版本對比

### 版本概覽

| 版本 | 特性 | 記憶體使用 | 適用場景 |
|------|------|------------|----------|
| `Automagic_CameAMP` | 基礎版本 | 100% | 一般訓練，穩定可靠 |
| `Automagic_CameAMP8bit` | 8-bit 量化 | ~25% | 大模型，記憶體受限 |
| `Automagic_CameAMP_COptim` | 上下文感知 | 100% | 高級訓練，智能調整 |
| `Automagic_CameAMP_COptim8bit` | 全功能版 | ~30% | 大模型高級訓練 |

## 🔧 核心技術

### 1. **CAME (Confidence-guided Adaptive Memory Efficient Optimization)**
```python
# 自信心引導的自適應記憶高效優化
exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
```
- **論文**: [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/pdf/2411.02853)
- **功能**: 提供穩定且高效的梯度縮放機制

### 2. **AGR (Adaptive Gradient Regularization)**
```python
# 自適應梯度正則化
abs_grad = torch.abs(p.grad)
alpha = abs_grad / sum_abs_all_group_grads
grad = p.grad * (1 - alpha)
```
- **論文**: [Adaptive Gradient Regularization](https://arxiv.org/pdf/2407.16944)
- **功能**: 減少梯度噪音，提高訓練穩定性

### 3. **Torque-Aware Momentum (TAM)**
```python
# 扭矩感知動量（早期訓練）
corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(scaled_grad, p=2.0, dim=0))
s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)
```
- **論文**: [Torque-Aware Momentum](https://arxiv.org/abs/2412.18790)
- **功能**: 早期訓練階段的智能動量調整

### 4. **Consistency Momentum**
```python
# 一致性動量（後期訓練）
beta1_t = max(beta1 * group['beta1_decay'] ** state["step"], 0.4)
exp_avg.mul_(beta1_t).add_(scaled_grad, alpha=1 - beta1_t)
```
- **論文**: [Towards Faster Training of Diffusion Models](https://arxiv.org/abs/2404.07946)
- **功能**: 基於一致性現象的動量更新

### 5. **Grams (Gradient Descent with Adaptive Momentum Scaling)**
```python
# 自適應動量縮放梯度下降
grams_update = update_p.abs() * grad.sign()
alpha = 1.0 * group['beta1_decay'] ** state["step"]
update_p = alpha * grams_update + (1 - alpha) * update_p
```
- **論文**: [Grams: Gradient Descent with Adaptive Momentum Scaling](https://arxiv.org/abs/2412.17107)
- **功能**: 動態調整動量縮放

### 6. **Orthogonal Gradient**
```python
# 正交梯度投影（早期暖身）
proj = torch.dot(w, g) / torch.dot(w, w).add(1e-30)
g_orth = g.sub_(w, alpha=proj)
```
- **論文**: [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697)
- **功能**: 提高數值穩定性，避免梯度爆炸

### 7. **AdaBelief Variance Estimation**
```python
# AdaBelief 變異數估計
res = (scaled_grad - exp_avg_bar).pow(2) + eps2
exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt())
```
- **論文**: [AdaBelief Optimizer](https://arxiv.org/abs/2010.07468)
- **功能**: 基於梯度預測的自適應學習率

### 8. **Automagic Learning Rate Mask**
```python
# 自動學習率遮罩
sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
new_lr = torch.where(sign_agree > 0, lr_mask + lr_bump, lr_mask - lr_bump)
```
- **來源**: [Automagic Optimizer](https://github.com/ostris/ai-toolkit)
- **功能**: 參數級別的學習率自適應調整

### 9. **ALLoRA (Adaptive Learning Rate Mitigates LoRA Fatal Flaws)**
```python
# 行縮放（適用於 LoRA）
row_norm = p.norm(dim=1, keepdim=True)
state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))
```
- **論文**: [ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws](https://arxiv.org/abs/2410.09692)
- **功能**: 針對 LoRA 微調的特殊優化

### 10. **Adaptive Weight Decay**
```python
# 自適應權重衰減
norm_grad = (param_abs_grad - mean_norm) / std_norm
theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
p.data.mul_(1 - new_lr * group["weight_decay"] * theta)
```
- **論文**: [Adaptive Weight Decay for Deep Neural Networks](https://arxiv.org/abs/1907.08931)
- **功能**: 根據梯度模式動態調整權重衰減

## ⚙️ 配置參數

### OptimizerConfig 類別

```python
@dataclass
class OptimizerConfig:
    lr: float = 1e-6                    # 基礎學習率
    min_lr: float = 1e-7                # 最小學習率
    max_lr: float = 1e-3                # 最大學習率
    lr_bump: float = 3e-6               # 學習率調整幅度
    eps: Tuple[float, float, float] = (1e-30, 1e-16, 1e-8)  # 數值穩定性參數
    clip_threshold: float = 1.0         # 梯度裁剪閾值
    betas: Tuple[float, float, float] = (0.8, 0.99, 0.999)  # 動量參數
    eta: float = 2.0                    # ALLoRA 參數
    beta1_decay: float = 0.9995         # Beta1 衰減率
    weight_decay: float = 5e-4          # 權重衰減
    warmup_steps: int = 500             # 暖身步數
    came: bool = True                   # 是否使用 CAME
    full_finetune: bool = False         # 是否全量微調
    verbose: bool = False               # 詳細輸出
```

### 參數詳解

#### 基礎參數
- **`lr`**: 基礎學習率，建議範圍 1e-6 到 1e-3
- **`min_lr` / `max_lr`**: 學習率的動態調整邊界
- **`lr_bump`**: 學習率遮罩的調整幅度

#### 穩定性參數
- **`eps`**: 三個不同場景的 epsilon 值
  - `eps[0]`: CAME 和數值穩定性 (1e-30)
  - `eps[1]`: AdaBelief 變異數 (1e-16)
  - `eps[2]`: 其他計算 (1e-8)
- **`clip_threshold`**: RMS 正規化的裁剪閾值

#### 動量參數
- **`betas`**: 三階動量參數
  - `beta1`: 一階動量衰減 (0.8)
  - `beta2`: 二階動量衰減 (0.99)
  - `beta3`: AdaBelief 衰減 (0.999)
- **`beta1_decay`**: Beta1 的時間衰減率

#### 訓練控制
- **`warmup_steps`**: 暖身階段的步數
- **`weight_decay`**: L2 正則化強度
- **`came`**: 是否使用 CAME 算法（vs AdaBelief）

## 🔧 使用方法

### 1. 基礎版本 - Automagic_CameAMP

```python
from library.automagic_cameamp import Automagic_CameAMP

# 創建優化器
optimizer = Automagic_CameAMP(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    warmup_steps=1000,
    verbose=True
)

# 訓練循環
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2. 8-bit 量化版本 - Automagic_CameAMP8bit

```python
from library.automagic_cameamp import Automagic_CameAMP8bit

# 需要 bitsandbytes 支援
try:
    optimizer = Automagic_CameAMP8bit(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        verbose=True
    )
except RuntimeError as e:
    print(f"8-bit 不可用: {e}")
    # 回退到 32-bit 版本
```

### 3. 上下文感知版本 - Automagic_CameAMP_COptim

```python
from library.automagic_cameamp import Automagic_CameAMP_COptim

optimizer = Automagic_CameAMP_COptim(
    model.parameters(),
    lr=1e-3,
    context_window=100,          # 上下文窗口大小
    edge_threshold=0.95,         # 邊緣情況檢測閾值
    adaptation_rate=0.1,         # 適應速率
    momentum_scales=[1, 5, 20, 100]  # 多尺度動量
)

# 監控上下文狀態
for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 檢查上下文狀態
    if batch_idx % 100 == 0:
        lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()
        is_edge = optimizer.c_optim.detect_edge_case()
        print(f"LR 乘數: {lr_mult:.3f}, 邊緣情況: {is_edge}")
```

### 4. 全功能版本 - Automagic_CameAMP_COptim8bit

```python
from library.automagic_cameamp import Automagic_CameAMP_COptim8bit

optimizer = Automagic_CameAMP_COptim8bit(
    model.parameters(),
    lr=1e-3,
    context_window=50,           # 8-bit 版本建議較小的窗口
    edge_threshold=0.8,
    adaptation_rate=0.2,
    verbose=True
)
```

## 📊 性能比較

### 記憶體使用

| 優化器版本 | 記憶體使用率 | 參數量支援 | 量化開銷 |
|------------|-------------|------------|----------|
| Automagic_CameAMP | 100% | 標準 | 無 |
| Automagic_CameAMP8bit | ~25% | 大模型 | 輕微 |
| Automagic_CameAMP_COptim | 105% | 標準 | 無 |
| Automagic_CameAMP_COptim8bit | ~30% | 大模型 | 輕微 |

### 特性對比

| 特性 | 基礎版 | 8-bit版 | C-Optim版 | 全功能版 |
|------|--------|---------|-----------|-----------|
| CAME 算法 | ✅ | ✅ | ✅ | ✅ |
| AGR 正則化 | ✅ | ✅ | ✅ | ✅ |
| 動量切換 | ✅ | ✅ | ✅ | ✅ |
| 學習率遮罩 | ✅ | ✅ | ✅ | ✅ |
| 8-bit 量化 | ❌ | ✅ | ❌ | ✅ |
| 上下文感知 | ❌ | ❌ | ✅ | ✅ |
| 多尺度動量 | ❌ | ❌ | ✅ | ✅ |
| 邊緣檢測 | ❌ | ❌ | ✅ | ✅ |

## 🎯 適用場景

### 1. 基礎版 (Automagic_CameAMP)
- **適用**: 一般深度學習訓練
- **優勢**: 穩定可靠，功能全面
- **場景**:
  - 中小型模型訓練
  - 穩定性優先的場景
  - 初次使用建議版本

### 2. 8-bit 版 (Automagic_CameAMP8bit)
- **適用**: 記憶體受限的大模型訓練
- **優勢**: 大幅節省記憶體（75%）
- **場景**:
  - 大語言模型微調
  - GPU 記憶體不足
  - 成本敏感的訓練

### 3. 上下文感知版 (Automagic_CameAMP_COptim)
- **適用**: 需要智能調整的高級訓練
- **優勢**: 自適應學習率，停滯檢測
- **場景**:
  - 複雜的訓練任務
  - 需要最佳性能
  - 研究和實驗

### 4. 全功能版 (Automagic_CameAMP_COptim8bit)
- **適用**: 大模型的高級訓練
- **優勢**: 結合記憶體效率和智能調整
- **場景**:
  - 大模型的高級微調
  - 資源受限但需要最佳性能
  - 生產環境推薦

## ⚡ 訓練階段特性

### 暖身階段 (0 ~ warmup_steps/2)
- **Torque-Aware Momentum**: 扭矩感知動量調整
- **Orthogonal Gradient**: 正交梯度投影
- **Adaptive Weight Decay**: 自適應權重衰減
- **學習率遮罩建立**: 參數級學習率優化

### 中期階段 (warmup_steps/2 ~ warmup_steps)
- **Consistency Momentum**: 切換到一致性動量
- **學習率遮罩穩定**: 繼續優化學習率分布

### 穩定階段 (warmup_steps+)
- **全功能運行**: 所有優化技術協同工作
- **動態調整**: 基於訓練狀態的實時優化

## 🔍 狀態管理

### 狀態字典結構

```python
state = {
    'step': int,                    # 當前步數
    'lr_mask': Tensor,              # 學習率遮罩
    'avg_lr': float,                # 平均學習率
    'exp_avg': Tensor,              # 一階動量
    'exp_avg_sq': Tensor,           # 二階動量 (CAME)
    'exp_avg_res': Tensor,          # AdaBelief 殘差
    's': Tensor,                    # TAM 狀態 (暖身期)
    'last_polarity': Tensor,        # 梯度極性歷史
    'lr_max': float,                # 最大學習率記錄
    'row_scaling': Tensor,          # ALLoRA 行縮放 (可選)
}
```

### 8-bit 版本額外狀態

```python
# 每個張量對應的量化狀態
state_8bit = {
    'lr_mask_q': Tensor,            # 量化的學習率遮罩
    'lr_mask_q_scale': Tensor,      # 量化比例因子
    'exp_avg_q': Tensor,            # 量化的動量
    'exp_avg_q_scale': Tensor,      # 動量比例因子
    # ... 其他張量的量化版本
}
```

### C-Optim 版本額外狀態

```python
c_optim_state = {
    'c_optim_context': Dict,        # 上下文信息
    'edge_case_count': int,         # 邊緣情況計數
    'contextual_lr_multiplier': float,  # 上下文學習率乘數
    'momentum_scale_*': Tensor,     # 多尺度動量
    'momentum_count_*': int,        # 動量計數器
}
```

## 🛠️ 狀態保存與載入

```python
# 保存優化器狀態
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 載入優化器狀態
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 檢查版本相容性
if 'magic_version' in checkpoint['optimizer_state_dict']:
    version = checkpoint['optimizer_state_dict']['magic_version']
    print(f"載入的優化器版本: {version}")
```

## 💡 最佳實踐

### 1. 參數調優建議

#### 學習率設定
```python
# 小模型 (< 100M 參數)
lr = 1e-3, warmup_steps = 500

# 中等模型 (100M - 1B 參數)
lr = 5e-4, warmup_steps = 1000

# 大模型 (> 1B 參數)
lr = 1e-4, warmup_steps = 2000
```

#### 記憶體優化
```python
# 記憶體充足
optimizer = Automagic_CameAMP_COptim(...)

# 記憶體受限
optimizer = Automagic_CameAMP_COptim8bit(
    ...,
    context_window=30,  # 減少上下文窗口
)
```

### 2. 監控建議

```python
# 基礎監控
if step % 100 == 0:
    avg_lr = optimizer._get_group_lr(optimizer.param_groups[0])
    print(f"平均學習率: {avg_lr:.6f}")

# 高級監控 (C-Optim 版本)
if hasattr(optimizer, 'c_optim'):
    lr_mult = optimizer.c_optim.compute_contextual_lr_multiplier()
    is_edge = optimizer.c_optim.detect_edge_case()
    grad_consistency = optimizer.c_optim.compute_gradient_consistency()

    print(f"上下文乘數: {lr_mult:.3f}")
    print(f"邊緣情況: {is_edge}")
    print(f"梯度一致性: {grad_consistency:.3f}")
```

### 3. 故障排除

#### 常見問題
1. **8-bit 初始化失敗**
   ```python
   # 檢查 bitsandbytes 安裝
   try:
       import bitsandbytes
       print(f"bitsandbytes 版本: {bitsandbytes.__version__}")
   except ImportError:
       print("請安裝: pip install bitsandbytes")
   ```

2. **學習率過小/過大**
   ```python
   # 檢查學習率遮罩分布
   for group in optimizer.param_groups:
       for p in group['params']:
           if p.grad is not None:
               state = optimizer.state[p]
               if 'lr_mask' in state:
                   lr_mask = state['lr_mask']
                   print(f"LR 範圍: {lr_mask.min():.6f} - {lr_mask.max():.6f}")
   ```

3. **上下文感知效果不佳**
   ```python
   # 調整 C-Optim 參數
   optimizer = Automagic_CameAMP_COptim(
       ...,
       edge_threshold=0.8,    # 降低閾值，更積極調整
       adaptation_rate=0.2,   # 提高適應速率
       context_window=50,     # 減少窗口，更靈敏
   )
   ```

## 📚 參考文獻

1. [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/pdf/2411.02853)
2. [Adaptive Gradient Regularization](https://arxiv.org/pdf/2407.16944)
3. [Torque-Aware Momentum](https://arxiv.org/abs/2412.18790)
4. [Consistency Phenomenon in Diffusion Models](https://arxiv.org/abs/2404.07946)
5. [Grams: Gradient Descent with Adaptive Momentum Scaling](https://arxiv.org/abs/2412.17107)
6. [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697)
7. [AdaBelief Optimizer](https://arxiv.org/abs/2010.07468)
8. [ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws](https://arxiv.org/abs/2410.09692)
9. [Adaptive Weight Decay for Deep Neural Networks](https://arxiv.org/abs/1907.08931)
10. [Automagic Optimizer Implementation](https://github.com/ostris/ai-toolkit)

## 📄 授權

本實現基於相關論文和開源項目，遵循對應的授權條款。使用時請確保符合相關授權要求。