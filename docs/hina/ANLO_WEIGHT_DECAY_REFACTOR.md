# ANLO 優化器權重衰減重構

## 概述

本次重構將 ANLO (Alternating Norm LoRA Optimizer) 的權重衰減實現從**梯度添加模式**改為**參數直接衰減模式**，使其與其他優化器（如 HinaAdamW）保持一致。

## 修改內容

### 修改前（梯度添加模式）
```python
# 權重衰減 - 添加到梯度中
if weight_decay != 0:
    for param in group['params']:
        if param.grad is not None:
            param.grad.add_(param, alpha=weight_decay)

# 參數更新
for param in group['params']:
    if param.grad is not None:
        param.add_(param.grad, alpha=-lr)
        param.grad.zero_()
```

### 修改後（參數直接衰減模式）
```python
# 參數更新
for param in group['params']:
    if param.grad is not None:
        param.add_(param.grad, alpha=-lr)
        param.grad.zero_()

# 權重衰減 - 直接應用到參數
if weight_decay != 0:
    for param in group['params']:
        param.data.add_(param.data, alpha=-lr * weight_decay)
```

## 技術細節

### 執行順序變更

**修改前順序：**
1. 權重衰減 → 添加到梯度
2. 歸一化 → 基於包含權重衰減的梯度
3. 參數更新 → 使用歸一化後的梯度

**修改後順序：**
1. 歸一化 → 基於原始梯度（不受權重衰減影響）
2. 參數更新 → 使用歸一化後的梯度
3. 權重衰減 → 直接應用到參數

### 數學等價性

兩種實現方式在數學上是等價的：

**梯度添加模式：**
```
param_new = param_old - lr * (grad + weight_decay * param_old)
         = param_old - lr * grad - lr * weight_decay * param_old
```

**參數直接衰減模式：**
```
param_new = param_old - lr * grad - lr * weight_decay * param_old
```

### 優勢

1. **歸一化一致性**：歸一化基於原始梯度范數，不受權重衰減影響
2. **實現一致性**：與其他優化器（HinaAdamW、AdaptiveHinaAdamW）保持一致
3. **代碼清晰性**：權重衰減作為獨立的步驟，邏輯更清晰
4. **調試便利性**：可以單獨觀察歸一化和權重衰減的效果

## 影響分析

### 對訓練的影響

1. **歸一化效果**：歸一化現在基於純梯度范數，可能提供更穩定的訓練
2. **權重衰減效果**：權重衰減效果保持不變
3. **數值穩定性**：可能提高數值穩定性，因為歸一化不受權重衰減干擾

### 向後兼容性

- 數學等價性確保訓練結果基本一致
- 用戶無需調整超參數
- API 保持不變

## 測試驗證

創建了測試腳本 `test_anlo_weight_decay.py` 來驗證：

1. 權重衰減正確應用
2. 參數值正確減小
3. 執行順序符合預期

## 相關文件

- `library/hina_anlo.py` - 主要修改文件
- `test_anlo_weight_decay.py` - 測試腳本
- `docs/hina/test_anlo_optimizer.py` - 現有測試文件

## 結論

本次重構提高了 ANLO 優化器的實現一致性，使其與項目中其他優化器保持相同的權重衰減模式。修改保持了數學等價性，同時提供了更清晰的代碼結構和更穩定的歸一化效果。