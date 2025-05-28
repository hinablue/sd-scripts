# Automagic_CameAMP_Improved_8Bit 優化器配置檔案使用指南

## 📋 概述

現在您可以在 `optimizer_kwargs` 中使用 `profile` 參數來載入預定義的 `Automagic_CameAMP_Improved_8Bit` 優化器配置。這讓您能夠快速使用針對不同場景優化的設定，同時保留自定義特定參數的靈活性。

## 🎯 可用的配置檔案

### 1. `memory_optimized` - 記憶體優化
**適用場景**: 有限的 GPU 記憶體、大型模型訓練
**特點**: 最大化記憶體節省，犧牲部分功能來降低記憶體使用

```python
{
    'force_8bit': True,           # 強制所有狀態使用 8bit
    'min_8bit_size': 1024,        # 降低 8bit 閾值
    'edge_suppression': False,    # 關閉邊緣抑制
    'spatial_awareness': False,   # 關閉空間感知
    'background_regularization': False,  # 關閉背景正則化
    'frequency_penalty': 0.0,     # 關閉頻率懲罰
    'lora_rank_penalty': False,   # 關閉 LoRA 優化
    'verbose': False
}
```

### 2. `quality_optimized` - 品質優化
**適用場景**: 充足的 GPU 記憶體、追求最佳訓練效果
**特點**: 啟用所有優化功能，提供最佳的訓練品質

```python
{
    'edge_suppression': True,      # 啟用邊緣抑制
    'edge_penalty': 0.15,          # 較強的邊緣懲罰
    'background_regularization': True,  # 啟用背景正則化
    'spatial_awareness': True,     # 啟用空間感知
    'lora_rank_penalty': True,     # 啟用 LoRA 優化
    'frequency_penalty': 0.08,     # 較強的頻率懲罰
    'rank_penalty_strength': 0.02, # 較強的秩懲罰
    'verbose': True
}
```

### 3. `balanced` - 平衡配置
**適用場景**: 中等 GPU 記憶體、記憶體與品質的平衡
**特點**: 啟用主要優化功能，保持合理的記憶體使用

```python
{
    'min_8bit_size': 4096,        # 標準 8bit 閾值
    'edge_suppression': True,     # 啟用邊緣抑制
    'edge_penalty': 0.1,          # 標準邊緣懲罰
    'background_regularization': True,  # 啟用背景正則化
    'spatial_awareness': True,    # 啟用空間感知
    'lora_rank_penalty': True,    # 啟用 LoRA 優化
    'frequency_penalty': 0.05,    # 標準頻率懲罰
    'verbose': True
}
```

## 🚀 使用方法

### 方法 1: 在命令列中使用

#### 基本用法
```bash
# 使用記憶體優化配置
--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=memory_optimized

# 使用品質優化配置
--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=quality_optimized

# 使用平衡配置
--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=balanced
```

#### 進階用法 - 結合自定義參數
```bash
# 基於品質優化配置，但自定義學習率
--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=quality_optimized lr=2e-4

# 基於平衡配置，但調整邊緣懲罰和啟用詳細輸出
--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=balanced edge_penalty=0.15 verbose=True

# 基於記憶體優化配置，但自定義多個參數
--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args profile=memory_optimized lr=3e-4 warmup_steps=1000 weight_decay=1e-3
```

#### 完全自定義（不使用 profile）
```bash
# 不使用預設配置，完全自定義
--optimizer_type=Automagic_CameAMP_Improved_8Bit --optimizer_args lr=1e-4 edge_suppression=True min_8bit_size=2048 verbose=True
```

### 方法 2: 在 Python 程式碼中使用

#### 直接創建優化器
```python
from library.automagic_cameamp_improved_8bit import Automagic_CameAMP_Improved_8Bit

# 使用配置檔案
optimizer = Automagic_CameAMP_Improved_8Bit(
    model.parameters(),
    profile='quality_optimized',
    lr=2e-4,
    verbose=True
)
```

#### 使用便利函數
```python
from library.automagic_cameamp_improved_8bit import create_improved_8bit_optimizer

# 基本用法
optimizer = create_improved_8bit_optimizer(
    model.parameters(),
    profile='balanced'
)

# 結合自定義參數
optimizer = create_improved_8bit_optimizer(
    model.parameters(),
    profile='memory_optimized',
    lr=1e-4,
    warmup_steps=800
)
```

#### 模擬 train_util.py 的用法
```python
# 模擬 optimizer_kwargs 的構建
optimizer_kwargs = {
    'profile': 'quality_optimized',
    'lr': 2e-4,
    'verbose': True,
    'edge_penalty': 0.12
}

# 創建優化器（模擬 train_util.py 的邏輯）
lr = optimizer_kwargs.pop('lr', 1e-6)
optimizer = Automagic_CameAMP_Improved_8Bit(
    model.parameters(),
    lr=lr,
    **optimizer_kwargs
)
```

## 🔧 參數覆蓋機制

當您同時指定 `profile` 和其他參數時，覆蓋機制如下：

1. **首先載入** profile 指定的預定義配置
2. **然後應用** 您提供的自定義參數，覆蓋相應的預設值
3. **最終配置** = 預定義配置 + 您的自定義參數

### 範例說明

```bash
--optimizer_args profile=memory_optimized edge_suppression=True lr=2e-4
```

這個命令會：
1. 載入 `memory_optimized` 配置（其中 `edge_suppression=False`）
2. 應用您的自定義參數：
   - `edge_suppression=True` (覆蓋預設的 False)
   - `lr=2e-4` (設定學習率)
3. 最終結果：記憶體優化配置 + 啟用邊緣抑制 + 自定義學習率

## 📊 性能比較

| 配置檔案 | 記憶體使用 | 訓練品質 | 推薦場景 |
|----------|------------|----------|----------|
| `memory_optimized` | 🟢 最低 | 🟡 良好 | GPU 記憶體 < 8GB |
| `balanced` | 🟡 中等 | 🟢 優秀 | GPU 記憶體 8-16GB |
| `quality_optimized` | 🔴 較高 | 🟢 最佳 | GPU 記憶體 > 16GB |

## 🛠️ 實際使用範例

### LoRA 訓練
```bash
# 記憶體受限的 LoRA 訓練
python train_network.py \
    --optimizer_type=Automagic_CameAMP_Improved_8Bit \
    --optimizer_args profile=memory_optimized lr=1e-4 \
    --network_module=networks.lora \
    ...

# 高品質 LoRA 訓練
python train_network.py \
    --optimizer_type=Automagic_CameAMP_Improved_8Bit \
    --optimizer_args profile=quality_optimized lr=8e-5 \
    --network_module=networks.lora \
    ...
```

### DreamBooth 訓練
```bash
# 平衡配置的 DreamBooth 訓練
python train_db.py \
    --optimizer_type=Automagic_CameAMP_Improved_8Bit \
    --optimizer_args profile=balanced lr=5e-6 warmup_steps=100 \
    --pretrained_model_name_or_path=... \
    ...
```

## 🔍 配置驗證

您可以使用以下程式碼來檢查最終的配置：

```python
from library.automagic_cameamp_improved_8bit import Automagic_CameAMP_Improved_8Bit

optimizer = Automagic_CameAMP_Improved_8Bit(
    model.parameters(),
    profile='balanced',
    lr=2e-4,
    verbose=True,  # 這會顯示配置信息
    edge_penalty=0.12
)

# 檢查配置
config = optimizer.config
print(f"最終學習率: {config.lr}")
print(f"邊緣懲罰: {config.edge_penalty}")
print(f"邊緣抑制: {config.edge_suppression}")

# 獲取記憶體效率報告
report = optimizer.get_memory_efficiency_report()
print(f"8bit 參數比例: {report['compression_ratio']:.2%}")
```

## ❓ 常見問題

### Q1: profile 參數區分大小寫嗎？
A: 是的，profile 參數區分大小寫。請使用確切的名稱：`memory_optimized`、`quality_optimized`、`balanced`。

### Q2: 可以不使用 profile 嗎？
A: 可以！如果不指定 `profile` 參數，優化器會使用所有參數的預設值，您可以完全自定義配置。

### Q3: 如何查看所有可用的參數？
A: 查看 `Improved8BitOptimizerConfig` 類別的定義，或運行範例程式 `library/optimizer_profile_example.py`。

### Q4: profile 會影響學習率嗎？
A: 不會。`profile` 主要影響優化器的功能性參數（如邊緣抑制、8bit 設定等），學習率需要單獨指定。

### Q5: 可以組合多個 profile 嗎？
A: 不可以。一次只能指定一個 `profile`，但您可以在載入 profile 後覆蓋任何參數。

## 📚 更多資源

- 查看 `library/optimizer_profile_example.py` 了解詳細使用範例
- 閱讀 `library/automagic_cameamp_improved_8bit.py` 了解實現細節
- 參考 `library/BITSANDBYTES_8BIT_GUIDE.md` 了解 8bit 量化的技術原理

## 🎉 總結

透過 `profile` 參數，您現在可以：

1. **快速開始** - 使用預定義配置立即開始訓練
2. **靈活自定義** - 在預定義配置基礎上調整特定參數
3. **場景適配** - 根據 GPU 記憶體和品質需求選擇合適的配置
4. **簡化命令** - 減少命令列參數的複雜度

開始使用吧！選擇適合您場景的 profile，享受更高效的訓練體驗！ 🚀