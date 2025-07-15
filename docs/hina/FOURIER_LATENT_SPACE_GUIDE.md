# SD-Scripts 環境最佳實踐指南

## 🎯 概述

本指南提供在 SD-Scripts 環境下使用 HinaAdaptive 優化器的最佳實踐，確保您能夠充分利用適用於 latent space 訓練的正則化技術。

## ⚠️ 重要更新

**傅立葉特徵損失功能已完全移除**，因為它不適用於 SD-Scripts 的 latent space 訓練環境。本指南已更新為展示推薦的替代方案。

## 🧠 SD-Scripts 架構理解

### 訓練流程

SD-Scripts 中的所有訓練都遵循以下流程：
```
images → VAE encode → latents → diffusion training
```

這意味著：
- **100% 的訓練發生在 latent space 中**
- 這不是配置選項，而是框架的核心設計
- 所有模型（SD 1.x/2.x、SDXL、SD3、FLUX）都使用這種方式

### 為什麼這很重要？

1. **頻率概念不適用**：在 latent space 中，"頻率" 失去了圖像空間中的物理意義
2. **正則化技術需要適配**：必須使用專門適用於 latent space 的正則化方法
3. **效果最佳化**：使用正確的技術能獲得更好的訓練結果

## 🛠️ 推薦配置

### 基本配置

```python
from library.hina_adaptive import HinaAdaptive

# 適用於 SD-Scripts 的基本配置
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # 記憶體優化
    memory_efficient=True,
    vram_budget_gb=8.0,
    # 適用於 latent space 的正則化技術
    edge_suppression=True,
    edge_penalty=0.1,
    spatial_awareness=True,
    frequency_penalty=0.05,
)
```

### 高級配置

```python
# 適用於大型模型的高級配置
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # 記憶體優化
    memory_efficient=True,
    vram_budget_gb=12.0,
    reduce_precision=True,
    cpu_offload_states=True,
    # 正則化技術組合
    edge_suppression=True,
    edge_penalty=0.1,
    spatial_awareness=True,
    frequency_penalty=0.05,
    background_regularization=True,
    lora_rank_penalty=True,
    rank_penalty_strength=0.01,
    # 動態自適應
    use_dynamic_adaptation=True,
    # 高級功能
    use_tam=True,
    use_cautious=True,
    use_spd=True,
)
```

## 📊 正則化技術詳解

### 1. 邊緣感知正則化

**適用原因**：檢測梯度中的尖銳變化，不依賴於圖像空間的頻率特徵

```python
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    edge_suppression=True,
    edge_penalty=0.1,       # 邊緣懲罰強度
    edge_threshold=0.6,     # 邊緣檢測閾值
)
```

**效果**：
- 防止過度擬合邊緣特徵
- 改善訓練穩定性
- 減少偽影產生

### 2. 空間感知正則化

**適用原因**：基於局部變異數，適用於任何空間結構的數據

```python
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    spatial_awareness=True,
    frequency_penalty=0.05,    # 頻率懲罰
    detail_preservation=0.8,   # 細節保持
)
```

**效果**：
- 保持空間一致性
- 提升細節質量
- 減少噪點

### 3. 背景正則化

**適用原因**：基於活動度檢測，適用於任何特徵空間

```python
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    background_regularization=True,
)
```

**效果**：
- 改善背景一致性
- 減少背景噪點
- 提升整體質量

### 4. LoRA 低秩正則化

**適用原因**：直接作用於權重矩陣的秩，與數據空間無關

```python
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    lora_rank_penalty=True,
    rank_penalty_strength=0.01,  # 秩懲罰強度
    low_rank_emphasis=1.2,       # 低秩強調
)
```

**效果**：
- 提升 LoRA 訓練效率
- 減少過擬合
- 改善泛化性能

## 🚀 不同任務的最佳配置

### LoRA 訓練

```python
# 適用於 LoRA 訓練的配置
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # LoRA 特化
    lora_rank_penalty=True,
    rank_penalty_strength=0.01,
    low_rank_emphasis=1.2,
    # 基本正則化
    edge_suppression=True,
    edge_penalty=0.08,
    # 記憶體優化
    memory_efficient=True,
    vram_budget_gb=8.0,
)
```

### 大模型訓練

```python
# 適用於大型模型的配置
optimizer = HinaAdaptive(
    model.parameters(),
    lr=8e-5,
    # 組合正則化
    edge_suppression=True,
    edge_penalty=0.12,
    spatial_awareness=True,
    frequency_penalty=0.06,
    background_regularization=True,
    # 動態自適應
    use_dynamic_adaptation=True,
    # 記憶體優化
    memory_efficient=True,
    vram_budget_gb=16.0,
    reduce_precision=True,
    cpu_offload_states=True,
)
```

### 快速實驗

```python
# 適用於快速實驗的輕量配置
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # 核心正則化
    edge_suppression=True,
    edge_penalty=0.1,
    # 記憶體優化
    memory_efficient=True,
    vram_budget_gb=6.0,
)
```

## 🔧 記憶體優化策略

### 基本記憶體優化

```python
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # 基本記憶體優化
    memory_efficient=True,
    vram_budget_gb=8.0,
)
```

### 高級記憶體優化

```python
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # 高級記憶體優化
    memory_efficient=True,
    vram_budget_gb=12.0,
    reduce_precision=True,      # 使用 FP16
    cpu_offload_states=True,    # 狀態卸載到 CPU
    max_buffer_memory_mb=500,   # 限制緩衝區大小
)
```

### 極限記憶體優化

```python
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # 極限記憶體優化
    memory_efficient=True,
    vram_budget_gb=4.0,
    reduce_precision=True,
    cpu_offload_states=True,
    max_buffer_memory_mb=200,
    # 動態記憶體調整
    adaptive_features=True,
)
```

## 📈 性能監控

### 使用優化器信息

```python
# 獲取優化器狀態
info = optimizer.get_optimization_info()
print(f"啟用的正則化技術: {info['features']}")
print(f"記憶體配置: {info['memory_optimization']}")
```

### 記憶體監控

```python
# 監控記憶體使用
memory_stats = optimizer.get_memory_stats()
print(f"記憶體壓力: {memory_stats['memory_pressure']:.2%}")
print(f"緩衝池記憶體: {memory_stats['buffer_pool_stats']['current_memory_mb']:.2f}MB")
```

### 動態調整

```python
# 根據記憶體使用動態調整
if memory_stats['memory_pressure'] > 0.8:
    optimizer.optimize_for_vram(target_vram_gb=6.0)
```

## 🎯 最佳實踐建議

### 1. 逐步啟用功能

```python
# 階段 1: 基本配置
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    memory_efficient=True,
)

# 階段 2: 添加正則化
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    memory_efficient=True,
    edge_suppression=True,
    edge_penalty=0.1,
)

# 階段 3: 完整配置
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    memory_efficient=True,
    edge_suppression=True,
    edge_penalty=0.1,
    spatial_awareness=True,
    frequency_penalty=0.05,
    background_regularization=True,
)
```

### 2. 參數調優策略

1. **從保守參數開始**
2. **觀察訓練效果**
3. **逐步增加正則化強度**
4. **監控記憶體使用**

### 3. 故障排除

#### 記憶體不足

```python
# 減少記憶體使用
optimizer.optimize_for_vram(target_vram_gb=6.0)
```

#### 訓練不穩定

```python
# 降低正則化強度
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    edge_suppression=True,
    edge_penalty=0.05,  # 降低
    spatial_awareness=True,
    frequency_penalty=0.02,  # 降低
)
```

## 🔄 遷移指南

### 從舊配置遷移

如果您之前使用了 `fourier_feature_loss=True`，請按以下方式遷移：

```python
# 舊配置（已不可用）
# optimizer = HinaAdaptive(
#     model.parameters(),
#     lr=1e-4,
#     fourier_feature_loss=True,  # 已移除
#     super_resolution_mode=True,  # 已移除
# )

# 新配置（推薦）
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-4,
    # 使用適用於 latent space 的正則化技術
    edge_suppression=True,
    edge_penalty=0.1,
    spatial_awareness=True,
    frequency_penalty=0.05,
    background_regularization=True,
    memory_efficient=True,
    vram_budget_gb=8.0,
)
```

## 📚 參考資源

### 測試腳本

```bash
# 測試基本功能
python docs/hina/test_fourier_super_resolution.py

# 測試維度處理
python docs/hina/test_fourier_dimension_fix.py

# 測試動態自適應
python docs/hina/test_adaptive_frequency_weighting.py
```

### 示例腳本

```bash
# 運行完整示例
python docs/hina/fourier_super_resolution_example.py
```

## 🚀 未來發展

我們正在開發專門針對 latent space 的新功能：

1. **語義感知正則化**：基於語義理解的正則化技術
2. **跨模態特徵對齊**：改善不同模態之間的特徵對齊
3. **自適應 latent 空間優化**：根據 latent 空間特性自動調整

---

**版本**: 功能移除版本
**狀態**: 推薦使用替代方案
**適用環境**: SD-Scripts latent space 訓練

> 💡 **提示**: 本指南反映了 SD-Scripts 的實際架構。所有推薦的正則化技術都已在 latent space 環境中驗證過效果。