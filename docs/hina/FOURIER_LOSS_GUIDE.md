# Fourier Loss 傅立葉損失功能指南

## 概述 Overview

Fourier Loss 是一種基於頻域分析的先進損失函數，專門設計用於改善深度學習模型在頻率特徵學習方面的能力。該功能特別適用於圖像生成、超分辨率和細節重建等任務。

**🎉 新功能：整合型傅立葉損失 (Unified Fourier Loss)**
最新版本引入了整合型損失計算，結合了多尺度、頻率加權和自適應三種策略，提供更強大和靈活的損失計算能力。

Fourier Loss is an advanced loss function based on frequency domain analysis, specifically designed to improve deep learning models' ability to learn frequency features. This functionality is particularly suitable for image generation, super-resolution, and detail reconstruction tasks.

## 主要特性 Key Features

### 🎯 核心優勢 Core Advantages
- **頻域特徵學習**: 直接在頻率域中優化模型，增強對細節和紋理的學習能力
- **多模式支持**: 提供基礎、加權、多尺度、自適應和**整合**五種不同的損失計算模式
- **數值穩定性**: 內建正規化和約束機制，確保訓練過程的穩定性
- **靈活配置**: 豐富的參數選項，支持不同應用場景的需求
- **智能組合**: 新的整合模式自動結合多種策略的優勢

### 🛡️ 穩定性保證 Stability Guarantees
- **自動正規化**: FFT 幅度譜自動正規化，防止數值爆炸
- **權重約束**: 智能權重限制，避免極端值影響訓練
- **動態調整**: 自適應權重機制，保持與基礎損失的平衡
- **異常處理**: 完善的錯誤處理和回退機制

## 損失模式詳解 Loss Modes Explained

### 1. 基礎模式 (Basic Mode)
```python
fourier_mode = "basic"
```
**用途**: 直接計算頻域特徵差異，適合初學者和基礎應用
**特點**:
- 計算簡單，資源消耗較低
- 對所有頻率成分給予相等權重
- 適合快速實驗和概念驗證

### 2. 加權模式 (Weighted Mode)
```python
fourier_mode = "weighted"
fourier_high_freq_weight = 2.0  # 高頻權重倍數
```
**用途**: 對高頻成分給予更高權重，增強細節學習
**特點**:
- 智能頻率權重分配
- 使用 sigmoid 函數平滑過渡
- 權重自動約束在 [1.0, 3.0] 範圍內
- 平衡計算效率與效果

**權重分配原理**:
```
低頻成分權重 = 1.0
高頻成分權重 = 1.0 + (high_freq_weight - 1.0) * sigmoid_factor
```

### 3. 多尺度模式 (Multiscale Mode)
```python
fourier_mode = "multiscale"
fourier_scales = [1, 2, 4]  # 多個尺度
fourier_scale_weights = [0.5, 0.35, 0.15]  # 可選：自定義尺度權重
```
**用途**: 在多個解析度上計算損失，捕捉不同層次的特徵
**特點**:
- 自動尺度權重計算：`weight = 1.0 / sqrt(scale)`
- 支持自定義尺度權重
- 智能尺度篩選，跳過無效尺度
- 適合需要多層次特徵的任務

### 4. 自適應模式 (Adaptive Mode)
```python
fourier_mode = "adaptive"
fourier_adaptive_max_weight = 2.5  # 最大權重
fourier_adaptive_min_weight = 0.8  # 最小權重
```
**用途**: 根據訓練進度動態調整權重，早期重視高頻，後期平衡
**特點**:
- 線性權重衰減：`weight = max_weight - (max_weight - min_weight) * progress`
- 訓練進度自動計算
- 適合長期訓練和微調

### 5. 整合模式 (Unified Mode) ⭐ **新功能**
```python
fourier_mode = "unified"
# 或使用簡化預設
fourier_mode = "unified_balanced"  # 推薦
fourier_mode = "unified_detail"    # 細節增強
fourier_mode = "unified_adaptive"  # 自適應策略
```

**用途**: 結合多尺度、頻率加權和自適應三種策略的統一實現
**特點**:
- **多層次整合**: 每個尺度都可以應用頻率加權
- **三種自適應曲線**: linear、cosine、exponential
- **動態組合權重**: 根據訓練進度調整多尺度和加權的比例
- **模組化設計**: 可選擇性啟用/禁用各個組件
- **預設配置**: 提供多種預設模式，開箱即用

#### 整合模式架構
```
整合損失 = 多尺度權重 × 多尺度損失 + 加權權重 × 單尺度加權損失

其中：
- 多尺度損失 = Σ(尺度權重 × 該尺度的頻率加權損失)
- 權重會根據訓練進度自適應調整
- 支持三種自適應曲線模式
```

#### 預設配置說明

**unified_basic**: 基礎整合模式
- 禁用多尺度，主要使用單尺度加權
- 線性自適應調整
- 適合快速測試和資源受限環境

**unified_balanced**: 平衡整合模式 ⭐ **推薦**
- 啟用雙尺度 [1, 2]
- 平衡的多尺度和加權比例 (0.6:0.4)
- 線性自適應調整
- 適合大多數應用場景

**unified_detail**: 細節增強模式
- 啟用三尺度 [1, 2, 4]
- 更高的頻率權重和多尺度比例
- 餘弦自適應調整，更平滑的過渡
- 適合需要高品質細節的任務

**unified_adaptive**: 自適應策略模式
- 啟用雙尺度配置
- 指數自適應調整，早期快速衰減
- 動態組合權重調整
- 適合長期訓練和復雜場景

## 使用方法 Usage Methods

### 方法1: 快速配置 (推薦)
```python
from library.train_util import apply_fourier_loss_to_args

# 使用預設配置
apply_fourier_loss_to_args(args, mode="unified_balanced")
```

### 方法2: 命令行參數
```bash
# 基礎使用
python train_network.py \
  --loss_type fourier \
  --fourier_mode unified_balanced \
  --fourier_weight 0.06

# 自定義整合模式
python train_network.py \
  --loss_type fourier \
  --fourier_mode unified \
  --fourier_weight 0.05 \
  --fourier_scales 1,2,4 \
  --fourier_high_freq_weight 2.0 \
  --fourier_adaptive_max_weight 2.5 \
  --fourier_adaptive_min_weight 0.8
```

### 方法3: 代碼中直接調用
```python
from library.fourier_loss import (
    fourier_latent_loss_unified,
    fourier_latent_loss_unified_simple
)

# 簡化版調用
loss = fourier_latent_loss_unified_simple(
    model_pred, target,
    mode="balanced",  # 或 "detail", "adaptive"
    current_step=step,
    total_steps=total_steps
)

# 完全自定義
loss = fourier_latent_loss_unified(
    model_pred, target,
    enable_multiscale=True,
    scales=[1, 2, 4],
    enable_frequency_weighting=True,
    enable_adaptive=True,
    adaptive_mode="cosine",
    max_weight=2.5,
    min_weight=0.8,
    current_step=step,
    total_steps=total_steps
)
```

## 參數詳解 Parameter Details

### 基本參數 Basic Parameters
- `fourier_weight`: 傅立葉損失權重 (0.01-0.12)
- `fourier_mode`: 損失模式選擇
- `fourier_norm`: 範數類型 ("l1" 或 "l2")
- `fourier_warmup_steps`: 預熱步數

### 加權模式參數 Weighted Mode Parameters
- `fourier_high_freq_weight`: 高頻權重倍數 (1.0-3.0)

### 多尺度模式參數 Multiscale Mode Parameters
- `fourier_scales`: 尺度列表，如 [1, 2, 4]
- `fourier_scale_weights`: 自定義尺度權重

### 自適應模式參數 Adaptive Mode Parameters
- `fourier_adaptive_max_weight`: 最大權重
- `fourier_adaptive_min_weight`: 最小權重

### 整合模式專用參數 Unified Mode Parameters
- `adaptive_mode`: 自適應曲線 ("linear", "cosine", "exponential")
- `multiscale_weight`: 多尺度分量權重 (預設 0.6)
- `weighted_weight`: 加權分量權重 (預設 0.4)
- `adaptive_scaling`: 是否啟用動態組合權重調整

## 應用場景配置 Application Configurations

### 🎨 圖像生成 Image Generation
```python
# 保守配置：注重穩定性
apply_fourier_loss_to_args(args, mode="unified_basic")

# 平衡配置：質量與效率並重 (推薦)
apply_fourier_loss_to_args(args, mode="unified_balanced")
```

### 🔍 超分辨率 Super Resolution
```python
# 細節增強：最高品質
apply_fourier_loss_to_args(args, mode="unified_detail")

# 或自定義多尺度配置
fourier_mode = "unified"
fourier_scales = [1, 2, 4, 8]
fourier_weight = 0.08
```

### 🎭 風格轉換 Style Transfer
```python
# 自適應策略：適應復雜變化
apply_fourier_loss_to_args(args, mode="unified_adaptive")

# 或自定義指數衰減
fourier_mode = "unified"
adaptive_mode = "exponential"
max_weight = 3.0
min_weight = 0.5
```

### 🖼️ 圖像修復 Image Restoration
```python
# 細節保留配置
fourier_mode = "unified_detail"
fourier_weight = 0.10
fourier_high_freq_weight = 2.5
```

## 性能調優指南 Performance Tuning Guide

### 🎯 權重調整策略
```
超保守 Ultra Conservative: 0.005 - 0.01
保守 Conservative:        0.01 - 0.03
平衡 Balanced:            0.03 - 0.06  ⭐ 推薦
積極 Aggressive:          0.06 - 0.10
超積極 Ultra Aggressive:   0.10 - 0.15
```

### 📊 模式選擇指南
| 場景 | 推薦模式 | 權重範圍 | 特點 |
|------|----------|----------|------|
| 快速原型 | unified_basic | 0.02-0.04 | 資源友好 |
| 日常訓練 | unified_balanced | 0.04-0.06 | 平衡效果 |
| 高品質生成 | unified_detail | 0.06-0.08 | 細節豐富 |
| 長期微調 | unified_adaptive | 0.05-0.07 | 策略靈活 |
| 自定義需求 | unified | 0.03-0.10 | 完全控制 |

### ⚡ 性能優化建議
1. **記憶體優化**: 使用較小的尺度列表，如 [1, 2] 而非 [1, 2, 4, 8]
2. **計算優化**: 選擇 "l2" 範數通常比 "l1" 更快
3. **預熱策略**: 設置適當的預熱步數，避免早期不穩定
4. **動態調整**: 使用自適應模式可以自動優化訓練過程

## 故障排除 Troubleshooting

### 常見問題及解決方案

**問題1: 損失過大 (> 10.0)**
```python
# 解決方案：降低權重
fourier_weight = 0.01
fourier_warmup_steps = 500
```

**問題2: 損失過小 (< 0.001)**
```python
# 解決方案：增加權重或使用更激進模式
fourier_weight = 0.08
fourier_mode = "unified_detail"
```

**問題3: 記憶體不足**
```python
# 解決方案：使用基礎模式或減少尺度
fourier_mode = "unified_basic"
# 或
fourier_scales = [1, 2]  # 減少尺度數量
```

**問題4: 訓練不穩定**
```python
# 解決方案：增加預熱期，使用平滑的自適應曲線
fourier_warmup_steps = 500
adaptive_mode = "cosine"  # 更平滑的過渡
```

**問題5: 效果不明顯**
```python
# 解決方案：確保在正確的訓練階段，調整組合權重
fourier_mode = "unified_detail"
fourier_weight = 0.08
multiscale_weight = 0.7  # 增強多尺度影響
```

### 調試技巧 Debug Tips

1. **監控損失比例**: 傅立葉損失應該是基礎損失的 5%-20%
2. **觀察收斂趨勢**: 正常情況下應該呈現平穩下降趨勢
3. **檢查梯度**: 使用測試腳本檢查梯度是否正常
4. **分階段測試**: 先測試基礎模式，再逐步增加復雜度

## 最佳實踐 Best Practices

### ✅ 推薦做法
1. **從預設開始**: 先使用 `unified_balanced` 模式
2. **逐步調整**: 根據結果逐步微調參數
3. **監控指標**: 密切關注損失比例和訓練穩定性
4. **記錄實驗**: 記錄不同配置的效果以便比較
5. **定期驗證**: 在驗證集上檢查效果

### ❌ 避免的做法
1. **過高權重**: 避免 fourier_weight > 0.15
2. **跳過預熱**: 總是設置適當的預熱期
3. **忽略監控**: 不監控損失比例的變化
4. **盲目複雜**: 不要一開始就使用最復雜的配置

## 技術原理 Technical Principles

### 傅立葉變換與頻域分析
```
F(u,v) = ∑∑ f(x,y) * e^(-j2π(ux/M + vy/N))
|F(u,v)| = sqrt(Re(F(u,v))² + Im(F(u,v))²)
```

### 整合架構數學表達
```
L_unified = α(t) * L_multiscale + β(t) * L_weighted

其中：
- α(t), β(t) 是時間依賴的權重函數
- L_multiscale = ∑ w_s * L_weighted(downsample(x, s))
- L_weighted = ∑ W(u,v) * |F_pred(u,v) - F_target(u,v)|^p
```

### 自適應函數
```python
# 線性 Linear
adaptive_factor = max_weight - (max_weight - min_weight) * progress

# 餘弦 Cosine
adaptive_factor = min_weight + (max_weight - min_weight) * 0.5 * (1 + cos(π * progress))

# 指數 Exponential
adaptive_factor = min_weight + (max_weight - min_weight) * exp(-5 * progress)
```

## 版本歷史 Version History

### v2.0 (最新) - 整合型損失
- ✨ 新增整合型傅立葉損失
- 🔧 三種自適應曲線支持
- 📦 預設配置模式
- ⚡ 性能和穩定性改進

### v1.0 - 基礎功能
- 🎯 四種基礎損失模式
- 🛡️ 數值穩定性機制
- 📚 完整文檔和範例

## 參考資料 References

- [快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md)
- [使用範例](./fourier_loss_examples.py)
- [測試腳本](./test_unified_fourier_loss.py)