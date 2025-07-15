# Fourier Loss 傅立葉損失功能指南

## 概述 Overview

Fourier Loss 是一種基於頻域分析的先進損失函數，專門設計用於改善深度學習模型在頻率特徵學習方面的能力。該功能特別適用於圖像生成、超分辨率和細節重建等任務。

Fourier Loss is an advanced loss function based on frequency domain analysis, specifically designed to improve deep learning models' ability to learn frequency features. This functionality is particularly suitable for image generation, super-resolution, and detail reconstruction tasks.

## 主要特性 Key Features

### 🎯 核心優勢 Core Advantages
- **頻域特徵學習**: 直接在頻率域中優化模型，增強對細節和紋理的學習能力
- **多模式支持**: 提供基礎、加權、多尺度和自適應四種不同的損失計算模式
- **數值穩定性**: 內建正規化和約束機制，確保訓練過程的穩定性
- **靈活配置**: 豐富的參數選項，支持不同應用場景的需求

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

### 2. 加權模式 (Weighted Mode) ⭐ 推薦
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
fourier_scale_weights = [1.0, 0.5, 0.25]  # 對應權重
```
**用途**: 在多個解析度上計算損失，適合處理多尺度特徵
**特點**:
- 支援任意數量的尺度
- 自動維度安全檢查
- 靈活的尺度權重配置
- 適合複雜場景的細節重建

### 4. 自適應模式 (Adaptive Mode)
```python
fourier_mode = "adaptive"
fourier_adaptive_max_weight = 2.0  # 訓練初期最大權重
fourier_adaptive_min_weight = 0.5  # 訓練後期最小權重
```
**用途**: 根據訓練進度動態調整高頻權重
**特點**:
- 早期訓練重視高頻細節
- 後期訓練逐漸平衡
- 自動進度計算
- 適合長期訓練項目

**權重演變公式**:
```
progress = current_step / total_steps
high_freq_weight = max_weight - (max_weight - min_weight) * progress
```

## 參數配置指南 Parameter Configuration Guide

### 🔧 基本參數 Basic Parameters

| 參數名 | 類型 | 預設值 | 範圍 | 說明 |
|--------|------|--------|------|------|
| `fourier_weight` | float | 0.05 | [0.001, 0.2] | Fourier 損失在總損失中的權重 |
| `fourier_mode` | str | "weighted" | - | 損失計算模式 |
| `fourier_norm` | str | "l2" | ["l1", "l2"] | 損失範數類型 |
| `fourier_eps` | float | 1e-8 | [1e-10, 1e-6] | 數值穩定性常數 |
| `fourier_warmup_steps` | int | 200 | [0, 1000] | 預熱步數 |

### 🎛️ 進階參數 Advanced Parameters

#### 加權模式參數
| 參數名 | 預設值 | 範圍 | 說明 |
|--------|--------|------|------|
| `fourier_high_freq_weight` | 2.0 | [1.0, 3.0] | 高頻成分權重倍數 |

#### 多尺度模式參數
| 參數名 | 預設值 | 說明 |
|--------|--------|------|
| `fourier_scales` | [1, 2] | 尺度列表 |
| `fourier_scale_weights` | None | 尺度權重（None 時自動計算） |

#### 自適應模式參數
| 參數名 | 預設值 | 範圍 | 說明 |
|--------|--------|------|------|
| `fourier_adaptive_max_weight` | 2.0 | [1.0, 3.0] | 最大高頻權重 |
| `fourier_adaptive_min_weight` | 0.5 | [0.5, 2.0] | 最小高頻權重 |

## 快速開始 Quick Start

### 方法一：使用預設配置 Using Preset Configurations

```python
from library.train_util import apply_fourier_loss_to_args, get_fourier_loss_config

# 查看可用配置
configs = ["conservative", "balanced", "aggressive"]

# 應用平衡配置（推薦）
apply_fourier_loss_to_args(args, mode="balanced")
```

**預設配置說明**:
- **conservative**: 保守配置，適合初學者 (`fourier_weight=0.01`)
- **balanced**: 平衡配置，適合大多數場景 (`fourier_weight=0.05`) ⭐
- **aggressive**: 激進配置，適合高品質需求 (`fourier_weight=0.1`)

### 方法二：手動配置 Manual Configuration

```python
# 在訓練腳本中添加參數
args.loss_type = "fourier"
args.fourier_weight = 0.05
args.fourier_mode = "weighted"
args.fourier_high_freq_weight = 2.0
args.fourier_warmup_steps = 200
```

### 方法三：命令行使用 Command Line Usage

```bash
python train_network.py \
  --loss_type fourier \
  --fourier_weight 0.05 \
  --fourier_mode weighted \
  --fourier_high_freq_weight 2.0 \
  --fourier_warmup_steps 200 \
  [其他訓練參數...]
```

## 應用場景 Application Scenarios

### 🎨 圖像生成 Image Generation
**推薦配置**:
```python
fourier_mode = "weighted"
fourier_weight = 0.03
fourier_high_freq_weight = 1.5
```
**效果**: 增強生成圖像的細節質量和紋理豐富度

### 🔍 超分辨率 Super Resolution
**推薦配置**:
```python
fourier_mode = "multiscale"
fourier_weight = 0.08
fourier_scales = [1, 2, 4]
```
**效果**: 提升高分辨率重建的清晰度和邊緣質量

### 🎭 風格轉換 Style Transfer
**推薦配置**:
```python
fourier_mode = "adaptive"
fourier_weight = 0.05
fourier_adaptive_max_weight = 2.5
fourier_adaptive_min_weight = 0.8
```
**效果**: 保持風格轉換中的細節一致性

### 🖼️ 圖像修復 Image Restoration
**推薦配置**:
```python
fourier_mode = "weighted"
fourier_weight = 0.06
fourier_high_freq_weight = 2.0
```
**效果**: 增強修復圖像的邊緣和紋理恢復

## 性能優化建議 Performance Optimization

### 💡 訓練策略 Training Strategies

1. **漸進式權重調整**:
   ```python
   # 從小權重開始
   initial_weight = 0.01
   target_weight = 0.05
   # 在訓練過程中逐漸增加
   ```

2. **預熱期設置**:
   ```python
   # 設置適當的預熱步數
   fourier_warmup_steps = max(200, total_steps // 50)
   ```

3. **動態監控**:
   ```python
   # 監控損失比例
   if fourier_loss / base_loss > 10.0:
       # 自動降低權重
   ```

### ⚡ 計算效率 Computational Efficiency

- **基礎模式**: 最快，適合快速實驗
- **加權模式**: 中等，推薦日常使用
- **多尺度模式**: 較慢，適合高品質需求
- **自適應模式**: 中等，適合長期訓練

### 🎯 記憶體使用 Memory Usage

```python
# 大張量處理建議
if tensor_size > threshold:
    # 使用基礎模式降低記憶體消耗
    fourier_mode = "basic"
    fourier_weight *= 0.8  # 適度降低權重
```

## 故障排除 Troubleshooting

### ❌ 常見問題 Common Issues

#### 1. 損失值過大
**現象**: Fourier 損失值 > 10.0
**解決方案**:
```python
# 降低權重
fourier_weight = 0.01

# 使用保守配置
apply_fourier_loss_to_args(args, mode="conservative")
```

#### 2. 損失值過小
**現象**: Fourier 損失值 < 0.001
**解決方案**:
```python
# 檢查輸入張量範圍
debug_fourier_magnitude_spectrum(tensor)

# 增加權重
fourier_weight = 0.08
```

#### 3. 記憶體不足
**現象**: CUDA out of memory
**解決方案**:
```python
# 使用較小的尺度
fourier_scales = [1, 2]  # 移除大尺度

# 降低批次大小
batch_size = batch_size // 2
```

#### 4. 訓練不穩定
**現象**: 損失劇烈震盪
**解決方案**:
```python
# 增加預熱步數
fourier_warmup_steps = 500

# 使用較小權重
fourier_weight = 0.02
```

### 🔍 診斷工具 Diagnostic Tools

```python
# 測試損失值範圍
from library.train_util import test_fourier_loss_ranges
test_fourier_loss_ranges()

# 調試幅度譜計算
from library.train_util import debug_fourier_magnitude_spectrum
debug_fourier_magnitude_spectrum(your_tensor)
```

## 技術原理 Technical Principles

### 📐 傅立葉變換基礎 FFT Fundamentals

Fourier Loss 基於離散傅立葉變換（DFT），將空間域信號轉換到頻率域：

```
F(u,v) = Σ Σ f(x,y) * exp(-2πi(ux/M + vy/N))
```

其中：
- `f(x,y)` 是空間域信號（圖像或特徵圖）
- `F(u,v)` 是頻率域表示
- `M, N` 是圖像尺寸

### 🔢 幅度譜計算 Magnitude Spectrum Calculation

```python
# 1. 計算 FFT
fft_result = torch.fft.fftn(tensor, dim=(-2, -1))

# 2. 計算幅度譜
magnitude = torch.abs(fft_result)

# 3. 正規化處理
magnitude = magnitude / (tensor_numel ** 0.5)
magnitude = magnitude / (torch.std(tensor) + eps)
```

### 🎛️ 頻率權重機制 Frequency Weighting Mechanism

加權模式使用 sigmoid 函數創建平滑的頻率權重：

```python
# 計算頻率幅度
freq_magnitude = torch.sqrt(freq_h_grid**2 + freq_w_grid**2)
freq_magnitude = freq_magnitude / freq_magnitude.max()

# 應用 sigmoid 平滑化
sigmoid_factor = 4.0
freq_sigmoid = torch.sigmoid(sigmoid_factor * (freq_magnitude - 0.5))

# 創建權重遮罩
weight_mask = 1.0 + (high_freq_weight - 1.0) * freq_sigmoid
```

### 🔄 動態權重調整 Dynamic Weight Adjustment

組合損失時的自適應機制：

```python
# 計算損失比例
ratio = fourier_loss_magnitude / base_loss_magnitude

# 動態調整權重
if ratio > 10.0:
    adaptive_weight = fourier_weight / (ratio / 10.0)
    adaptive_weight = max(adaptive_weight, fourier_weight * 0.1)

# 最終損失
total_loss = base_loss + adaptive_weight * fourier_loss
```

## 最佳實踐 Best Practices

### ✅ 推薦做法 Recommended Practices

1. **從保守配置開始**:
   ```python
   apply_fourier_loss_to_args(args, mode="conservative")
   ```

2. **監控訓練過程**:
   ```python
   # 定期檢查損失比例
   fourier_ratio = fourier_loss / total_loss
   if fourier_ratio > 0.3:  # 如果 Fourier 損失佔比過高
       # 調整權重
   ```

3. **階段性調整**:
   ```python
   # 訓練初期使用較小權重
   if epoch < 10:
       current_fourier_weight = fourier_weight * 0.5
   else:
       current_fourier_weight = fourier_weight
   ```

### ❌ 避免事項 Things to Avoid

1. **不要使用過大的權重** (> 0.2)
2. **不要跳過預熱期** (warmup_steps = 0)
3. **不要忽略記憶體限制**
4. **不要在不穩定時繼續增加權重**

## 版本更新 Version Updates

### 🆕 最新改進 Latest Improvements

- ✅ **數值穩定性增強**: 新增 FFT 幅度譜正規化
- ✅ **權重約束優化**: 智能權重範圍限制
- ✅ **錯誤處理完善**: 完整的異常處理機制
- ✅ **性能優化**: 減少不必要的計算開銷
- ✅ **文檔完善**: 詳細的使用指南和故障排除

### 🔄 向後兼容性 Backward Compatibility

現有的 Fourier Loss 配置完全兼容新版本，無需修改現有代碼。

## 貢獻與反饋 Contribution & Feedback

### 🤝 如何貢獻 How to Contribute

1. 提交問題報告和功能建議
2. 分享使用案例和最佳實踐
3. 貢獻代碼改進和優化
4. 完善文檔和示例

### 📧 聯繫方式 Contact

如有任何問題或建議，請通過以下方式聯繫：
- GitHub Issues
- 技術討論區
- 郵件反饋

---

**注意**: 本文檔會持續更新，請定期查看最新版本以獲取最新功能和最佳實踐。

**Note**: This documentation is continuously updated. Please check regularly for the latest features and best practices.