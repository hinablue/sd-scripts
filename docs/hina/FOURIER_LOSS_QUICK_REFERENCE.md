# Fourier Loss 快速參考 Quick Reference

## 🚀 一行配置 One-Line Setup

```python
# 🌟 新推薦：整合平衡配置 (New Recommended: Unified Balanced)
from library.train_util import apply_fourier_loss_to_args
apply_fourier_loss_to_args(args, mode="unified_balanced")
```

## 📋 參數速查表 Parameter Cheat Sheet

### 基本配置 Basic Configuration
```bash
--loss_type fourier
--fourier_weight 0.06
--fourier_mode unified_balanced
--fourier_warmup_steps 250
```

## 🎯 五種模式對比 Five Modes Comparison

### 經典模式 Classic Modes
| 模式 Mode | 用途 Usage | 權重 Weight | 性能 Performance |
|-----------|------------|-------------|------------------|
| `basic` | 快速測試 | 0.02 | ⚡⚡⚡ |
| `weighted` | 日常使用 | 0.05 | ⚡⚡ |
| `multiscale` | 高品質 | 0.08 | ⚡ |
| `adaptive` | 長期訓練 | 0.05 | ⚡⚡ |

### 🌟 整合模式 Unified Modes (新功能)
| 模式 Mode | 用途 Usage | 權重 Weight | 特點 Features |
|-----------|------------|-------------|---------------|
| `unified_basic` | 輕量整合 | 0.03 | 📱 資源友好 |
| `unified_balanced` ⭐ | 平衡整合 | 0.06 | 🎯 推薦使用 |
| `unified_detail` | 細節增強 | 0.08 | 🔍 高品質 |
| `unified_adaptive` | 智能自適應 | 0.07 | 🧠 動態調整 |

## 🎯 應用場景配置 Application Configs

### 圖像生成 Image Generation
```python
# 日常使用
fourier_mode = "unified_balanced"
fourier_weight = 0.05

# 高品質生成
fourier_mode = "unified_detail"
fourier_weight = 0.07
```

### 超分辨率 Super Resolution
```python
# 細節優先
fourier_mode = "unified_detail"
fourier_weight = 0.08

# 自定義多尺度
fourier_mode = "unified"
fourier_scales = [1, 2, 4, 8]
```

### 風格轉換 Style Transfer
```python
# 自適應策略
fourier_mode = "unified_adaptive"
fourier_weight = 0.06
adaptive_mode = "cosine"
```

### 圖像修復 Image Restoration
```python
# 細節保留
fourier_mode = "unified_detail"
fourier_weight = 0.09
fourier_high_freq_weight = 2.5
```

## 🛠️ 故障排除 Troubleshooting

| 問題 Problem | 症狀 Symptom | 解決方案 Solution |
|--------------|--------------|-------------------|
| 損失過大 | > 10.0 | `fourier_weight = 0.01` |
| 損失過小 | < 0.001 | `fourier_mode = "unified_detail"` |
| 記憶體不足 | OOM | `fourier_mode = "unified_basic"` |
| 訓練不穩定 | 震盪 | `fourier_warmup_steps = 500` |
| 效果不明顯 | 無改善 | `fourier_weight = 0.08` |

## 🎚️ 權重調整指南 Weight Tuning Guide

```
超保守 Ultra Conservative: 0.005 - 0.01
保守 Conservative:        0.01 - 0.03
平衡 Balanced:            0.03 - 0.06  ⭐ 推薦
積極 Aggressive:          0.06 - 0.10
超積極 Ultra Aggressive:   0.10 - 0.15
```

## 🔧 進階配置 Advanced Configuration

### 自定義整合模式 Custom Unified Mode
```python
fourier_mode = "unified"
enable_multiscale = True
enable_frequency_weighting = True
enable_adaptive = True
scales = [1, 2, 4]
adaptive_mode = "cosine"  # linear, cosine, exponential
max_weight = 2.5
min_weight = 0.8
```

### 三種自適應曲線 Three Adaptive Curves
```python
adaptive_mode = "linear"       # 線性衰減，穩定平滑
adaptive_mode = "cosine"       # 餘弦衰減，中期緩和 ⭐
adaptive_mode = "exponential"  # 指數衰減，早期激進
```

## 📊 效果對比 Performance Comparison

### 計算開銷 Computational Cost
```
basic < weighted < adaptive ≈ unified_basic < unified_balanced < multiscale < unified_detail
```

### 內存使用 Memory Usage
```
basic < weighted < adaptive < unified_basic < multiscale < unified_balanced < unified_detail
```

### 效果品質 Quality
```
basic < weighted < multiscale < adaptive < unified_balanced < unified_detail
```

## ⚠️ 注意事項 Important Notes

### ✅ 最佳實踐 Best Practices
- 🎯 **從預設開始**: 使用 `unified_balanced` 模式
- 📈 **逐步調整**: 根據效果調整權重
- 📊 **監控比例**: 傅立葉損失應為基礎損失的 5%-20%
- 🔄 **設置預熱**: 總是使用適當的預熱期
- 📝 **記錄實驗**: 追蹤不同配置的效果

### ❌ 常見錯誤 Common Mistakes
- ❌ 權重過高 (> 0.15)
- ❌ 跳過預熱期
- ❌ 忽略監控損失比例
- ❌ 一開始就用最複雜配置

## 🚀 快速開始範例 Quick Start Examples

### 30秒快速配置 30-Second Setup
```bash
# 複製貼上即可使用
python train_network.py \
  --loss_type fourier \
  --fourier_mode unified_balanced \
  --fourier_weight 0.06 \
  [其他參數...]
```

### 5分鐘自定義配置 5-Minute Custom Setup
```python
from library.train_util import apply_fourier_loss_to_args

# 根據需求選擇模式
mode = "unified_detail"  # 高品質
# mode = "unified_balanced"  # 平衡
# mode = "unified_adaptive"  # 自適應

apply_fourier_loss_to_args(args, mode=mode)

# 可選：微調權重
args.fourier_weight = 0.08  # 根據實際效果調整
```

## 📞 快速幫助 Quick Help

### 🔗 相關文檔
- 📚 [完整指南](./FOURIER_LOSS_GUIDE.md) - 詳細功能說明
- 💾 [程式範例](./fourier_loss_examples.py) - 實用配置代碼
- 🧪 [測試腳本](./test_unified_fourier_loss.py) - 功能驗證

### 🆘 問題求助
1. **檢查基礎**: 確認基本參數設置正確
2. **查看日誌**: 觀察損失變化趨勢
3. **降級測試**: 從簡單模式開始測試
4. **參考文檔**: 查閱完整指南獲取詳細說明

### 💡 優化建議
- 🎯 效果不佳？嘗試 `unified_detail` 模式
- ⚡ 速度太慢？使用 `unified_basic` 模式
- 🔧 需要控制？切換到 `unified` 自定義模式
- 📈 想要最佳平衡？堅持使用 `unified_balanced` ⭐