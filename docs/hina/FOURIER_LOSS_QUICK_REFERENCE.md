# Fourier Loss 快速參考 Quick Reference

## 🚀 一行配置 One-Line Setup

```python
# 推薦：平衡配置 (Recommended: Balanced)
from library.train_util import apply_fourier_loss_to_args
apply_fourier_loss_to_args(args, mode="balanced")
```

## 📋 參數速查表 Parameter Cheat Sheet

### 基本配置 Basic Configuration
```bash
--loss_type fourier
--fourier_weight 0.05
--fourier_mode weighted
--fourier_warmup_steps 200
```

### 四種模式對比 Four Modes Comparison

| 模式 Mode | 用途 Usage | 權重 Weight | 性能 Performance |
|-----------|------------|-------------|------------------|
| `basic` | 快速測試 | 0.02 | ⚡⚡⚡ |
| `weighted` ⭐ | 日常使用 | 0.05 | ⚡⚡ |
| `multiscale` | 高品質 | 0.08 | ⚡ |
| `adaptive` | 長期訓練 | 0.05 | ⚡⚡ |

## 🎯 應用場景配置 Application Configs

### 圖像生成 Image Generation
```python
fourier_mode = "weighted"
fourier_weight = 0.03
fourier_high_freq_weight = 1.5
```

### 超分辨率 Super Resolution
```python
fourier_mode = "multiscale"
fourier_weight = 0.08
fourier_scales = [1, 2, 4]
```

### 風格轉換 Style Transfer
```python
fourier_mode = "adaptive"
fourier_weight = 0.05
fourier_adaptive_max_weight = 2.5
```

## 🛠️ 故障排除 Troubleshooting

| 問題 Problem | 症狀 Symptom | 解決方案 Solution |
|--------------|--------------|-------------------|
| 損失過大 | > 10.0 | `fourier_weight = 0.01` |
| 損失過小 | < 0.001 | `fourier_weight = 0.08` |
| 記憶體不足 | OOM | `fourier_mode = "basic"` |
| 訓練不穩定 | 震盪 | `fourier_warmup_steps = 500` |

## 🎚️ 權重調整指南 Weight Tuning Guide

```
保守 Conservative: 0.01 - 0.02
平衡 Balanced:     0.03 - 0.06  ⭐
激進 Aggressive:   0.07 - 0.12
```

## ⚠️ 注意事項 Important Notes

- ✅ 從小權重開始 (Start with small weights)
- ✅ 設置預熱期 (Set warmup period)
- ✅ 監控損失比例 (Monitor loss ratio)
- ❌ 避免權重 > 0.2 (Avoid weights > 0.2)

## 📞 快速幫助 Quick Help

需要詳細說明？查看完整指南：[FOURIER_LOSS_GUIDE.md](./FOURIER_LOSS_GUIDE.md)