# Fourier Loss 文檔索引 Documentation Index

## 📚 文檔結構 Documentation Structure

本目錄包含 Fourier Loss 功能的完整文檔，按使用場景組織：

This directory contains comprehensive documentation for Fourier Loss functionality, organized by use case:

### 🎯 快速入門 Quick Start
- **[快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md)** - 一頁速查，包含常用配置和故障排除
- **[範例配置](./fourier_loss_examples.py)** - 實用的配置範例代碼

### 📖 詳細指南 Detailed Guide
- **[完整指南](./FOURIER_LOSS_GUIDE.md)** - 全面的功能說明、參數解釋和最佳實踐

## 🚀 推薦學習路徑 Recommended Learning Path

### 初學者 Beginners
1. 閱讀 [快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md) 了解基本概念
2. 使用 [範例配置](./fourier_loss_examples.py) 中的基礎配置
3. 參考 [完整指南](./FOURIER_LOSS_GUIDE.md) 的"快速開始"章節

### 進階用戶 Advanced Users
1. 深入學習 [完整指南](./FOURIER_LOSS_GUIDE.md) 的技術原理
2. 根據應用場景選擇 [範例配置](./fourier_loss_examples.py) 中的專用配置
3. 根據需要進行自定義調整

## 📋 功能概覽 Feature Overview

### ✨ 主要特性 Key Features
- 🎯 **四種損失模式**: Basic、Weighted、Multiscale、Adaptive
- 🛡️ **數值穩定性**: 內建正規化和約束機制
- ⚡ **性能優化**: 智能權重調整和記憶體管理
- 🎛️ **靈活配置**: 豐富的參數選項和預設配置

### 📈 適用場景 Use Cases
- 🎨 圖像生成 (Image Generation)
- 🔍 超分辨率 (Super Resolution)
- 🎭 風格轉換 (Style Transfer)
- 🖼️ 圖像修復 (Image Restoration)

## 🔧 快速配置 Quick Configuration

### 一行配置 One-Line Setup
```python
from library.train_util import apply_fourier_loss_to_args
apply_fourier_loss_to_args(args, mode="balanced")  # 推薦配置
```

### 命令行使用 Command Line
```bash
python train_network.py \
  --loss_type fourier \
  --fourier_weight 0.05 \
  --fourier_mode weighted \
  --fourier_warmup_steps 200
```

## 🛠️ 故障排除 Troubleshooting

### 常見問題 Common Issues
| 問題 | 解決方案 | 參考文檔 |
|------|----------|----------|
| 損失值過大 | 降低 `fourier_weight` | [快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md#故障排除) |
| 訓練不穩定 | 增加 `fourier_warmup_steps` | [完整指南](./FOURIER_LOSS_GUIDE.md#故障排除) |
| 記憶體不足 | 使用 `basic` 模式 | [範例配置](./fourier_loss_examples.py) |

## 📞 支援與反饋 Support & Feedback

### 🤝 獲取幫助 Getting Help
1. **常見問題**: 查看 [故障排除](./FOURIER_LOSS_GUIDE.md#故障排除) 章節
2. **配置問題**: 參考 [範例配置](./fourier_loss_examples.py)
3. **技術問題**: 閱讀 [技術原理](./FOURIER_LOSS_GUIDE.md#技術原理) 章節

### 📝 反饋渠道 Feedback Channels
- GitHub Issues
- 技術討論區
- 文檔改進建議

## 🔄 版本說明 Version Notes

### 最新更新 Latest Updates
- ✅ 數值穩定性大幅改善
- ✅ 新增四種損失模式
- ✅ 完善的文檔和範例
- ✅ 智能的動態權重調整

### 向後兼容 Backward Compatibility
現有配置完全兼容，無需修改代碼。

---

**快速導航 Quick Navigation**:
- [🚀 快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md)
- [📖 完整指南](./FOURIER_LOSS_GUIDE.md)
- [💻 範例配置](./fourier_loss_examples.py)

**需要幫助？** 從 [快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md) 開始，或查看 [完整指南](./FOURIER_LOSS_GUIDE.md) 獲取詳細信息。