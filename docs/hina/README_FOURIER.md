# Fourier Loss 文檔索引 Documentation Index

## 📚 文檔結構 Documentation Structure

本目錄包含 Fourier Loss 功能的完整文檔，按使用場景組織：

This directory contains comprehensive documentation for Fourier Loss functionality, organized by use case:

### 🎯 快速入門 Quick Start
- **[快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md)** - 一頁速查，包含常用配置和故障排除
- **[範例配置](./fourier_loss_examples.py)** - 實用的配置範例代碼
- **[測試腳本](./test_unified_fourier_loss.py)** - 新功能驗證和性能測試

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
3. 使用 [測試腳本](./test_unified_fourier_loss.py) 驗證和調優設定
4. 根據需要進行自定義調整

## 📋 功能概覽 Feature Overview

### ✨ 主要特性 Key Features
- 🎯 **五種損失模式**: Basic、Weighted、Multiscale、Adaptive、**Unified (新)**
- 🧠 **智能整合**: 結合多尺度、頻率加權和自適應的統一實現
- 🛡️ **數值穩定性**: 內建正規化和約束機制
- ⚡ **性能優化**: 智能權重調整和記憶體管理
- 🎛️ **靈活配置**: 豐富的參數選項和預設配置

### 🌟 最新功能 Latest Features (v2.0)
- **整合型損失**: 結合三種策略的統一計算架構
- **三種自適應曲線**: Linear、Cosine、Exponential 選擇
- **動態組合權重**: 根據訓練進度自動調整策略比例
- **預設配置模式**: unified_basic、unified_balanced、unified_detail、unified_adaptive
- **模組化設計**: 可選擇性啟用/禁用各個組件

### 📈 適用場景 Use Cases
- 🎨 圖像生成 (Image Generation)
- 🔍 超分辨率 (Super Resolution)
- 🎭 風格轉換 (Style Transfer)
- 🖼️ 圖像修復 (Image Restoration)
- 🔧 細節增強 (Detail Enhancement)

## 🔧 快速配置 Quick Configuration

### 🌟 一行配置 One-Line Setup (推薦)
```python
from library.train_util import apply_fourier_loss_to_args
apply_fourier_loss_to_args(args, mode="unified_balanced")  # 最新推薦配置
```

### 命令行使用 Command Line
```bash
# 基礎整合模式
python train_network.py \
  --loss_type fourier \
  --fourier_mode unified_balanced \
  --fourier_weight 0.06

# 高品質細節模式
python train_network.py \
  --loss_type fourier \
  --fourier_mode unified_detail \
  --fourier_weight 0.08

# 自定義整合模式
python train_network.py \
  --loss_type fourier \
  --fourier_mode unified \
  --fourier_scales 1,2,4 \
  --fourier_adaptive_mode cosine
```

### 配置比較 Configuration Comparison
| 模式 Mode | 適用場景 Use Case | 權重 Weight | 特點 Features |
|-----------|------------------|-------------|---------------|
| `unified_basic` | 快速測試 | 0.03 | 📱 輕量級 |
| `unified_balanced` ⭐ | 日常使用 | 0.06 | 🎯 推薦 |
| `unified_detail` | 高品質 | 0.08 | 🔍 細節豐富 |
| `unified_adaptive` | 復雜場景 | 0.07 | 🧠 智能調整 |

## 📊 性能指標 Performance Metrics

### 計算效率 Computational Efficiency
```
unified_basic > unified_balanced > unified_adaptive > unified_detail
```

### 記憶體使用 Memory Usage
```
unified_basic < unified_balanced < unified_adaptive < unified_detail
```

### 效果品質 Quality Impact
```
unified_basic < unified_balanced < unified_adaptive < unified_detail
```

## 🛠️ 故障排除 Quick Troubleshooting

### 常見問題 Common Issues
| 問題 | 現象 | 解決方案 |
|------|------|----------|
| 損失過大 | >10.0 | 降低權重或增加預熱 |
| 效果不明顯 | 無改善 | 使用 unified_detail 模式 |
| 記憶體不足 | OOM | 切換到 unified_basic |
| 訓練不穩定 | 損失震盪 | 增加預熱步數 |

### 優化建議 Optimization Tips
- 🎯 **新手**: 直接使用 `unified_balanced`
- ⚡ **追求速度**: 選擇 `unified_basic`
- 🔍 **追求品質**: 選擇 `unified_detail`
- 🧠 **復雜場景**: 選擇 `unified_adaptive`

## 📚 文檔更新 Documentation Updates

### v2.0 更新內容 Version 2.0 Updates
- ✨ 新增整合型傅立葉損失功能文檔
- 🔄 更新所有配置範例和參數說明
- 📊 新增性能對比和選擇指南
- 🧪 提供完整的測試腳本和驗證方法
- 📖 重新組織文檔結構，更易導覽

### 向後兼容性 Backward Compatibility
- ✅ 所有舊版配置完全兼容
- ✅ 原有四種模式保持不變
- ✅ 參數名稱和用法保持一致
- 🔄 建議逐步遷移到新的整合模式

## 🤝 貢獻指南 Contribution Guide

### 如何改進 How to Improve
1. **測試新配置**: 在不同場景下測試整合模式
2. **分享經驗**: 提供實際使用案例和效果對比
3. **回報問題**: 發現問題時提供詳細的重現步驟
4. **建議優化**: 提出新的功能需求或改進建議

### 提交反饋 Submit Feedback
- 📧 技術問題：通過 GitHub Issues
- 💡 功能建議：技術討論區
- 📝 文檔改進：Pull Request
- 🗣️ 經驗分享：社區論壇

## 🔗 相關資源 Related Resources

### 核心文檔 Core Documentation
- [快速參考](./FOURIER_LOSS_QUICK_REFERENCE.md) - 速查手冊
- [完整指南](./FOURIER_LOSS_GUIDE.md) - 詳細說明
- [程式範例](./fourier_loss_examples.py) - 實用代碼
- [測試腳本](./test_unified_fourier_loss.py) - 功能驗證

### 其他相關文檔 Other Related Docs
- [LoRA 優化指南](./README_LoRA_Optimization.md)
- [記憶體優化指南](./MEMORY_OPTIMIZED_ADAPTIVE_ADAMW_GUIDE.md)
- [自定義優化器指南](./CUSTOM_OPTIMIZER_USAGE_GUIDE.md)

---

**📌 重要提醒**:
- 新用戶建議從 `unified_balanced` 模式開始
- 定期查看文檔更新以獲取最新功能
- 遇到問題時優先查閱快速參考和故障排除

**📌 Important Notes**:
- New users should start with `unified_balanced` mode
- Check documentation updates regularly for latest features
- Consult quick reference and troubleshooting first when encountering issues