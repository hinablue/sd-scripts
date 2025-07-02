# 重構版 Automagic_CameAMP 優化器

這是 Automagic_CameAMP 優化器的重構版本，專門針對 LoRA 訓練和深度學習模型微調進行優化。重構後的版本具有更好的代碼結構、可維護性和擴展性。

## 🌟 主要特性

### 🔧 核心優化技術
- **CAME (Confidence-guided Adaptive Memory Efficient)** 算法
- **自適應梯度正則化 (AGR)**
- **扭矩感知動量 (Torque-Aware Momentum)**
- **選擇性投影衰減 (SPD)**
- **參數級自適應學習率遮罩**

### 🎯 LoRA 特化功能
- **低秩正則化**: 使用 SVD 分解鼓勵學習低秩結構
- **秩感知權重衰減**: 對不同秩的成分採用不同的衰減策略
- **低秩方向強化**: 在動量更新中強調低秩方向

### 🛡️ 過擬合控制
- **邊緣抑制**: 使用拉普拉斯算子檢測和抑制邊緣過擬合
- **頻率感知**: FFT 分析高頻噪聲並施加懲罰
- **背景正則化**: 減少背景區域的無效更新
- **空間感知**: 根據空間變異數調整優化策略

### 📈 階段性優化
- **早期預熱階段**: 激進探索，使用扭矩感知動量
- **後期預熱階段**: 過渡階段，逐步穩定
- **穩定階段**: 謹慎優化，強調一致性
- **成熟階段**: 精細調整，保持穩定性

## 🏗️ 重構改進

### 📦 模組化設計
- **策略模式**: 正則化和動量策略可獨立替換
- **工廠模式**: 提供便利函數快速創建優化器
- **組件分離**: 將複雜功能分解為獨立類別

### 🔍 程式碼品質
- **完整的類型提示**: 提高代碼可讀性和 IDE 支援
- **詳細的文檔**: 每個類別和方法都有完整的 docstring
- **錯誤處理**: 輸入驗證和異常處理
- **測試友好**: 純函數設計，易於單元測試

### 🚀 性能優化
- **記憶體效率**: 及時清理不需要的狀態
- **計算優化**: 避免不必要的張量操作
- **異常處理**: 優雅處理計算失敗（如 SVD 分解）

## 📋 安裝和使用

### 基本使用

```python
from automagic_cameamp_refactored import create_lora_optimizer

# LoRA 微調
optimizer = create_lora_optimizer(
    model.parameters(),
    lr=1e-4,
    warmup_steps=500
)

# 訓練迴圈
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### 自定義配置

```python
from automagic_cameamp_refactored import OptimizerConfig, RefactoredAutomagicCameAMP

# 創建自定義配置
config = OptimizerConfig(
    lr=2e-4,
    warmup_steps=300,
    edge_suppression=True,
    spatial_awareness=True,
    lora_rank_penalty=True,
    verbose=True
)

# 創建優化器
optimizer = RefactoredAutomagicCameAMP(model.parameters(), config)
```

### 便利函數

```python
# LoRA 微調
lora_optimizer = create_lora_optimizer(params, lr=1e-4, warmup_steps=500)

# 全量微調
full_optimizer = create_full_finetune_optimizer(params, lr=5e-5)
```

## 🔧 配置參數

### 基本參數
- `lr`: 學習率 (默認: 1e-6)
- `min_lr`: 最小學習率 (默認: 1e-7)
- `max_lr`: 最大學習率 (默認: 1e-3)
- `warmup_steps`: 預熱步數 (默認: 500)

### 邊緣和背景控制
- `edge_suppression`: 是否啟用邊緣抑制 (默認: True)
- `edge_threshold`: 邊緣檢測閾值 (默認: 0.6)
- `edge_penalty`: 邊緣懲罰強度 (默認: 0.1)
- `background_regularization`: 是否啟用背景正則化 (默認: True)

### LoRA 特定參數
- `lora_rank_penalty`: 是否啟用低秩懲罰 (默認: True)
- `rank_penalty_strength`: 秩懲罰強度 (默認: 0.01)
- `low_rank_emphasis`: 低秩強調因子 (默認: 1.2)

### 空間感知參數
- `spatial_awareness`: 是否啟用空間感知 (默認: True)
- `frequency_penalty`: 頻率懲罰強度 (默認: 0.05)
- `detail_preservation`: 細節保留因子 (默認: 0.8)

## 📊 使用場景

### 1. LoRA 微調
```python
# 最適合 LoRA 微調的配置
optimizer = create_lora_optimizer(
    model.parameters(),
    lr=1e-4,
    warmup_steps=500,
    edge_suppression=True,
    spatial_awareness=True
)
```

### 2. 全量微調
```python
# 全量微調配置
optimizer = create_full_finetune_optimizer(
    model.parameters(),
    lr=5e-5,
    warmup_steps=200,
    edge_suppression=False  # 全量微調時可以關閉
)
```

### 3. 圖像模型訓練
```python
# 針對圖像模型的配置
config = OptimizerConfig(
    lr=1e-4,
    edge_suppression=True,    # 重要：抑制邊緣過擬合
    frequency_penalty=0.08,   # 增強：抑制高頻噪聲
    spatial_awareness=True,   # 重要：空間感知
    background_regularization=True  # 減少背景過擬合
)
```

### 4. 語言模型微調
```python
# 針對語言模型的配置
config = OptimizerConfig(
    lr=2e-5,
    lora_rank_penalty=True,   # 重要：低秩結構
    edge_suppression=False,   # 語言模型通常不需要
    spatial_awareness=False,  # 1D 序列不需要空間感知
    warmup_steps=1000        # 語言模型需要更長預熱
)
```

## 🔍 進階功能

### 自定義正則化策略
```python
from automagic_cameamp_refactored import RegularizationStrategy

class CustomRegularizer(RegularizationStrategy):
    def apply(self, grad, **kwargs):
        # 實現自定義正則化邏輯
        return modified_grad

# 可以通過繼承擴展功能
```

### 狀態監控
```python
# 啟用詳細輸出
config = OptimizerConfig(verbose=True)
optimizer = RefactoredAutomagicCameAMP(model.parameters(), config)

# 每步都會輸出學習率統計
```

### 檢查點保存
```python
# 保存狀態
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': optimizer.config.__dict__
}

# 載入狀態
new_optimizer = RefactoredAutomagicCameAMP(model.parameters(), config)
new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 🚨 注意事項

### 記憶體使用
- 優化器會為每個參數維護多個狀態張量
- 對於大型模型，建議監控記憶體使用情況
- 可以通過調整 `warmup_steps` 來控制記憶體使用

### 計算成本
- SVD 分解和 FFT 變換會增加計算成本
- 可以通過關閉不需要的功能來降低成本
- 生產環境建議關閉 `verbose` 模式

### 數值穩定性
- 優化器內建了多種數值穩定性保護
- 極端情況下可能需要調整 `eps` 參數
- 建議使用混合精度訓練時特別注意

## 📈 性能建議

### 學習率設定
- LoRA 微調：建議 1e-4 到 1e-5
- 全量微調：建議 5e-5 到 1e-5
- 大型模型：建議更小的學習率

### 預熱步數
- 小型模型：200-500 步
- 中型模型：500-1000 步
- 大型模型：1000-2000 步

### 功能開關
- 圖像任務：啟用 edge_suppression 和 spatial_awareness
- 文本任務：關閉 spatial_awareness，啟用 lora_rank_penalty
- 性能優先：關閉不必要的正則化功能

## 🧪 範例程式

詳細的使用範例請參考 `examples/optimizer_usage_example.py`，包含：
- LoRA 微調範例
- 全量微調範例
- 自定義配置範例
- 狀態保存和載入
- 錯誤處理
- 性能比較

## 🤝 貢獻指南

歡迎貢獻代碼！請遵循以下指南：
1. 保持代碼風格一致
2. 添加適當的類型提示
3. 撰寫完整的 docstring
4. 添加單元測試
5. 更新文檔

## 📄 授權

本項目採用 MIT 授權條款。

## 🙏 致謝

感謝原始 Automagic_CameAMP 優化器的作者，本重構版本在其基礎上進行了大幅改進和優化。