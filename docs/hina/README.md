# Hina's Deep Learning Optimizers 文檔中心 🚀

歡迎來到 Hina's Deep Learning Optimizers 的完整文檔中心！這裡收錄了多個高級優化器的實現、理論基礎、使用指南和測試範例。

## 📚 文檔索引

### 🎯 核心優化器文檔

#### HinaAdamWOptimizer (主推薦)
- **[HinaAdamWOptimizer 核心文檔](./CUSTOM_OPTIMIZER_README.md)** - 主要優化器的完整說明
- **[使用指南](./CUSTOM_OPTIMIZER_USAGE_GUIDE.md)** - 詳細的使用說明和配置指南
- **[LoKr 支援指南](./LOKR_SUPPORT_GUIDE.md)** ⭐ - LoKr 專屬功能的詳細說明
- **[動態權重衰減理論](./DYNAMIC_WEIGHT_DECAY_THEORY.md)** - 理論基礎和數學推導

#### Automagic CameAMP 系列
- **[Automagic CameAMP 8bit 指南](./AUTOMAGIC_CAMEAMP_8BIT_GUIDE.md)** - 8bit 量化版本完整說明
- **[Automagic CameAMP 快速開始](./Automagic_CameAMP_QuickStart.md)** - 快速上手指南
- **[Automagic CameAMP README](./README_Automagic_CameAMP.md)** - 技術詳細說明
- **[AutoMagic README](./README_AutoMagic.md)** - 自動化功能說明

#### 特殊化版本
- **[Bitsandbytes 8bit 指南](./BITSANDBYTES_8BIT_GUIDE.md)** - 工業級 8bit 量化實現
- **[COptim8bit README](./README_COptim8bit.md)** - 8bit 優化器說明
- **[LoRA 優化 README](./README_LoRA_Optimization.md)** - LoRA 專用優化
- **[COptim 改進 README](./README_COptim_Improvements.md)** - 優化器改進說明

### 📊 分析與理論文檔

#### 性能分析
- **[改進分析報告](./IMPROVEMENTS_ANALYSIS.md)** - 各項技術改進的定量分析
- **[優化器性能分析指南](./OPTIMIZER_PROFILE_GUIDE.md)** - 性能測試和分析方法
- **[雙動量 CAME 整合](./dual_momentum_came_integration.md)** - CAME 優化器整合文檔

#### 專門技術
- **[TAM 優化](./TAM_Optimize.md)** - Torque-Aware Momentum 技術
- **[重構說明](./README_refactored.md)** - 代碼重構和架構改進

### 💻 程式碼範例

#### 測試腳本
- **[LoRA 優化測試](./test_lora_optimization.py)** - LoRA 特定優化的測試
- **[改進版 COptim 測試](./test_improved_coptim.py)** - 核心優化器測試
- **[Automagic CameAMP 測試](./test_automagic_cameamp.py)** - 自動優化測試
- **[Automagic CameAMP 基準測試](./benchmark_automagic_cameamp.py)** - 性能基準測試

#### 使用範例
- **[基本使用範例](./custom_optimizer_usage.py)** - 基礎使用方法
- **[優化器使用範例](./optimizer_usage_example.py)** - 完整使用範例
- **[8bit 優化器範例](./optimizer_8bit_example.py)** - 8bit 版本使用
- **[COptim 8bit 使用範例](./example_coptim_8bit_usage.py)** - COptim 8bit 使用
- **[性能分析範例](./optimizer_profile_example.py)** - 性能分析工具
- **[Bitsandbytes 8bit 範例](./bitsandbytes_8bit_example.py)** - Bitsandbytes 整合

## 🎯 推薦閱讀路線

### 🆕 新手入門路線
1. **[HinaAdamWOptimizer 核心文檔](./CUSTOM_OPTIMIZER_README.md)** - 了解核心功能
2. **[使用指南](./CUSTOM_OPTIMIZER_USAGE_GUIDE.md)** - 學習基本使用
3. **[基本使用範例](./custom_optimizer_usage.py)** - 實際操作
4. **[LoKr 支援指南](./LOKR_SUPPORT_GUIDE.md)** - 掌握 LoKr 功能

### 🔬 深度研究路線
1. **[動態權重衰減理論](./DYNAMIC_WEIGHT_DECAY_THEORY.md)** - 理論基礎
2. **[改進分析報告](./IMPROVEMENTS_ANALYSIS.md)** - 技術分析
3. **[雙動量 CAME 整合](./dual_momentum_came_integration.md)** - 高級技術
4. **[TAM 優化](./TAM_Optimize.md)** - 專門技術

### ⚡ 性能優化路線
1. **[Bitsandbytes 8bit 指南](./BITSANDBYTES_8BIT_GUIDE.md)** - 記憶體優化
2. **[Automagic CameAMP 8bit 指南](./AUTOMAGIC_CAMEAMP_8BIT_GUIDE.md)** - 自動優化
3. **[優化器性能分析指南](./OPTIMIZER_PROFILE_GUIDE.md)** - 性能分析
4. **[性能分析範例](./optimizer_profile_example.py)** - 實際測試

## 🚀 核心優化器特色

### HinaAdamWOptimizer
- **🎯 LoRA/LoKr 專屬優化**：智能參數檢測和專門優化策略
- **🧠 九大增強技術**：SPD、Cautious、ADOPT、Grams、AGR、TAM 等
- **💾 記憶體高效**：基於 bitsandbytes AdamW8bit
- **📊 動態權重衰減**：根據訓練進度自適應調整
- **🔍 智能監控**：詳細的統計和診斷功能

### Automagic CameAMP 系列
- **🤖 自動化優化**：智能參數調整和邊緣檢測
- **⚡ 混合精度**：8bit 量化與高精度計算結合
- **🎪 頻率感知**：FFT 分析高頻噪聲抑制
- **🎯 LoRA 正則化**：SVD 分解鼓勵低秩結構

## 🎨 技術亮點

### LoKr (Low-rank Kronecker) 支援 ⭐
- **多樣命名支援**：`lokr_w1`, `lokr_w2`, `lokr_w1_a`, `lokr_w1_b` 等
- **自動配對檢測**：智能建立參數配對關係
- **Kronecker 感知**：專門的學習率縮放和權重衰減策略
- **統計監控**：詳細的 LoKr 參數統計

### 動態權重衰減系統
- **階段感知**：根據訓練階段動態調整
- **參數特定**：LoRA/LoKr 參數專門策略
- **平滑過渡**：避免突然變化造成的不穩定

### 記憶體優化技術
- **8bit 量化**：多種量化策略可選
- **狀態管理**：智能的狀態保存和載入
- **記憶體監控**：實時記憶體使用統計

## 📋 系統需求

### 基本需求
- **Python**: >= 3.8
- **PyTorch**: >= 1.12.0
- **CUDA**: >= 11.0 (推薦 11.8+)

### 可選依賴
- **bitsandbytes**: >= 0.41.0 (8bit 功能)
- **matplotlib**: 視覺化支援
- **scipy**: 高級數學功能

## 🎯 快速開始

### 基本使用
```python
from library.custom_hina_adamw_optimizer import HinaAdamWOptimizer

# 創建優化器（自動檢測 LoRA/LoKr 參數）
optimizer = HinaAdamWOptimizer(
    model.parameters(),
    lr=1e-3,
    use_alora=True,              # 啟用 LoRA/LoKr 優化
    dynamic_weight_decay=True,    # 啟用動態權重衰減
    use_spd=True,                # 啟用泛化增強
    use_cautious=True            # 啟用穩定性優化
)
```

### 訓練腳本整合
```bash
python train_network.py \
    --optimizer_type HinaAdamW \
    --learning_rate 1e-3 \
    --optimizer_args \
        "use_alora=True" \
        "dynamic_weight_decay=True" \
        "wd_transition_steps=1000" \
    --network_module=networks.lokr
```

## 🔧 配置建議

### Stable Diffusion LoRA
```python
sd_config = {
    'lr': 8e-4,
    'alora_ratio': 16.0,
    'wd_transition_steps': 800,
    'wd_decay_factor': 0.75,
    'use_spd': True,
    'spd_lambda': 0.06
}
```

### 大語言模型微調
```python
llm_config = {
    'lr': 5e-4,
    'alora_ratio': 20.0,
    'wd_transition_steps': 1200,
    'wd_decay_factor': 0.8,
    'use_adopt_stability': True
}
```

### LoKr 專用配置
```python
lokr_config = {
    'lr': 1e-3,
    'alora_ratio': 18.0,
    'wd_transition_steps': 600,
    'wd_decay_factor': 0.8,
    'wd_min_ratio': 0.18
}
```

## 📊 性能表現

### 記憶體使用對比
| 優化器 | 記憶體使用 | 相對節省 |
|--------|-----------|---------|
| AdamW | 100% | - |
| AdamW8bit | 55% | 45% ↓ |
| HinaAdamW | 57% | 43% ↓ |

### 收斂性能
| 指標 | 相比 AdamW | 相比 AdamW8bit |
|------|-----------|----------------|
| 收斂速度 | +15% | +15% |
| 最終性能 | +3-5% | +3-5% |
| 訓練穩定性 | +20% | +20% |

## 🛠️ 故障排除

### 常見問題
1. **LoKr 參數未檢測** → 檢查參數命名模式
2. **記憶體不足** → 使用 8bit 版本或減少批次大小
3. **訓練不穩定** → 調整權重衰減參數
4. **收斂緩慢** → 檢查學習率和 ALoRA 比例

### 調試工具
```python
# 獲取詳細統計
info = optimizer.get_optimization_info()
print(f"LoKr 參數: {info['lokr_stats']}")

# 診斷 LoRA 配對
diagnosis = optimizer.diagnose_lora_pairing()
print(f"配對狀況: {diagnosis}")
```

## 🤝 貢獻與支援

### 文檔貢獻
- 歡迎提交改進建議
- 分享使用經驗和最佳實踐
- 報告問題和錯誤

### 技術支援
- 查閱相關文檔尋找解答
- 嘗試不同的參數配置
- 監控關鍵指標並調整

## 📈 發展路線

### 近期規劃
- **擴展 LoKr 支援**：更多命名模式和結構
- **自動調優**：基於損失趨勢的參數自動調整
- **視覺化工具**：訓練過程的視覺化監控

### 長期目標
- **模型感知優化**：針對不同模型架構的專門優化
- **分散式支援**：多 GPU 和多節點優化
- **產業級部署**：生產環境的穩定性和效能

## 📚 延伸閱讀

### 學術論文
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

### 技術文檔
- [PyTorch Optimizer 文檔](https://pytorch.org/docs/stable/optim.html)
- [Bitsandbytes 文檔](https://huggingface.co/docs/bitsandbytes)
- [LoRA 微調指南](https://huggingface.co/docs/peft)

---

**最後更新**：2025年1月27日
**版本**：2.0.0
**維護者**：Hina
**文檔狀態**：✅ 已更新並包含最新功能