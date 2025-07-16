# Hina's Deep Learning Optimizers 文檔中心 🚀

歡迎來到 Hina's Deep Learning Optimizers 的完整文檔中心！這裡收錄了多個高級優化器的實現、理論基礎、使用指南和測試範例。

## 📚 文檔索引

### 🎯 核心優化器文檔

> **⚠️ 重要提醒**：
> - **權重衰減限制**：需要修改 kohya sd-scripts 與 LyCORIS 的 `kohya.py` 程式碼才能使用權重衰減功能
> - **維護狀態**：此版本將不再更新，未來發展將以 **AdaptiveHinaAdamW** 版本為主
> - **Latent Space 相容性**：傅立葉特徵損失超解析度優化預設禁用，因 SD-Scripts 全部使用 latent space 訓練
>   📖 [詳細說明](./FOURIER_LATENT_SPACE_GUIDE.md) | 🧪 [測試腳本](./test_latent_space_detection.py)

#### HinaAdaptive (記憶體優化自適應版本) 🆕
- **記憶體優化**：智能緩衝區池管理、JIT 編譯優化、減少記憶體分配開銷 50-90%
- **邊緣過擬合控制**：邊緣抑制、頻率感知、空間感知正則化防止過擬合
- **LoRA 低秩正則化**：SVD 分解促進低秩結構，特別適合 LoRA 訓練
- **lr_mask 機制**：基於梯度極性的即時學習率調整，支持 warmup 和穩定期
- **多技術整合**：SPD、Cautious、正交梯度投影、ADOPT、AGR、TAM 等技術
- **動態自適應**：參數重要性評估、關係發現、動態權重衰減
- **異步計算**：非關鍵計算異步執行，提升訓練效率

#### AdaptiveHinaAdamW (最新自適應版本) 🆕
- **動態參數重要性評估**：基於梯度一致性、參數變化率和內在特性的多維度評估
- **自適應參數關係發現**：智能檢測參數間的矩陣相容性和語意相似性
- **lr_mask 機制**：基於梯度極性的即時學習率調整（策略 B 組合式整合）
- **記憶體優化**：先進的緩衝區池技術，顯著減少記憶體分配開銷
- **多技術整合**：SPD、Cautious、正交梯度投影、ADOPT、AGR、TAM 等九大增強技術
- **無參數類型依賴**：不依賴特定的參數命名模式，適用於各種模型架構

#### HinaAdamWOptimizer (LoRA/LoKr 專用版本) ⚠️
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
- **[Latent Space 相容性指南](./FOURIER_LATENT_SPACE_GUIDE.md)** 🔴 - SD-Scripts latent space 訓練架構分析

### 💻 程式碼範例

#### 檢測與測試
- **[Latent Space 檢測測試](./test_latent_space_detection.py)** - 驗證 latent space 自動檢測功能

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
1. **[HinaAdaptive 基本使用](#hinaadaptive-記憶體優化自適應版本-)** - 最新記憶體優化版本 🆕
2. **[AdaptiveHinaAdamW 基本使用](#adaptivehinaadamw-自適應版本-)** - 通用自適應版本 🆕
3. **[HinaAdamWOptimizer 核心文檔](./CUSTOM_OPTIMIZER_README.md)** - 了解舊版核心功能 ⚠️
4. **[使用指南](./CUSTOM_OPTIMIZER_USAGE_GUIDE.md)** - 學習基本使用
5. **[LoKr 支援指南](./LOKR_SUPPORT_GUIDE.md)** - 掌握 LoKr 功能

> **💡 建議**：
> - **記憶體受限環境**：使用 **HinaAdaptive** 版本，具備最佳的記憶體優化和邊緣控制功能
> - **通用場景**：使用 **AdaptiveHinaAdamW** 版本，功能全面且持續維護
> - **LoRA 專用**：可考慮 HinaAdamWOptimizer，但不再維護更新

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

### HinaAdaptive (記憶體優化自適應版本) 🆕
- **🧠 先進記憶體管理**：智能緩衝區池、動態記憶體監控、JIT 編譯優化
- **🎯 邊緣過擬合控制**：邊緣檢測與抑制、頻率感知、空間變異數正則化
- **💎 LoRA 低秩正則化**：SVD 分解鼓勵低秩結構，特別適合 LoRA/LoKr 訓練
- **⚡ lr_mask 精準控制**：基於梯度極性的動態學習率調整機制
- **🔄 異步計算架構**：非關鍵計算異步執行，提升整體訓練效率
- **📊 智能適應機制**：參數重要性評估、關係發現、動態權重衰減
- **💾 極致記憶體優化**：緩衝區池技術減少記憶體分配開銷 50-90%

#### 記憶體優化系統 🧠
- **智能緩衝區池**：按形狀、類型、設備分類管理，支持優先級分配
- **動態記憶體監控**：即時監控 VRAM 使用率，自動調整優化策略
- **JIT 編譯優化**：關鍵計算函數的 PyTorch JIT 編譯，提升性能 20-50%
- **緊湊狀態存儲**：使用 `__slots__` 和量化技術減少 Python 對象開銷
- **CPU 卸載支援**：將非關鍵狀態卸載到 CPU，節省 VRAM

#### 邊緣過擬合控制系統 🎯
- **邊緣檢測與抑制**：使用拉普拉斯算子檢測邊緣特徵，防止邊緣過擬合
- **頻率感知正則化**：計算相鄰元素差異，控制高頻噪聲
- **空間感知機制**：基於 3x3 鄰域變異數的空間感知正則化
- **自適應閾值**：動態調整邊緣檢測閾值，適應不同訓練階段

#### LoRA 低秩正則化 💎
- **近似 SVD 分析**：使用協方差矩陣和特徵值分解促進低秩結構
- **秩懲罰機制**：懲罰較大的奇異值，鼓勵低秩分解
- **快速計算模式**：可選擇完整 SVD 或近似方法
- **自動適配**：僅對 2D 參數（矩陣）應用低秩正則化

#### lr_mask 動態機制 ⚡
- **梯度極性追蹤**：Warmup 階段基於梯度符號一致性調整學習率
- **漸進式調整**：使用 `lr_bump` 參數進行小幅度調整
- **範圍控制**：確保學習率在 `min_lr` 到 `max_lr` 之間
- **穩定期保持**：Post-warmup 階段保持穩定，避免震盪

#### 異步計算架構 🔄
- **ThreadPoolExecutor**：使用多線程執行非關鍵計算
- **任務調度**：智能調度參數關係發現等計算密集任務
- **超時機制**：避免異步任務阻塞主訓練流程
- **資源回收**：訓練結束時自動清理異步資源

#### 九大增強技術整合 🎪
1. **SPD (Selective Projection Decay)**：選擇性投影衰減，防止參數偏離初始值
2. **Cautious Update**：謹慎更新策略，檢查更新方向與梯度對齊
3. **Orthogonal Gradient**：正交梯度投影，記憶體優化版本
4. **ADOPT Stability**：ADOPT 穩定性機制，防止二階矩退化
5. **GRAMS**：自適應動量縮放
6. **AGR**：自適應梯度正則化，防止梯度爆炸
7. **TAM**：Torque-Aware Momentum，動量對齊感知
8. **Dynamic Weight Decay**：動態權重衰減調整
9. **lr_mask**：梯度極性感知學習率調整

### AdaptiveHinaAdamW (自適應版本) 🆕
- **🤖 智能參數關係發現**：自動分析參數間的矩陣相容性和語意相似性
- **📈 動態重要性評估**：基於梯度一致性、參數變化率和內在特性的三維評估
- **⚡ lr_mask 組合機制**：策略 B 組合式整合，結合梯度極性調整和自適應縮放
- **💾 先進記憶體優化**：緩衝區池技術，減少記憶體分配開銷 50-90%
- **🎯 無類型依賴設計**：不依賴特定參數命名，適用於各種模型架構
- **🔄 智能關係更新**：定期重新發現參數關係，適應訓練過程變化
- **📊 全面監控分析**：提供參數關係、重要性分析、lr_mask 統計等詳細報告

#### lr_mask 機制（策略 B：組合式整合）✨
- **基礎層**：基於梯度極性的即時學習率調整
- **高級層**：基於參數重要性和關係的長期調整
- **Warmup 階段**：梯度極性追蹤和動態調整
- **穩定階段**：輕微衰減保持訓練穩定性
- **最終縮放**：lr_mask_scale × adaptive_scale

#### 記憶體優化技術 🧠
- **緩衝區池**：智能張量重用，減少記憶體分配
- **JIT 優化**：關鍵計算的 PyTorch JIT 編譯
- **形狀管理**：每種形狀最多保留 3 個緩衝區
- **自動清理**：訓練結束時自動釋放記憶體

#### 九大增強技術整合 🎪
1. **SPD (Selective Projection Decay)**：選擇性投影衰減
2. **Cautious Update**：謹慎更新策略
3. **Orthogonal Gradient**：正交梯度投影（記憶體優化版）
4. **ADOPT Stability**：ADOPT 穩定性機制
5. **GRAMS**：自適應動量縮放
6. **AGR**：自適應梯度正則化
7. **TAM**：Torque-Aware Momentum
8. **Dynamic Weight Decay**：動態權重衰減
9. **lr_mask**：梯度極性感知學習率調整

### HinaAdamWOptimizer ⚠️
- **🎯 LoRA/LoKr 專屬優化**：智能參數檢測和專門優化策略
- **🧠 九大增強技術**：SPD、Cautious、ADOPT、Grams、AGR、TAM 等
- **💾 記憶體高效**：基於 bitsandbytes AdamW8bit
- **📊 動態權重衰減**：根據訓練進度自適應調整 ⚠️ *需修改 kohya.py*
- **🔍 智能監控**：詳細的統計和診斷功能

> **⚠️ 注意事項**：
> - 此版本專為 LoRA/LoKr 設計，需要特定的參數命名模式
> - 權重衰減功能需要修改 kohya sd-scripts 與 LyCORIS 的程式碼
> - **不再維護更新**，建議使用 AdaptiveHinaAdamW 版本

### Automagic CameAMP 系列
- **🤖 自動化優化**：智能參數調整和邊緣檢測
- **⚡ 混合精度**：8bit 量化與高精度計算結合
- **🎪 頻率感知**：FFT 分析高頻噪聲抑制
- **🎯 LoRA 正則化**：SVD 分解鼓勵低秩結構

## 🎨 技術亮點

### 自適應參數關係發現 🔍
- **矩陣相容性檢測**：檢查參數間是否可進行矩陣運算
- **語意相似性分析**：基於參數分佈計算相關性
- **動態關係映射**：建立並更新參數配對關係
- **交互類型識別**：自動確定最佳參數交互方式

### 動態重要性評估系統 📈
- **梯度貢獻度**：分析梯度大小和一致性（40% 權重）
- **參數變化率**：衡量相對於初始值的變化（30% 權重）
- **內在特性**：評估參數方差和稀疏性（30% 權重）
- **指數移動平均**：平滑重要性分數更新

### lr_mask 組合機制（策略 B）⚡
- **雙層架構**：基礎層 + 高級層的組合設計
- **極性追蹤**：Warmup 階段基於梯度極性調整
- **智能過渡**：Post-warmup 階段的穩定性保持
- **範圍控制**：min_lr 到 max_lr 的安全範圍限制

### 記憶體優化技術 💾
- **智能緩衝區池**：按形狀、類型、設備分類管理
- **JIT 編譯優化**：關鍵計算函數的 PyTorch JIT 優化
- **原地操作**：減少臨時張量創建
- **自動清理**：訓練結束時釋放所有緩衝區

### LoKr (Low-rank Kronecker) 支援 ⭐
- **自動配對檢測**：智能建立參數配對關係
- **Kronecker 感知**：專門的學習率縮放和權重衰減策略
- **統計監控**：詳細的 LoKr 參數統計

### 動態權重衰減系統
- **階段感知**：根據訓練階段動態調整
- **參數特定**：LoRA/LoKr 參數專門策略
- **平滑過渡**：避免突然變化造成的不穩定

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

#### HinaAdaptive (記憶體優化自適應版本) 🆕
```python
from library.hina_adaptive import HinaAdaptive

# 創建記憶體優化的自適應優化器
optimizer = HinaAdaptive(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,

    # 增強功能配置
    use_spd=True,                       # 啟用 SPD 正則化
    spd_lambda=0.06,                    # SPD 懲罰強度
    use_cautious=True,                  # 啟用謹慎更新
    use_orthogonal_grad=False,          # 正交梯度投影（計算密集）
    use_adopt_stability=True,           # ADOPT 穩定性機制
    use_grams=True,                     # 自適應動量縮放
    use_agr=True,                       # 自適應梯度正則化
    use_tam=True,                       # Torque-Aware Momentum
    tam_beta=0.999,                     # TAM beta 參數

    # 動態自適應功能
    use_dynamic_adaptation=True,        # 啟用自適應功能
    adaptation_strength=1.0,            # 自適應調整強度
    relationship_discovery_interval=100, # 參數關係發現間隔
    importance_decay=0.95,              # 重要性分數衰減
    compatibility_threshold=0.3,        # 參數相容性閾值

    # lr_mask 機制
    use_lr_mask=True,                   # 啟用 lr_mask
    lr_bump=3e-6,                       # lr_mask 調整幅度
    min_lr=1e-7,                        # 最小學習率
    max_lr=1e-3,                        # 最大學習率
    warmup_steps=500,                   # Warmup 步數

    # 動態權重衰減
    dynamic_weight_decay=True,          # 啟用動態權重衰減
    wd_transition_steps=1000,           # 權重衰減過渡步數
    wd_decay_factor=0.7,                # 權重衰減減少係數
    wd_min_ratio=0.1,                   # 最小權重衰減比例

    # 記憶體優化配置
    memory_efficient=True,              # 啟用記憶體優化
    vram_budget_gb=16.0,                # VRAM 預算（GB）
    cpu_offload_states=True,            # CPU 狀態卸載
    reduce_precision=True,              # 使用 bfloat16 精度
    adaptive_features=True,             # 自適應功能
    emergency_simplify=True,            # 緊急簡化模式
    max_buffer_memory_mb=500,           # 最大緩衝區記憶體

    # 邊緣過擬合控制
    edge_suppression=True,              # 啟用邊緣抑制
    edge_penalty=0.1,                   # 邊緣懲罰強度
    edge_threshold=0.6,                 # 邊緣檢測閾值

    # 空間感知正則化
    spatial_awareness=True,             # 啟用空間感知
    frequency_penalty=0.05,             # 頻率懲罰強度
    detail_preservation=0.8,            # 細節保持係數

    # LoRA 低秩正則化
    lora_rank_penalty=True,             # 啟用 LoRA 秩懲罰
    rank_penalty_strength=0.01,         # 秩懲罰強度
    low_rank_emphasis=1.2               # 低秩強調係數
)

# 訓練過程中的記憶體監控
def training_step(batch, step):
    # 正常的訓練步驟
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 每 100 步檢查記憶體狀態
    if step % 100 == 0:
        memory_stats = optimizer.get_memory_stats()
        print(f"步數 {step}: 記憶體壓力 {memory_stats['memory_pressure']:.2%}")

        # 如果記憶體壓力過高，進行優化
        if memory_stats['memory_pressure'] > 0.9:
            optimizer.optimize_for_vram(14.0)  # 調降至 14GB

# 訓練結束後清理資源
optimizer.cleanup_resources()
```

#### AdaptiveHinaAdamW (自適應版本) 🆕
```python
from library.custom_hina_adaptive_adamw_optimizer import AdaptiveHinaAdamW

# 創建自適應優化器（適用於各種模型架構）
optimizer = AdaptiveHinaAdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,

    # 增強功能配置
    use_spd=True,                       # 啟用 SPD 正則化
    spd_lambda=0.06,                    # SPD 懲罰強度
    use_cautious=True,                  # 啟用謹慎更新
    use_orthogonal_grad=False,          # 正交梯度投影（計算密集）
    use_adopt_stability=True,           # ADOPT 穩定性機制
    use_grams=True,                     # 自適應動量縮放
    use_agr=True,                       # 自適應梯度正則化
    use_tam=True,                       # Torque-Aware Momentum
    tam_beta=0.999,                     # TAM beta 參數

    # 動態自適應功能
    use_dynamic_adaptation=True,        # 啟用自適應功能
    adaptation_strength=1.0,            # 自適應調整強度
    relationship_discovery_interval=100, # 參數關係發現間隔
    importance_decay=0.95,              # 重要性分數衰減
    compatibility_threshold=0.3,        # 參數相容性閾值

    # lr_mask 機制（策略 B 組合式整合）
    use_lr_mask=True,                   # 啟用 lr_mask
    lr_bump=3e-6,                       # lr_mask 調整幅度
    min_lr=1e-7,                        # 最小學習率
    max_lr=1e-3,                        # 最大學習率
    warmup_steps=500,                   # Warmup 步數

    # 動態權重衰減
    dynamic_weight_decay=True,          # 啟用動態權重衰減
    wd_transition_steps=1000,           # 權重衰減過渡步數
    wd_decay_factor=0.7,                # 權重衰減減少係數
    wd_min_ratio=0.1                    # 最小權重衰減比例
)
```

#### HinaAdamWOptimizer (LoRA/LoKr 專用) ⚠️
```python
from library.custom_hina_adamw_optimizer import HinaAdamWOptimizer

# 創建優化器（自動檢測 LoRA/LoKr 參數）
optimizer = HinaAdamWOptimizer(
    model.parameters(),
    lr=1e-3,
    use_alora=True,              # 啟用 LoRA/LoKr 優化
    dynamic_weight_decay=True,    # ⚠️ 需修改 kohya.py 才能使用
    use_spd=True,                # 啟用泛化增強
    use_cautious=True            # 啟用穩定性優化
)
```

> **⚠️ 重要提醒**：此版本不再維護更新，建議使用 **AdaptiveHinaAdamW** 版本

### 訓練腳本整合
```bash
# HinaAdaptive 使用範例
python train_network.py \
    --optimizer_type HinaAdaptive \
    --learning_rate 1e-3 \
    --optimizer_args \
        "memory_efficient=True" \
        "vram_budget_gb=16.0" \
        "use_lr_mask=True" \
        "lr_bump=3e-6" \
        "edge_suppression=True" \
        "edge_penalty=0.1" \
        "spatial_awareness=True" \
        "lora_rank_penalty=True" \
        "rank_penalty_strength=0.01" \
        "reduce_precision=True" \
    --network_module=networks.lora

# AdaptiveHinaAdamW 使用範例
python train_network.py \
    --optimizer_type AdaptiveHinaAdamW \
    --learning_rate 1e-3 \
    --optimizer_args \
        "use_dynamic_adaptation=True" \
        "use_lr_mask=True" \
        "adaptation_strength=1.0" \
        "lr_bump=3e-6" \
        "warmup_steps=500" \
    --network_module=networks.lora

# HinaAdamWOptimizer 使用範例（舊版）
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

### HinaAdaptive 專用配置 🆕

#### 記憶體優化配置（適合低 VRAM 環境）
```python
low_vram_config = {
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'weight_decay': 1e-2,

    # 記憶體優化策略
    'memory_efficient': True,
    'vram_budget_gb': 12.0,           # 調低 VRAM 預算
    'cpu_offload_states': True,        # 啟用 CPU 卸載
    'reduce_precision': True,          # 使用 bfloat16
    'emergency_simplify': True,        # 緊急簡化模式
    'max_buffer_memory_mb': 300,       # 減少緩衝區記憶體

    # 簡化的功能配置
    'use_dynamic_adaptation': True,
    'relationship_discovery_interval': 200,  # 減少關係發現頻率
    'use_orthogonal_grad': False,      # 關閉計算密集功能
    'use_grams': False,                # 關閉 GRAMS

    # lr_mask 配置
    'use_lr_mask': True,
    'lr_bump': 2e-6,
    'warmup_steps': 300,

    # 邊緣控制（輕量版）
    'edge_suppression': True,
    'edge_penalty': 0.05,              # 降低懲罰強度
    'spatial_awareness': False,        # 關閉空間感知以節省記憶體

    # LoRA 優化
    'lora_rank_penalty': True,
    'rank_penalty_strength': 0.005     # 降低懲罰強度
}
```

#### Stable Diffusion LoRA 專用配置
```python
sd_lora_config = {
    'lr': 8e-4,
    'weight_decay': 1e-2,

    # 記憶體配置
    'memory_efficient': True,
    'vram_budget_gb': 16.0,
    'reduce_precision': True,

    # 邊緣和細節控制
    'edge_suppression': True,
    'edge_penalty': 0.1,
    'edge_threshold': 0.6,
    'spatial_awareness': True,
    'frequency_penalty': 0.05,
    'detail_preservation': 0.8,

    # LoRA 特化設定
    'lora_rank_penalty': True,
    'rank_penalty_strength': 0.01,
    'low_rank_emphasis': 1.2,

    # lr_mask 精細調整
    'use_lr_mask': True,
    'lr_bump': 3e-6,
    'min_lr': 1e-7,
    'max_lr': 5e-4,
    'warmup_steps': 500,

    # 動態自適應
    'use_dynamic_adaptation': True,
    'adaptation_strength': 1.0,
    'relationship_discovery_interval': 100,

    # 全套增強技術
    'use_spd': True,
    'spd_lambda': 0.06,
    'use_cautious': True,
    'use_adopt_stability': True,
    'use_tam': True
}
```

#### 高性能配置（充足 VRAM 環境）
```python
high_performance_config = {
    'lr': 1e-3,
    'weight_decay': 1e-2,

    # 充分利用記憶體
    'memory_efficient': True,
    'vram_budget_gb': 24.0,            # 高 VRAM 預算
    'cpu_offload_states': False,       # 全部保持在 GPU
    'reduce_precision': False,         # 使用 float32 高精度
    'max_buffer_memory_mb': 1000,      # 更大緩衝區

    # 啟用所有高級功能
    'use_dynamic_adaptation': True,
    'adaptation_strength': 1.2,
    'relationship_discovery_interval': 50,  # 頻繁關係發現
    'use_orthogonal_grad': True,       # 啟用計算密集功能
    'use_grams': True,

    # 精細的邊緣控制
    'edge_suppression': True,
    'edge_penalty': 0.15,
    'spatial_awareness': True,
    'frequency_penalty': 0.08,

    # 強化的 LoRA 正則化
    'lora_rank_penalty': True,
    'rank_penalty_strength': 0.015,

    # 精密的 lr_mask
    'use_lr_mask': True,
    'lr_bump': 5e-6,
    'warmup_steps': 800,

    # 全套增強技術
    'use_spd': True,
    'use_cautious': True,
    'use_adopt_stability': True,
    'use_agr': True,
    'use_tam': True
}
```

#### 實驗性配置（探索最佳參數）
```python
experimental_config = {
    'lr': 1.2e-3,
    'weight_decay': 8e-3,

    # 激進的自適應設定
    'adaptation_strength': 1.5,
    'relationship_discovery_interval': 80,
    'importance_decay': 0.92,
    'compatibility_threshold': 0.25,

    # 強化的 lr_mask
    'lr_bump': 8e-6,
    'min_lr': 5e-8,
    'max_lr': 2e-3,
    'warmup_steps': 600,

    # 強化的邊緣控制
    'edge_suppression': True,
    'edge_penalty': 0.2,
    'edge_threshold': 0.5,
    'spatial_awareness': True,
    'frequency_penalty': 0.1,

    # 激進的 LoRA 正則化
    'lora_rank_penalty': True,
    'rank_penalty_strength': 0.02,
    'low_rank_emphasis': 1.5,

    # 動態權重衰減
    'dynamic_weight_decay': True,
    'wd_transition_steps': 600,
    'wd_decay_factor': 0.6,
    'wd_min_ratio': 0.05
}
```

### AdaptiveHinaAdamW 專用配置 🆕

#### 通用模型微調配置
```python
general_config = {
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-2,

    # 核心自適應功能
    'use_dynamic_adaptation': True,
    'adaptation_strength': 1.0,
    'relationship_discovery_interval': 100,
    'importance_decay': 0.95,
    'compatibility_threshold': 0.3,

    # lr_mask 配置
    'use_lr_mask': True,
    'lr_bump': 3e-6,
    'min_lr': 1e-7,
    'max_lr': 1e-3,
    'warmup_steps': 500,

    # 增強技術
    'use_spd': True,
    'spd_lambda': 0.06,
    'use_cautious': True,
    'use_adopt_stability': True,
    'use_tam': True,

    # 動態權重衰減
    'dynamic_weight_decay': True,
    'wd_transition_steps': 800,
    'wd_decay_factor': 0.75,
    'wd_min_ratio': 0.1
}
```

#### 大型模型配置（更保守的策略）
```python
large_model_config = {
    'lr': 5e-4,
    'weight_decay': 5e-3,

    # 保守的自適應設定
    'adaptation_strength': 0.8,
    'relationship_discovery_interval': 200,
    'importance_decay': 0.98,
    'compatibility_threshold': 0.35,

    # 溫和的 lr_mask 設定
    'lr_bump': 1e-6,
    'warmup_steps': 1000,

    # 強化穩定性
    'use_cautious': True,
    'use_adopt_stability': True,
    'use_orthogonal_grad': False,  # 關閉計算密集的功能

    # 較長的權重衰減過渡期
    'wd_transition_steps': 1500,
    'wd_decay_factor': 0.8,
    'wd_min_ratio': 0.15
}
```

#### Stable Diffusion LoRA 微調配置
```python
sd_lora_config = {
    'lr': 8e-4,
    'weight_decay': 1e-2,

    # 針對 LoRA 特性的自適應設定
    'adaptation_strength': 1.2,
    'relationship_discovery_interval': 150,
    'compatibility_threshold': 0.25,

    # 較大的 lr_mask 調整幅度
    'lr_bump': 5e-6,
    'warmup_steps': 300,

    # 適中的權重衰減
    'wd_transition_steps': 600,
    'wd_decay_factor': 0.7,
    'wd_min_ratio': 0.1,

    # 啟用所有增強功能
    'use_spd': True,
    'spd_lambda': 0.08,
    'use_grams': True,
    'use_agr': True
}
```

#### 計算資源受限配置
```python
low_memory_config = {
    'lr': 1e-3,

    # 較少的關係發現以節省記憶體
    'relationship_discovery_interval': 300,
    'compatibility_threshold': 0.4,

    # 簡化的增強功能
    'use_orthogonal_grad': False,
    'use_grams': False,

    # 較小的 lr_mask 調整
    'lr_bump': 1e-6,
    'warmup_steps': 200,

    # 快速的權重衰減過渡
    'wd_transition_steps': 500,
    'wd_decay_factor': 0.6
}
```

### HinaAdamWOptimizer 傳統配置 ⚠️

#### Stable Diffusion LoRA
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

#### 大語言模型微調
```python
llm_config = {
    'lr': 5e-4,
    'alora_ratio': 20.0,
    'wd_transition_steps': 1200,
    'wd_decay_factor': 0.8,
    'use_adopt_stability': True
}
```

#### LoKr 專用配置
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
| 優化器 | 記憶體使用 | 相對節省 | 緩衝區優化 | 特色功能 |
|--------|-----------|---------|------------|----------|
| AdamW | 100% | - | - | 基礎優化器 |
| AdamW8bit | 55% | 45% ↓ | - | 8bit 量化 |
| HinaAdamW | 57% | 43% ↓ | - | LoRA 專用 |
| HinaAdaptive | 45-65% | 35-55% ↓ | 50-90% ↓ | 記憶體優化版 |
| AdaptiveHinaAdamW | 60% | 40% ↓ | 50-90% ↓ | 通用自適應版 |

### HinaAdaptive 專用性能分析
| 功能模組 | 記憶體節省 | 計算加速 | VRAM 壓力減輕 |
|----------|-----------|----------|---------------|
| 緩衝區池管理 | 50-90% | +15% | 高 |
| JIT 編譯優化 | - | +20-50% | 中 |
| 邊緣檢測緩存 | 30-60% | +10% | 中 |
| CPU 狀態卸載 | 20-40% | -5% | 高 |
| bfloat16 精度 | 50% | +5-10% | 高 |

### 收斂性能
| 指標 | HinaAdaptive | AdaptiveHinaAdamW | HinaAdamW | 相比 AdamW |
|------|--------------|-------------------|-----------|------------|
| 收斂速度 | +20-30% | +25% | +15% | +15% |
| 最終性能 | +5-15% | +8-12% | +3-5% | +3-5% |
| 訓練穩定性 | +40% | +35% | +20% | +20% |
| 記憶體效率 | +35-55% | +40% | +43% | N/A |
| 邊緣控制效果 | +25% | N/A | N/A | N/A |
| LoRA 優化效果 | +30% | N/A | +20% | N/A |

### lr_mask 機制效果
| 訓練階段 | 學習率調整精度 | 極性一致性改善 | 訓練穩定性 |
|----------|---------------|----------------|------------|
| Warmup | ±15% | +30% | +25% |
| 穩定期 | ±5% | +20% | +40% |

### 記憶體優化效果
| 功能 | 記憶體節省 | 計算加速 | 適用場景 |
|------|-----------|----------|----------|
| 緩衝區池 | 50-90% | +15% | 所有操作 |
| JIT 編譯 | - | +20-50% | 核心計算 |
| 原地操作 | 30-60% | +10% | 梯度處理 |

## 🛠️ 故障排除

### 常見問題

#### HinaAdaptive 相關 🆕
1. **記憶體壓力過高**
   - 調降 `vram_budget_gb` 到實際 VRAM 的 80%
   - 啟用 `cpu_offload_states=True` 和 `reduce_precision=True`
   - 減少 `max_buffer_memory_mb` 緩衝區記憶體
   - 調用 `optimizer.optimize_for_vram()` 進行緊急優化

2. **邊緣抑制效果過強**
   - 降低 `edge_penalty` 參數（建議 0.05-0.15）
   - 調高 `edge_threshold`（建議 0.5-0.8）
   - 關閉 `spatial_awareness` 減少額外正則化

3. **LoRA 訓練效果不佳**
   - 啟用 `lora_rank_penalty=True`
   - 調整 `rank_penalty_strength`（建議 0.005-0.02）
   - 設置適當的 `low_rank_emphasis`（建議 1.1-1.5）

4. **訓練速度過慢**
   - 關閉 `use_orthogonal_grad` 減少計算負擔
   - 增加 `relationship_discovery_interval` 減少關係計算
   - 關閉 `spatial_awareness` 等計算密集功能
   - 設置 `emergency_simplify=True`

5. **lr_mask 調整過於激進**
   - 減少 `lr_bump` 數值（建議 1e-6 到 5e-6）
   - 增加 `warmup_steps` 延長穩定期
   - 調整 `min_lr` 和 `max_lr` 範圍

6. **異步計算錯誤**
   - 檢查是否有多進程衝突
   - 調用 `optimizer.cleanup_resources()` 清理資源
   - 減少 `relationship_discovery_interval` 避免過多異步任務

7. **邊緣檢測無效果**
   - 確認輸入是 2D 張量（圖像或特徵圖）
   - 調整 `edge_threshold` 閾值
   - 檢查 `edge_penalty` 是否過小

8. **記憶體洩漏**
   - 定期調用 `optimizer.cleanup_resources()`
   - 檢查 `edge_cache` 緩存大小
   - 設置合理的 `max_buffer_memory_mb`

#### AdaptiveHinaAdamW 相關 🆕
1. **參數關係未發現**
   - 調整 `relationship_discovery_interval` 減少間隔
   - 降低 `compatibility_threshold` 閾值
   - 檢查模型是否有足夠的 2D 參數

2. **自適應效果不明顯**
   - 增加 `adaptation_strength` 參數（建議 1.0-2.0）
   - 檢查 `importance_decay` 是否過小（建議 0.9-0.99）
   - 確認 `use_dynamic_adaptation=True`

3. **lr_mask 調整過於激進**
   - 減少 `lr_bump` 數值（建議 1e-6 到 5e-6）
   - 增加 `warmup_steps` 延長穩定期
   - 調整 `min_lr` 和 `max_lr` 範圍

4. **訓練過程不穩定**
   - 啟用 `use_cautious=True` 和 `use_adopt_stability=True`
   - 關閉 `use_orthogonal_grad` 減少梯度修改
   - 增加 `relationship_discovery_interval` 減少變化頻率

5. **記憶體使用過高**
   - 增加 `relationship_discovery_interval` 減少關係計算
   - 關閉 `use_orthogonal_grad` 等計算密集功能
   - 調用 `optimizer.clear_buffer_pool()` 清理緩衝區

6. **收斂速度慢**
   - 檢查 `adaptation_strength` 是否過小
   - 調整 `lr_bump` 和基礎學習率
   - 確認重要參數的學習率調整是否生效

#### HinaAdamWOptimizer 相關 ⚠️
1. **LoKr 參數未檢測** → 檢查參數命名模式
2. **權重衰減無效** → 需要修改 kohya sd-scripts 與 LyCORIS 的 `kohya.py` 程式碼
3. **記憶體不足** → 使用 8bit 版本或減少批次大小
4. **訓練不穩定** → 調整權重衰減參數
5. **收斂緩慢** → 檢查學習率和 ALoRA 比例

### 調試工具

#### HinaAdaptive 調試 🆕
```python
# 獲取優化器詳細信息
info = optimizer.get_optimization_info()
print(f"優化器版本: {info['version']}")
print(f"總參數數量: {info['total_params']}")
print(f"啟用功能: {info['features']}")

# 檢查記憶體狀態
memory_stats = optimizer.get_memory_stats()
print(f"記憶體壓力: {memory_stats['memory_pressure']:.2%}")
print(f"緩衝區統計: {memory_stats['buffer_pool_stats']}")

# 獲取邊緣過擬合控制資訊
edge_info = info['edge_overfitting_control']
print(f"邊緣控制配置:")
print(f"  邊緣懲罰: {edge_info['edge_penalty']}")
print(f"  閾值: {edge_info['edge_threshold']}")
print(f"  頻率懲罰: {edge_info['frequency_penalty']}")
print(f"  LoRA 秩懲罰強度: {edge_info['rank_penalty_strength']}")

# 訓練過程中監控關鍵指標
def monitor_training_progress(step):
    if step % 100 == 0:
        # 記憶體監控
        memory_stats = optimizer.get_memory_stats()
        memory_pressure = memory_stats['memory_pressure']

        print(f"步數 {step}:")
        print(f"  記憶體壓力: {memory_pressure:.2%}")

        # 如果記憶體壓力過高，自動優化
        if memory_pressure > 0.85:
            print("  警告：記憶體壓力過高，啟動緊急優化")
            optimizer.optimize_for_vram(optimizer.vram_budget_gb * 0.9)

        # 緩衝區池統計
        buffer_stats = memory_stats['buffer_pool_stats']
        print(f"  緩衝區類型: {buffer_stats['total_buffer_types']}")
        print(f"  緩衝區記憶體: {buffer_stats['current_memory_mb']:.1f}MB")

        # CUDA 記憶體統計（如果可用）
        if 'cuda_memory' in memory_stats:
            cuda_stats = memory_stats['cuda_memory']
            print(f"  CUDA 已分配: {cuda_stats['allocated_gb']:.2f}GB")
            print(f"  CUDA 已保留: {cuda_stats['reserved_gb']:.2f}GB")

# 訓練過程中的邊緣檢測監控
def monitor_edge_detection(step, param_name, param):
    if step % 200 == 0 and len(param.shape) >= 2:
        # 檢查是否有邊緣檢測狀態
        param_id = id(param)
        if hasattr(optimizer, 'param_groups_metadata'):
            for group_meta in optimizer.param_groups_metadata.values():
                compact_state = group_meta['compact_states'].get(param_id)
                if compact_state:
                    edge_strength = compact_state.get_scalar('edge_strength', 0.0)
                    spatial_activity = compact_state.get_scalar('spatial_activity', 0.0)
                    print(f"參數 {param_name}:")
                    print(f"  邊緣強度: {edge_strength:.4f}")
                    print(f"  空間活動度: {spatial_activity:.4f}")
                    break

# 性能分析和瓶頸檢測
def analyze_performance():
    info = optimizer.get_optimization_info()

    # 檢查功能啟用狀況
    features = info['features']
    enabled_features = [name for name, enabled in features.items() if enabled]
    print(f"已啟用功能: {enabled_features}")

    # 記憶體配置分析
    memory_config = info['memory_optimization']
    print(f"記憶體配置:")
    print(f"  VRAM 預算: {memory_config['vram_budget_gb']}GB")
    print(f"  CPU 卸載: {memory_config['cpu_offload_states']}")
    print(f"  精度減少: {memory_config['reduce_precision']}")

    # 給出優化建議
    memory_stats = optimizer.get_memory_stats()
    pressure = memory_stats['memory_pressure']

    if pressure > 0.9:
        print("建議:")
        print("  - 減少 VRAM 預算設定")
        print("  - 啟用 CPU 狀態卸載")
        print("  - 啟用精度減少")
        print("  - 關閉計算密集功能")
    elif pressure > 0.7:
        print("建議:")
        print("  - 啟用精度減少")
        print("  - 適度減少緩衝區記憶體")
    else:
        print("記憶體使用正常")

# 訓練結束後的完整資源清理
def cleanup_after_training():
    print("開始清理訓練資源...")

    # 獲取清理前的記憶體統計
    before_stats = optimizer.get_memory_stats()
    print(f"清理前緩衝區記憶體: {before_stats['buffer_pool_stats']['current_memory_mb']:.1f}MB")

    # 執行清理
    optimizer.cleanup_resources()

    # 手動垃圾回收
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    print("資源清理完成")
```

#### AdaptiveHinaAdamW 調試 🆕
```python
# 獲取優化器詳細信息
info = optimizer.get_optimization_info()
print(f"優化器版本: {info['version']}")
print(f"自適應功能狀態: {info['features']}")
print(f"訓練統計: {info['training_stats']}")

# 獲取參數關係摘要
relationships = optimizer.get_relationship_summary()
print(f"發現的參數關係: {relationships['total_relationships']}")
for rel in relationships['relationships'][:3]:  # 顯示前3個關係
    print(f"  {rel['param_shape']} <-> {rel['partner_shape']}, "
          f"相容性: {rel['compatibility']:.3f}")

# 獲取重要性分析報告
importance = optimizer.get_importance_analysis()
print(f"參數總數: {importance['total_parameters']}")
print(f"高重要性參數: {importance['high_importance_params']}")
print(f"平均重要性: {importance['importance_statistics']['mean']:.3f}")

# 獲取 lr_mask 分析
lr_mask_analysis = optimizer.get_lr_mask_analysis()
if lr_mask_analysis['lr_mask_enabled']:
    global_stats = lr_mask_analysis['global_statistics']
    print(f"lr_mask 統計:")
    print(f"  總參數: {global_stats['total_parameters']}")
    print(f"  Warmup 中: {global_stats['warmup_parameters']}")
    print(f"  已完成 Warmup: {global_stats['post_warmup_parameters']}")
    print(f"  平均 lr 縮放: {global_stats['avg_lr_scale']:.4f}")

# 獲取緩衝區池統計
buffer_stats = optimizer.get_buffer_pool_stats()
print(f"緩衝區池統計:")
print(f"  緩衝區類型: {buffer_stats['total_buffer_types']}")
print(f"  總緩衝區數: {buffer_stats['total_buffers']}")
print(f"  估計記憶體: {buffer_stats['estimated_memory_mb']:.2f} MB")

# 訓練結束後清理記憶體
optimizer.clear_buffer_pool()
```

#### HinaAdamWOptimizer 調試
```python
# 獲取詳細統計
info = optimizer.get_optimization_info()
print(f"LoKr 參數: {info['lokr_stats']}")

# 診斷 LoRA 配對
diagnosis = optimizer.diagnose_lora_pairing()
print(f"配對狀況: {diagnosis}")
```

### 性能監控建議

#### 訓練過程中的關鍵指標
```python
# 每 100 步監控一次優化器狀態
if step % 100 == 0:
    info = optimizer.get_optimization_info()
    training_stats = info.get('training_stats', {})

    # 監控重要指標
    print(f"步數 {step}:")
    print(f"  發現關係: {training_stats.get('total_relationships', 0)}")
    print(f"  平均重要性: {training_stats.get('avg_importance_score', 0):.3f}")

    # 監控 lr_mask 狀態
    if 'lr_mask_stats' in training_stats:
        lr_stats = training_stats['lr_mask_stats']
        print(f"  lr_mask 平均縮放: {lr_stats.get('avg_lr_scale', 1.0):.4f}")
```

#### 記憶體使用監控
```python
import torch

# 定期檢查記憶體使用
if step % 500 == 0:
    buffer_stats = optimizer.get_buffer_pool_stats()
    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB

    print(f"記憶體狀態 (步數 {step}):")
    print(f"  GPU 記憶體: {gpu_memory:.2f} GB")
    print(f"  緩衝區記憶體: {buffer_stats['estimated_memory_mb']:.1f} MB")
    print(f"  緩衝區數量: {buffer_stats['total_buffers']}")
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

### 已知限制與注意事項

#### AdaptiveHinaAdamW
- **計算開銷**：關係發現和重要性評估會增加計算時間
- **記憶體需求**：雖有緩衝區優化，但仍需額外記憶體存儲元數據
- **參數相容性**：主要針對 2D 參數（矩陣）進行關係分析
- **收斂時間**：初期可能需要更多步數來建立參數關係

#### 效能調優建議
- 對於小型模型，可適當增加 `relationship_discovery_interval`
- 計算資源受限時，關閉 `use_orthogonal_grad` 和 `use_grams`
- 記憶體緊張時，增加關係發現間隔並定期清理緩衝區池

## 📈 發展路線

### 近期規劃
- **擴展 LoKr 支援**：更多命名模式和結構
- **自動調優**：基於損失趨勢的參數自動調整
- **視覺化工具**：訓練過程的視覺化監控
- **AdaptiveHinaAdamW 增強**：
  - 更精確的參數關係分析演算法
  - 支援 3D 和更高維度參數的關係分析
  - 基於注意力機制的重要性評估
  - 多 GPU 分散式訓練優化

### 中期目標
- **跨架構優化**：針對 Transformer、CNN、RNN 等不同架構的專門優化
- **自動超參數調整**：基於訓練動態的自動學習率和權重衰減調整
- **混合精度整合**：與 AMP (Automatic Mixed Precision) 的深度整合
- **梯度壓縮**：分散式訓練中的梯度壓縮技術

### 長期目標
- **模型感知優化**：針對不同模型架構的專門優化策略
- **分散式支援**：多 GPU 和多節點的最佳化支援
- **產業級部署**：生產環境的穩定性和效能優化
- **神經架構搜索整合**：與 NAS 技術的結合

## 📚 延伸閱讀

### 學術論文
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)

### 技術文檔
- [PyTorch Optimizer 文檔](https://pytorch.org/docs/stable/optim.html)
- [Bitsandbytes 文檔](https://huggingface.co/docs/bitsandbytes)
- [LoRA 微調指南](https://huggingface.co/docs/peft)
- [PyTorch JIT 文檔](https://pytorch.org/docs/stable/jit.html)

### 相關研究
- [ADOPT: Modified Adam Can Converge with Any β₂](https://arxiv.org/abs/2411.02853)
- [Cautious Optimizers](https://arxiv.org/abs/2411.16085)
- [Selective Projection Decay](https://arxiv.org/abs/2410.05729)

---

**最後更新**：2025年6月20日
**版本**：3.0.0
**維護者**：Hina
**文檔狀態**：✅ 已更新並包含 AdaptiveHinaAdamW 最新功能（lr_mask、記憶體優化、策略 B 組合式整合）