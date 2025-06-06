# Custom AdamW Optimizer - 增強型優化器

## 概述

Custom AdamW Optimizer 是一個基於 `bitsandbytes.optim.AdamW8bit` 的增強型優化器，整合了多種先進的優化技術，專為改善深度學習模型的訓練效果而設計，特別針對 LoRA (Low-Rank Adaptation) 微調進行了優化。

## 主要功能

### 1. 泛化增強技術

#### Selective Projection Decay (SPD)
- **目的**：避免模型在微調時過度偏離預訓練初始化，提高內分佈泛化能力和外分佈穩健性
- **原理**：對選定層施加選擇性懲罰，正則化模型相對於原始參數的偏差
- **參數**：`use_spd`, `spd_lambda`

#### Cautious Optimizer
- **目的**：只在更新方向與當前梯度對齊時才應用更新，提高訓練穩定性
- **原理**：計算對齊遮罩，確保負更新與梯度間具有非負內積
- **參數**：`use_cautious`

#### Orthogonal Gradient (⊥Grad)
- **目的**：減輕神經崩潰和數值誤差，改善泛化能力
- **原理**：將梯度投影為與權重向量正交
- **參數**：`use_orthogonal_grad`

### 2. 自適應學習率技術

#### ADOPT 穩定性
- **目的**：提供更穩定的訓練，特別是在小批量或高梯度噪音情況下
- **原理**：修改動量更新和二階矩估計的順序，從二階矩估計中移除當前梯度
- **參數**：`use_adopt_stability`

#### Grams 自適應動量縮放
- **目的**：分離參數更新的方向和幅度，實現更快的收斂
- **原理**：更新方向來自當前梯度，動量僅用於自適應幅度縮放
- **參數**：`use_grams`

#### Adaptive Gradient Regularization (AGR)
- **目的**：平滑損失曲面，提高訓練效率和泛化性能
- **原理**：使用梯度向量總和歸一化作為係數，自適應控制梯度下降方向
- **參數**：`use_agr`

#### Torque-Aware Momentum (TAM)
- **目的**：穩定訓練期間的更新方向，增強探索能力
- **原理**：基於新梯度與先前動量間的角度引入阻尼因子
- **參數**：`use_tam`, `tam_beta`

### 3. LoRA 專屬優化

#### ALoRA 風格自適應學習率
- **目的**：解決 LoRA 的主要缺陷，快速擺脫零點並精確發現最佳方向
- **原理**：採用與低秩適應矩陣 BA 的行向量 L2 範數成反比的自適應學習率
- **參數**：`use_alora`, `alora_ratio`

#### 動態權重衰減
- **目的**：根據訓練進度和參數類型動態調整權重衰減
- **原理**：對 LoRA 參數在訓練後期可能減少權重衰減
- **參數**：`dynamic_weight_decay`

### 4. LoKr 專屬優化

#### Kronecker 積感知的學習率縮放
- **目的**：針對 LoKr 的 Kronecker 積結構提供專門的學習率優化
- **原理**：計算各個子矩陣乘積的範數，使用更溫和的縮放策略避免 Kronecker 積的範數放大效應
- **特色**：自動檢測 LoKr 參數並建立配對關係

#### LoKr 動態權重衰減
- **目的**：為 LoKr 參數提供更保守的權重衰減策略
- **原理**：使用比 LoRA 更溫和的衰減曲線和更高的最小權重衰減比例
- **優化**：針對不同類型的 LoKr 參數（w1_a, w1_b, w2_a, w2_b）應用層次化學習率

#### 智能參數檢測
- **支援命名**：`lokr_w1_a`, `lokr_w1_b`, `lokr_w2_a`, `lokr_w2_b`, `lokr_w1`, `lokr_w2` 等多種模式
- **自動分組**：建立 LoKr 參數的配對關係和組別結構
- **統計監控**：提供詳細的 LoKr 參數統計信息

## 使用方法

### 基本使用

```python
from library.custom_adamw_optimizer import CustomAdamWOptimizer

# 創建優化器
optimizer = CustomAdamWOptimizer(
    model.parameters(),
    lr=1e-3,
    use_spd=True,           # 啟用 SPD
    use_cautious=True,      # 啟用謹慎優化器
    use_adopt_stability=True, # 啟用 ADOPT 穩定性
    use_grams=True,         # 啟用 Grams
    use_agr=True,           # 啟用 AGR
    use_tam=True,           # 啟用 TAM
    use_alora=True,         # 啟用 ALoRA
    dynamic_weight_decay=True # 啟用動態權重衰減
)
```

### 在訓練腳本中使用

```bash
python train_network.py \
    --optimizer_type CustomAdamW \
    --learning_rate 1e-3 \
    --optimizer_args \
        "use_spd=True" \
        "spd_lambda=0.1" \
        "use_cautious=True" \
        "use_adopt_stability=True" \
        "use_grams=True" \
        "use_agr=True" \
        "use_tam=True" \
        "tam_beta=0.999" \
        "use_alora=True" \
        "alora_ratio=21.0" \
        "dynamic_weight_decay=True" \
    # 其他訓練參數...
```

### 便利函數

```python
from library.custom_adamw_optimizer import get_custom_adamw_optimizer

optimizer = get_custom_adamw_optimizer(
    model.parameters(),
    lr=1e-3,
    optimizer_kwargs={
        'use_spd': True,
        'use_alora': True,
        # 其他參數...
    }
)
```

## 參數配置

### 完整參數列表

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `use_spd` | bool | True | 啟用 SPD 正則化 |
| `spd_lambda` | float | 0.1 | SPD 懲罰強度 |
| `use_cautious` | bool | True | 啟用謹慎優化器機制 |
| `use_orthogonal_grad` | bool | False | 啟用正交梯度投影 |
| `use_adopt_stability` | bool | True | 啟用 ADOPT 穩定性機制 |
| `use_grams` | bool | True | 啟用 Grams 自適應動量縮放 |
| `use_agr` | bool | True | 啟用自適應梯度正則化 |
| `use_tam` | bool | True | 啟用 Torque-Aware Momentum |
| `tam_beta` | float | 0.999 | TAM 的 beta 參數 |
| `use_alora` | bool | True | 啟用 ALoRA 風格學習率 |
| `alora_ratio` | float | 21.0 | ALoRA 學習率比例 (ηB/ηA) |
| `dynamic_weight_decay` | bool | True | 啟用動態權重衰減 |

### 推薦配置

#### 保守配置（穩定性優先）
```python
conservative_config = {
    'use_spd': True,
    'use_cautious': True,
    'use_adopt_stability': True,
    'use_grams': False,
    'use_agr': False,
    'use_tam': True,
    'use_alora': True,
    'dynamic_weight_decay': True
}
```

#### 激進配置（性能優先）
```python
aggressive_config = {
    'use_spd': True,
    'use_cautious': True,
    'use_orthogonal_grad': True,
    'use_adopt_stability': True,
    'use_grams': True,
    'use_agr': True,
    'use_tam': True,
    'use_alora': True,
    'dynamic_weight_decay': True
}
```

#### LoRA 專用配置
```python
lora_config = {
    'use_spd': True,
    'spd_lambda': 0.05,
    'use_cautious': True,
    'use_adopt_stability': True,
    'use_alora': True,
    'alora_ratio': 22.0,  # 針對 Llama 模型的最佳比率
    'dynamic_weight_decay': True
}
```

#### LoKr 專用配置
```python
lokr_config = {
    'use_spd': True,
    'spd_lambda': 0.08,            # 略低於 LoRA
    'use_cautious': True,
    'use_adopt_stability': True,
    'use_alora': True,             # 自動支援 LoKr
    'alora_ratio': 18.0,          # LoKr 適用範圍：14.0-20.0
    'dynamic_weight_decay': True,
    'wd_transition_steps': 600,   # 較快過渡
    'wd_decay_factor': 0.8,       # 溫和衰減
    'wd_min_ratio': 0.18,         # 較高最小權重衰減
    'use_tam': True,
    'tam_beta': 0.995             # 略低於預設值
}
```

## 監控和調試

### 獲取優化器信息

```python
opt_info = optimizer.get_optimization_info()
print(f"優化器類型: {opt_info['optimizer_type']}")
print(f"總參數數量: {opt_info['total_params']}")
print(f"啟用功能: {opt_info['features']}")
print(f"LoRA 統計: {opt_info['lora_stats']}")

# LoKr 專用統計（如果檢測到 LoKr 參數）
if 'lokr_stats' in opt_info:
    lokr_stats = opt_info['lokr_stats']
    print(f"LoKr 統計: {lokr_stats}")
    print(f"  LoKr 組別: {lokr_stats['lokr_groups']}")
    print(f"  配對關係: {lokr_stats['lokr_pairs']}")
    print(f"  W1 參數: {lokr_stats['lokr_w1_params']} + {lokr_stats['lokr_w1_a_params']}A + {lokr_stats['lokr_w1_b_params']}B")
    print(f"  W2 參數: {lokr_stats['lokr_w2_params']} + {lokr_stats['lokr_w2_a_params']}A + {lokr_stats['lokr_w2_b_params']}B")

    # 檢查 LoKr 支援狀態
    if lokr_stats['lokr_groups'] > 0:
        print("✅ LoKr 參數檢測成功，啟用專門優化策略")
    else:
        print("⚠️ 未檢測到 LoKr 參數")
```

### 日誌輸出

優化器會自動記錄以下信息：
- 啟用的功能列表
- 檢測到的 LoRA/LoKr 參數統計
- 功能特定的警告和信息
- LoKr 專用的參數檢測和配對信息

## 相關文檔

- [LoKr 支援詳細指南](./LOKR_SUPPORT_GUIDE.md) - LoKr 支援的完整實現細節和使用指南
- [動態權重衰減理論](./DYNAMIC_WEIGHT_DECAY_THEORY.md) - 動態權重衰減的理論基礎
- [LoRA 優化文檔](./README_LoRA_Optimization.md) - LoRA 專用優化的詳細說明

## 更新日誌

### v2.1.0 - LoKr 支援
- ✅ 新增 LoKr (Low-rank Kronecker product adaptation) 完整支援
- ✅ 智能 LoKr 參數檢測和分組
- ✅ Kronecker 積感知的學習率縮放
- ✅ LoKr 專屬的動態權重衰減策略
- ✅ 詳細的 LoKr 統計和監控功能
- ✅ 多種 LoKr 命名模式支援
- ✅ 向後相容，不影響現有 LoRA 功能

## 性能考量

### 記憶體使用
- 相比標準 AdamW8bit 增加約 5-10% 的記憶體使用
- SPD 功能需要存儲初始參數，會增加一倍的參數記憶體使用

### 計算開銷
- 每個功能都會增加額外的計算開銷
- 建議根據實際需求選擇性啟用功能
- 在資源受限的環境中可以使用保守配置

### 收斂速度
- 通常能提供更快的收斂和更好的最終性能
- 某些功能可能在訓練初期稍微降低收斂速度，但會在後期帶來更好的穩定性

## 故障排除

### 常見問題

1. **記憶體不足**
   - 解決方案：禁用 SPD (`use_spd=False`) 或減少批量大小

2. **訓練不穩定**
   - 解決方案：啟用 `use_cautious` 和 `use_adopt_stability`

3. **LoRA 參數未被識別**
   - 檢查參數名稱是否包含 'lora_down', 'lora_up', 'lora_A', 或 'lora_B'
   - 確保參數具有 `param_name` 屬性

4. **性能下降**
   - 嘗試不同的功能組合
   - 調整超參數，特別是 `spd_lambda` 和 `alora_ratio`

### 除錯提示

```python
# 檢查優化器狀態
for group_idx, group in enumerate(optimizer.param_groups):
    print(f"參數組 {group_idx}: {len(group['params'])} 個參數")

# 檢查 LoRA 參數配對
for group_idx, metadata in optimizer.param_groups_metadata.items():
    print(f"組 {group_idx} LoRA 配對: {len(metadata['lora_pairs'])}")
```

## 引用和致謝

這個優化器整合了以下研究的成果：

- Selective Projection Decay (SPD)
- Cautious Optimizers
- Orthogonal Gradient (⊥Grad)
- ADOPT: Modified Adam Can Converge with Any β₂ with the Optimal Rate
- Grams: Gradient Descent with Adaptive Momentum Scaling
- Adaptive Gradient Regularization (AGR)
- Torque-Aware Momentum (TAM)
- ALoRA: Allocating Low-Rank Adaptation

詳細的論文引用請參考原始文件。

## 支援和貢獻

如有問題或建議，請：
1. 查看此文檔的故障排除部分
2. 運行測試腳本 `examples/custom_optimizer_usage.py`
3. 在項目 GitHub 頁面提交 issue

歡迎貢獻改進和新功能！