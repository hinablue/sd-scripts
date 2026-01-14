# reg_timestep 功能實現總結

## 已完成的修改

已成功為 SD3、Flux 和 Hunyuan 模型添加了 `--reg_min_timestep` 和 `--reg_max_timestep` 支持。

---

## 修改文件列表

### 1. SD3 模型

#### `library/sd3_train_utils.py`
- **函數**: `get_noisy_model_input_and_timesteps`
- **修改**: 添加 `is_reg` 參數，根據 `is_reg` 使用不同的 timestep 範圍
- **行號**: 第 918-960 行

#### `sd3_train_network.py`
- **方法**: `get_noise_pred_and_target`
- **修改**: 從 batch 獲取 `is_reg` 並傳遞給 `get_noisy_model_input_and_timesteps`
- **行號**: 第 329-335 行

---

### 2. Flux 模型

#### `library/flux_train_utils.py`
- **函數**: `get_noisy_model_input_and_timesteps`
- **修改**:
  - 添加 `is_reg` 參數
  - 在 `qinglong_flux` 模式中根據 `is_reg` 使用不同的 timestep 範圍
- **行號**: 第 471-575 行

#### `flux_train_network.py`
- **方法**: `get_noise_pred_and_target`
- **修改**: 從 batch 獲取 `is_reg` 並傳遞給 `get_noisy_model_input_and_timesteps`
- **行號**: 第 327-334 行

---

### 3. Hunyuan 模型

#### `hunyuan_image_train_network.py`
- **方法**: `get_noise_pred_and_target`
- **修改**: 從 batch 獲取 `is_reg` 並傳遞給 Flux 的 `get_noisy_model_input_and_timesteps`
- **行號**: 第 536-542 行

**注意**: Hunyuan 使用 Flux 的函數，因此自動獲得支持。

---

## 實現邏輯

### SD3 實現

```python
if is_reg is not None and is_reg.any():
    # 分別處理訓練和正則化圖像
    train_mask = ~is_reg
    reg_mask = is_reg

    train_min = args.min_timestep if args.min_timestep is not None else 0
    train_max = args.max_timestep if args.max_timestep is not None else 1000
    reg_min = args.reg_min_timestep if args.reg_min_timestep is not None else train_min
    reg_max = args.reg_max_timestep if args.reg_max_timestep is not None else train_max

    # 分別計算 timesteps
    indices = torch.zeros((bsz,), dtype=torch.long, device=device)
    if train_mask.any():
        train_u = u[train_mask]
        train_indices = (train_u * (train_max - train_min) + train_min).long()
        indices[train_mask] = train_indices
    if reg_mask.any():
        reg_u = u[reg_mask]
        reg_indices = (reg_u * (reg_max - reg_min) + reg_min).long()
        indices[reg_mask] = reg_indices

    timesteps = indices.to(device=device, dtype=dtype)
else:
    # 原有邏輯
    ...
```

### Flux (qinglong_flux 模式) 實現

```python
if is_reg is not None and is_reg.any():
    # 分別處理訓練和正則化圖像
    train_mask = ~is_reg
    reg_mask = is_reg

    train_t_min = args.min_timestep if args.min_timestep is not None else 0
    train_t_max = args.max_timestep if args.max_timestep is not None else 1000.0
    reg_t_min = args.reg_min_timestep if args.reg_min_timestep is not None else train_t_min
    reg_t_max = args.reg_max_timestep if args.reg_max_timestep is not None else train_t_max

    # 轉換為 0-1 範圍
    train_t_min /= 1000.0
    train_t_max /= 1000.0
    reg_t_min /= 1000.0
    reg_t_max /= 1000.0

    # 分別應用範圍
    t_result = torch.zeros((bsz,), device=device)
    if train_mask.any():
        t_train = t[train_mask] * (train_t_max - train_t_min) + train_t_min
        t_result[train_mask] = t_train
    if reg_mask.any():
        t_reg = t[reg_mask] * (reg_t_max - reg_t_min) + reg_t_min
        t_result[reg_mask] = t_reg

    timesteps = t_result * 1000.0
    timesteps += 1
else:
    # 原有邏輯
    ...
```

---

## 使用方式

### 命令行使用

```bash
# SD3
python sd3_train_network.py \
    --train_data_dir ./train_data \
    --reg_data_dir ./reg_data \
    --min_timestep 0 \
    --max_timestep 1000 \
    --reg_min_timestep 200 \
    --reg_max_timestep 800 \
    ...

# Flux
python flux_train_network.py \
    --train_data_dir ./train_data \
    --reg_data_dir ./reg_data \
    --timestep_sampling qinglong_flux \
    --min_timestep 0 \
    --max_timestep 1000 \
    --reg_min_timestep 200 \
    --reg_max_timestep 800 \
    ...

# Hunyuan
python hunyuan_image_train_network.py \
    --train_data_dir ./train_data \
    --reg_data_dir ./reg_data \
    --timestep_sampling qinglong_flux \
    --min_timestep 0 \
    --max_timestep 1000 \
    --reg_min_timestep 200 \
    --reg_max_timestep 800 \
    ...
```

### 配置文件使用

```toml
[train]
train_data_dir = "./train_data"
reg_data_dir = "./reg_data"
min_timestep = 0
max_timestep = 1000
reg_min_timestep = 200
reg_max_timestep = 800
```

---

## 特性

### ✅ 已實現的功能

1. **SD3**: 完全支持 `is_reg` 的獨立 timestep 範圍
2. **Flux (qinglong_flux 模式)**: 支持 `is_reg` 的獨立 timestep 範圍
3. **Hunyuan**: 自動支持（使用 Flux 的函數）

### ⚠️ 注意事項

1. **Flux 其他模式**:
   - 目前只支持 `qinglong_flux` 模式
   - 其他模式（uniform, sigmoid, shift, flux_shift）沒有應用 min/max 限制
   - 如果需要支持其他模式，需要額外修改

2. **向後兼容性**:
   - 如果未提供 `--reg_min_timestep` 和 `--reg_max_timestep`，正則化圖像將使用與訓練圖像相同的 timestep 範圍
   - 如果 batch 中沒有 `is_reg` 字段，將使用原有邏輯

3. **參數驗證**:
   - 建議在使用時驗證 `reg_min_timestep >= 0` 和 `reg_max_timestep <= 1000`
   - 建議驗證 `reg_min_timestep < reg_max_timestep`

---

## 測試建議

1. **基本功能測試**:
   - 使用包含正則化圖像的數據集進行訓練
   - 驗證正則化圖像和訓練圖像使用不同的 timestep 範圍

2. **邊界情況測試**:
   - 所有樣本都是正則化圖像
   - 所有樣本都是訓練圖像
   - 未提供 `--reg_min_timestep` 和 `--reg_max_timestep`

3. **向後兼容性測試**:
   - 使用沒有 `is_reg` 字段的舊版本數據集
   - 驗證不會出現錯誤

---

## 修改統計

- **修改文件數**: 5 個
- **新增參數**: 2 個（`--reg_min_timestep`, `--reg_max_timestep`）
- **修改函數**: 3 個
- **修改方法**: 3 個
- **代碼行數**: 約 100+ 行

---

## 總結

所有修改已完成並通過 linter 檢查。SD3、Flux (qinglong_flux 模式) 和 Hunyuan 現在都支持為正則化圖像設置獨立的 timestep 範圍。這使得用戶可以更精細地控制正則化圖像的訓練過程。
