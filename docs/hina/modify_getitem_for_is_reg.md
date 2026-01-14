# 修改 __getitem__ 以支持直接訪問 is_reg

## 修改方案

需要在 `library/train_util.py` 的 `BaseDataset.__getitem__` 方法中添加 `is_reg` 字段到返回的 batch 中。

---

## 修改步驟

### 步驟 1：在循環開始前添加 is_reg_list

在 ```1571:1582:library/train_util.py``` 處，添加 `is_reg_list`：

**修改前**：
```python
        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs_list = []
        custom_attributes = []
```

**修改後**：
```python
        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs_list = []
        custom_attributes = []
        is_reg_list = []  # 新增：收集 is_reg 信息
```

---

### 步驟 2：在循環中收集 is_reg 值

在 ```1584:1591:library/train_util.py``` 的循環中，添加 `is_reg` 的收集：

**修改前**：
```python
        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
```

**修改後**：
```python
        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
            is_reg_list.append(image_info.is_reg)  # 新增：收集 is_reg 值
```

---

### 步驟 3：在返回的 example 中添加 is_reg 字段

在 ```1792:1835:library/train_util.py``` 處，在構建 `example` 字典時添加 `is_reg` 字段：

**修改前**：
```python
        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)
        # ... 其他字段 ...
        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example
```

**修改後**：
```python
        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["is_reg"] = torch.BoolTensor(is_reg_list)  # 新增：is_reg 字段
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)
        # ... 其他字段 ...
        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example
```

---

## 完整修改代碼

### 修改位置 1：添加 is_reg_list 初始化

**文件**: `library/train_util.py`
**行號**: 約 1571-1582

```python
        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs_list = []
        custom_attributes = []
        is_reg_list = []  # 新增
```

### 修改位置 2：在循環中收集 is_reg

**文件**: `library/train_util.py`
**行號**: 約 1584-1591

```python
        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
            is_reg_list.append(image_info.is_reg)  # 新增
```

### 修改位置 3：在返回字典中添加 is_reg 字段

**文件**: `library/train_util.py`
**行號**: 約 1793-1795

```python
        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["is_reg"] = torch.BoolTensor(is_reg_list)  # 新增
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)
```

---

## 使用示例

修改後，在 `process_batch` 中可以直接訪問 `is_reg`：

```python
def process_batch(
    self,
    batch,
    text_encoders,
    unet,
    network,
    vae,
    noise_scheduler,
    vae_dtype,
    weight_dtype,
    accelerator,
    args,
    text_encoding_strategy: strategy_base.TextEncodingStrategy,
    tokenize_strategy: strategy_base.TokenizeStrategy,
    is_train=True,
    train_text_encoder=True,
    train_unet=True,
    step=None,
    global_step=None,
) -> torch.Tensor:
    """
    Process a batch for the network
    """
    # 直接訪問 is_reg
    is_reg = batch["is_reg"]  # shape: (batch_size,), dtype: torch.bool

    # 使用示例 1: 分別處理正則化圖像和訓練圖像
    if is_reg.any():
        reg_indices = is_reg.nonzero(as_tuple=True)[0]
        reg_loss = loss[reg_indices]
        # 對正則化圖像進行特殊處理

    if (~is_reg).any():
        train_indices = (~is_reg).nonzero(as_tuple=True)[0]
        train_loss = loss[train_indices]
        # 對訓練圖像進行處理

    # 使用示例 2: 創建掩碼
    is_reg_mask = is_reg.float()  # 轉換為 float 用於乘法
    is_train_mask = (~is_reg).float()

    # 使用示例 3: 統計信息
    num_reg_samples = is_reg.sum().item()
    num_train_samples = (~is_reg).sum().item()

    # ... 原有的損失計算邏輯 ...
    loss_weights = batch["loss_weights"]
    loss = loss * loss_weights

    return loss.mean()
```

---

## 數據類型選擇

### 選項 1：使用 `torch.BoolTensor`（推薦）

```python
example["is_reg"] = torch.BoolTensor(is_reg_list)
```

**優點**：
- 內存效率高（每個值 1 字節）
- 語義清晰（布爾值）
- 可以直接用於布爾運算（`is_reg.any()`, `is_reg.all()` 等）

**使用**：
```python
is_reg = batch["is_reg"]  # dtype: torch.bool
if is_reg.any():
    # 處理正則化圖像
```

### 選項 2：使用 `torch.FloatTensor`

```python
example["is_reg"] = torch.FloatTensor([1.0 if reg else 0.0 for reg in is_reg_list])
```

**優點**：
- 可以直接用於數學運算（例如與損失相乘）
- 與 `loss_weights` 類型一致

**使用**：
```python
is_reg = batch["is_reg"]  # dtype: torch.float32
reg_mask = is_reg  # 可以直接用於乘法
```

### 選項 3：使用 `torch.LongTensor`

```python
example["is_reg"] = torch.LongTensor([1 if reg else 0 for reg in is_reg_list])
```

**優點**：
- 整數類型，可以用於索引

**使用**：
```python
is_reg = batch["is_reg"]  # dtype: torch.int64
reg_indices = (is_reg == 1).nonzero(as_tuple=True)[0]
```

---

## 推薦方案

**推薦使用 `torch.BoolTensor`**，因為：
1. 語義最清晰
2. 內存效率最高
3. 可以直接用於布爾運算
4. 如果需要轉換為其他類型，可以輕鬆轉換：
   ```python
   is_reg_float = is_reg.float()  # 轉換為 float
   is_reg_long = is_reg.long()     # 轉換為 long
   ```

---

## 注意事項

1. **向後兼容性**：
   - 如果其他代碼依賴於 batch 的結構，需要確保不會破壞現有功能
   - 建議在添加新字段時，確保不會影響現有的 `process_batch` 實現

2. **性能影響**：
   - 添加 `is_reg` 字段的開銷很小（只是多了一個布爾值列表）
   - 不會對訓練性能造成明顯影響

3. **測試**：
   - 修改後需要測試確保：
     - `is_reg` 字段正確設置
     - 不會影響現有的訓練流程
     - 在 `process_batch` 中可以正確訪問

---

## 完整修改對比

### 修改前

```python
def __getitem__(self, index):
    # ...
    loss_weights = []
    # ... 其他列表 ...
    custom_attributes = []

    for image_key in bucket[image_index : image_index + bucket_batch_size]:
        image_info = self.image_data[image_key]
        subset = self.image_to_subset[image_key]

        custom_attributes.append(subset.custom_attributes)
        loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
        # ...

    example = {}
    example["loss_weights"] = torch.FloatTensor(loss_weights)
    # ... 其他字段 ...
    return example
```

### 修改後

```python
def __getitem__(self, index):
    # ...
    loss_weights = []
    # ... 其他列表 ...
    custom_attributes = []
    is_reg_list = []  # 新增

    for image_key in bucket[image_index : image_index + bucket_batch_size]:
        image_info = self.image_data[image_key]
        subset = self.image_to_subset[image_key]

        custom_attributes.append(subset.custom_attributes)
        loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
        is_reg_list.append(image_info.is_reg)  # 新增
        # ...

    example = {}
    example["loss_weights"] = torch.FloatTensor(loss_weights)
    example["is_reg"] = torch.BoolTensor(is_reg_list)  # 新增
    # ... 其他字段 ...
    return example
```

---

## 驗證修改

修改後，可以通過以下方式驗證：

```python
# 在 process_batch 中測試
def process_batch(self, batch, ...):
    # 驗證 is_reg 字段存在
    assert "is_reg" in batch, "is_reg field not found in batch"

    is_reg = batch["is_reg"]
    assert is_reg.dtype == torch.bool, f"Expected bool, got {is_reg.dtype}"
    assert len(is_reg) == len(batch["loss_weights"]), "Length mismatch"

    # 驗證 is_reg 與 loss_weights 的一致性
    prior_loss_weight = args.prior_loss_weight
    if prior_loss_weight != 1.0:
        expected_reg_mask = (batch["loss_weights"] == prior_loss_weight)
        assert torch.equal(is_reg, expected_reg_mask), "is_reg and loss_weights mismatch"

    # ... 繼續處理 ...
```

---

## 總結

通過以上三個簡單的修改步驟，就可以在 `__getitem__` 中添加 `is_reg` 字段，使得 `process_batch` 可以直接訪問該信息，無需通過 `loss_weights` 間接推斷。
