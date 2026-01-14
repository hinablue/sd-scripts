# 資料集分布分析：A 和 B 在 2000 steps 中的分布方式

## 問題設定

- **A 資料集**: 100 張圖片，`repeats=10`，`is_reg=False`
- **B 資料集**: 100 張圖片，`repeats=10`，`is_reg=True`
- **1 epoch 總共**: 2000 steps

## 答案：完全隨機混合分布

**A 和 B 的圖像是完全混合並隨機分布在 2000 steps 中的，不是交替分布，也不是按順序分布。**

---

## 詳細分析

### 1. 資料集初始化階段

#### 1.1 圖像註冊

根據 ```2151:2161:library/train_util.py```：

```python
for img_path, caption, size in zip(img_paths, captions, sizes):
    info = ImageInfo(img_path, num_repeats, caption, subset.is_reg, img_path)
    # ...
    if subset.is_reg:
        reg_infos.append((info, subset))
    else:
        self.register_image(info, subset)
```

- **A 資料集**（`is_reg=False`）: 100 張圖片直接註冊到 `self.image_data`
- **B 資料集**（`is_reg=True`）: 100 張圖片先收集到 `reg_infos`，然後在後續處理中註冊

#### 1.2 正則化圖像註冊

根據 ```2177:2191:library/train_util.py```，B 的圖像最終也會註冊到 `self.image_data` 中：

```python
while n < num_train_images:  # n < 1000
    for info, subset in reg_infos:  # 100 個 B 的圖像
        if first_loop:
            self.register_image(info, subset)
            n += info.num_repeats  # n += 10
        # ...
```

結果：
- A: 100 個圖像，每個 `num_repeats=10`
- B: 100 個圖像，每個 `num_repeats=10`
- 總共 200 個不同的圖像，但每個都會被使用 10 次

---

### 2. Bucket 組織階段

#### 2.1 圖像添加到 Bucket

根據 ```1062:1064:library/train_util.py```：

```python
for image_info in self.image_data.values():
    for _ in range(image_info.num_repeats):
        self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)
```

**關鍵點**：
- 所有圖像（A 和 B）都被添加到**同一個 bucket** 中（如果它們的分辨率相同）
- 每個圖像根據 `num_repeats` 被添加多次：
  - A: 100 張 × 10 = **1000 個條目**
  - B: 100 張 × 10 = **1000 個條目**
  - **總共 2000 個條目在同一個 bucket 中**

#### 2.2 批次索引創建

根據 ```1084:1092:library/train_util.py```：

```python
self.buckets_indices: List[BucketBatchIndex] = []
for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
    batch_count = int(math.ceil(len(bucket) / self.batch_size))
    for batch_index in range(batch_count):
        self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

self.shuffle_buckets()
```

假設 `batch_size=1`：
- bucket 中有 2000 個圖像條目
- 創建 2000 個 `buckets_indices`
- 每個索引指向 bucket 中的一個位置

---

### 3. 隨機打亂階段

#### 3.1 Shuffle 操作

根據 ```1094:1099:library/train_util.py```：

```python
def shuffle_buckets(self):
    # set random seed for this epoch
    random.seed(self.seed + self.current_epoch)

    random.shuffle(self.buckets_indices)  # 打亂批次索引順序
    self.bucket_manager.shuffle()         # 打亂 bucket 內的圖像順序
```

根據 ```252:254:library/train_util.py```：

```python
def shuffle(self):
    for bucket in self.buckets:
        random.shuffle(bucket)  # 打亂每個 bucket 內的圖像順序
```

**關鍵點**：
1. `buckets_indices` 被**完全隨機打亂**
2. bucket 內的圖像順序也被**完全隨機打亂**
3. 這意味著 A 和 B 的圖像條目**完全混合在一起**

---

### 4. 訓練循環階段

#### 4.1 批次獲取

根據 ```1549:train_network.py```：

```python
for step, batch in enumerate(skipped_dataloader or train_dataloader):
    # 訓練邏輯
```

根據 ```1563:1584:library/train_util.py```：

```python
def __getitem__(self, index):
    bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
    bucket_batch_size = self.buckets_indices[index].bucket_batch_size
    image_index = self.buckets_indices[index].batch_index * bucket_batch_size

    # ...
    for image_key in bucket[image_index : image_index + bucket_batch_size]:
        image_info = self.image_data[image_key]
        subset = self.image_to_subset[image_key]
        # ...
        loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
```

**流程**：
1. 根據打亂後的 `buckets_indices[index]` 獲取批次
2. 從 bucket 中按順序取圖像（但 bucket 內的順序已經被打亂）
3. 每個圖像根據 `is_reg` 標誌設置不同的 `loss_weights`

---

## 分布方式示意圖

### 打亂前的狀態（概念上）

```
Bucket (2000 個條目):
[A1, A1, A1, ..., A1 (10次), A2, A2, ..., A2 (10次), ..., A100 (10次),
 B1, B1, B1, ..., B1 (10次), B2, B2, ..., B2 (10次), ..., B100 (10次)]
```

### 打亂後的狀態（實際）

```
Bucket (2000 個條目，完全隨機順序):
[B47, A12, A89, B3, A1, B100, A56, B23, A78, B5, ...]
 ↑     ↑    ↑    ↑   ↑    ↑     ↑    ↑    ↑    ↑
 完全隨機混合，無法預測順序
```

### 訓練時的分布（2000 steps）

```
Step 1:  [B47]  (B 圖像)
Step 2:  [A12]  (A 圖像)
Step 3:  [A89]  (A 圖像)
Step 4:  [B3]   (B 圖像)
Step 5:  [A1]   (A 圖像)
Step 6:  [B100] (B 圖像)
Step 7:  [A56]  (A 圖像)
Step 8:  [B23]  (B 圖像)
...
Step 2000: [隨機的 A 或 B]
```

---

## 統計特性

### 在 2000 steps 中：

1. **A 圖像總數**: 1000 次（100 張 × 10 repeats）
2. **B 圖像總數**: 1000 次（100 張 × 10 repeats）
3. **分布方式**: **完全隨機混合**
4. **平均分布**: 長期來看，A 和 B 各佔 50%，但**不是嚴格交替**

### 可能的分布模式

- ❌ **不是**: 奇數 step 訓練 A，偶數 step 訓練 B
- ❌ **不是**: 前 1000 steps 訓練 A，後 1000 steps 訓練 B
- ❌ **不是**: 每 10 個 A 後跟 10 個 B
- ✅ **是**: 完全隨機混合，每個 step 都有 50% 機率是 A 或 B（長期平均）

### 實際分布示例

在實際訓練中，可能會出現：
- 連續多個 A（例如：Step 10-15 都是 A）
- 連續多個 B（例如：Step 100-105 都是 B）
- A 和 B 交替出現
- 任何其他隨機模式

但**長期平均**會趨向於 50% A 和 50% B。

---

## 批次大小 > 1 的情況

如果 `batch_size > 1`（例如 `batch_size=4`）：

- 每個批次包含 4 個圖像
- 這 4 個圖像可能是：
  - 全部是 A
  - 全部是 B
  - A 和 B 混合（例如：3 個 A + 1 個 B）
- 分布仍然是**完全隨機的**

---

## 關鍵代碼位置總結

1. **圖像註冊**: ```2151:2161:library/train_util.py```
2. **添加到 Bucket**: ```1062:1064:library/train_util.py```
3. **批次索引創建**: ```1084:1092:library/train_util.py```
4. **隨機打亂**: ```1094:1099:library/train_util.py``` 和 ```252:254:library/train_util.py```
5. **批次獲取**: ```1563:1591:library/train_util.py```
6. **訓練循環**: ```1549:train_network.py```

---

## 結論

**A 和 B 的圖像是完全隨機混合分布在 2000 steps 中的。**

- ✅ 每個 step 都有 50% 機率是 A 或 B（長期平均）
- ✅ 分布是**完全隨機的**，不是交替或順序的
- ✅ 可能會出現連續多個 A 或連續多個 B 的情況
- ✅ 這是通過 `random.shuffle()` 實現的隨機分布

這種隨機分布有助於：
1. 防止模型過度記憶特定的圖像順序
2. 確保訓練過程的隨機性
3. 提高模型的泛化能力
