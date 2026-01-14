# reg_data_dir 使用流程分析

## 概述

`reg_data_dir` 是用于指定正则化（regularization）图像目录的参数，在 DreamBooth 训练方法中使用。正则化图像用于防止模型过拟合，通过让模型同时学习原始类别图像和正则化图像来保持模型的泛化能力。

## 完整使用流程

### 1. 参数解析阶段

**位置**: `library/train_util.py`

```python
parser.add_argument(
    "--reg_data_dir", type=str, default=None,
    help="directory for regularization images / 正則化画像データのディレクトリ"
)
```

`reg_data_dir` 作为命令行参数被解析，默认值为 `None`。

---

### 2. 配置生成阶段

**位置**: `train_network.py` (第 562-572 行)

当使用 DreamBooth 方法且没有提供 `dataset_config` 时，系统会调用 `generate_dreambooth_subsets_config_by_subdirs` 函数：

```567:569:train_network.py
                                config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
```

**位置**: `library/config_util.py` (第 607-644 行)

该函数扫描 `reg_data_dir` 目录下的子目录：

```607:644:library/config_util.py
def generate_dreambooth_subsets_config_by_subdirs(train_data_dir: Optional[str] = None, reg_data_dir: Optional[str] = None):
    def extract_dreambooth_params(name: str) -> Tuple[int, str]:
        tokens = name.split("_")
        try:
            n_repeats = int(tokens[0])
        except ValueError as e:
            logger.warning(f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {name}")
            return 0, ""
        caption_by_folder = "_".join(tokens[1:])
        return n_repeats, caption_by_folder

    def generate(base_dir: Optional[str], is_reg: bool):
        if base_dir is None:
            return []

        base_dir: Path = Path(base_dir)
        if not base_dir.is_dir():
            return []

        subsets_config = []
        for subdir in base_dir.iterdir():
            if not subdir.is_dir():
                continue

            num_repeats, class_tokens = extract_dreambooth_params(subdir.name)
            if num_repeats < 1:
                continue

            subset_config = {"image_dir": str(subdir), "num_repeats": num_repeats, "is_reg": is_reg, "class_tokens": class_tokens}
            subsets_config.append(subset_config)

        return subsets_config

    subsets_config = []
    subsets_config += generate(train_data_dir, False)
    subsets_config += generate(reg_data_dir, True)

    return subsets_config
```

**关键点**:
- 子目录命名格式：`{num_repeats}_{class_tokens}`，例如 `10_dog` 表示重复 10 次，类别标记为 "dog"
- 对于 `reg_data_dir`，`is_reg=True` 被设置
- 生成的配置包含：`image_dir`、`num_repeats`、`is_reg`、`class_tokens`

---

### 3. 数据集初始化阶段

**位置**: `library/train_util.py` (第 1910-2193 行)

在 `DreamBoothDataset.__init__` 中处理正则化图像：

#### 3.1 扫描正则化子集

```2146:2164:library/train_util.py
            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(img_path, num_repeats, caption, subset.is_reg, img_path)
                info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)
```

- 对于 `is_reg=True` 的子集，图像信息被收集到 `reg_infos` 列表中
- 计算正则化图像总数：`num_reg_images += num_repeats * len(img_paths)`

#### 3.2 注册正则化图像到数据集

```2175:2193:library/train_util.py
        if num_reg_images == 0:
            logger.warning("no regularization images / 正則化画像が見つかりませんでした")
        else:
            # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images
```

**关键逻辑**:
- 确保正则化图像的数量与训练图像的数量匹配
- 如果训练图像数量少于正则化图像数量，会发出警告
- 通过循环注册，使正则化图像的总数等于训练图像总数

---

### 4. 批次数据加载阶段

**位置**: `library/train_util.py` (第 1563-1591 行)

在 `BaseDataset.__getitem__` 中，为每个图像设置损失权重：

```1590:1591:library/train_util.py
            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
```

**关键点**:
- 如果 `image_info.is_reg == True`，使用 `prior_loss_weight`（默认 1.0，可通过 `--prior_loss_weight` 参数调整）
- 如果 `image_info.is_reg == False`，使用 `1.0`
- `loss_weights` 被添加到批次数据中，供训练循环使用

---

### 5. 训练循环阶段

**位置**: `train_network.py` (第 380-513 行)

在 `process_batch` 方法中处理批次数据：

#### 5.1 计算损失

```500:506:train_network.py
            loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c, step, global_step)

        if weighting is not None:
            loss = loss * weighting
        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
            loss = apply_masked_loss(loss, batch)
        loss = loss.mean([1, 2, 3])
```

#### 5.2 应用损失权重

```508:513:train_network.py
        loss_weights = batch["loss_weights"]  # 各sampleごとのweight
        loss = loss * loss_weights

        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler, global_step)

        return loss.mean()
```

**关键点**:
- `loss_weights` 从批次数据中获取
- 每个样本的损失乘以对应的权重
- 正则化图像的损失会被 `prior_loss_weight` 加权
- 最终返回平均损失

---

## 数据流图

```
命令行参数 (--reg_data_dir)
    ↓
generate_dreambooth_subsets_config_by_subdirs()
    ↓
生成配置 (is_reg=True)
    ↓
DreamBoothDataset.__init__()
    ↓
扫描 reg_data_dir 子目录 → 收集 reg_infos
    ↓
注册正则化图像到数据集 (匹配训练图像数量)
    ↓
BaseDataset.__getitem__()
    ↓
根据 is_reg 设置 loss_weights
    (is_reg=True → prior_loss_weight, is_reg=False → 1.0)
    ↓
训练循环 process_batch()
    ↓
loss = loss * loss_weights
    ↓
反向传播和优化
```

---

## 关键参数

### prior_loss_weight

**位置**: `library/train_util.py` (第 4282 行)

```python
parser.add_argument(
    "--prior_loss_weight", type=float, default=1.0,
    help="loss weight for regularization images / 正則化画像のlossの重み"
)
```

- **默认值**: 1.0
- **作用**: 控制正则化图像损失的权重
- **使用场景**:
  - 通常设为 1.0
  - 如果正则化图像质量较差，可以降低此值
  - 如果希望更强调正则化，可以增加此值

---

## 目录结构要求

`reg_data_dir` 目录应遵循以下结构：

```
reg_data_dir/
├── {num_repeats}_{class_tokens}/
│   ├── image1.jpg
│   ├── image1.txt  (可选，包含 caption)
│   ├── image2.jpg
│   └── ...
├── {num_repeats}_{class_tokens}/
│   └── ...
└── ...
```

**示例**:
```
reg_data_dir/
├── 10_dog/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
└── 5_cat/
    ├── cat1.jpg
    └── ...
```

---

## 注意事项

1. **仅用于 DreamBooth 方法**: `reg_data_dir` 只在 DreamBooth 训练模式下使用（即不使用 `--in_json` 参数时）

2. **与 dataset_config 冲突**: 如果提供了 `--dataset_config`，`reg_data_dir` 参数会被忽略

3. **正则化图像数量**: 系统会自动调整正则化图像的数量以匹配训练图像数量

4. **损失权重**: 正则化图像使用 `prior_loss_weight` 作为损失权重，而训练图像使用 1.0

5. **子目录命名**: 子目录名称必须遵循 `{num_repeats}_{class_tokens}` 格式，否则会被忽略

---

## 相关文件

- `train_network.py`: 网络训练主脚本
- `library/train_util.py`: 训练工具函数和数据集类
- `library/config_util.py`: 配置生成工具
- `train_db.py`: DreamBooth 训练脚本
- `sdxl_train.py`, `sd3_train.py`, `flux_train.py`, `lumina_train.py`: 各模型的训练脚本

---

## 总结

`reg_data_dir` 的数据在训练过程中的使用时机：

1. **初始化阶段**: 扫描目录并生成配置
2. **数据集构建阶段**: 加载图像并注册到数据集
3. **批次加载阶段**: 为每个图像设置 `is_reg` 标志和 `loss_weights`
4. **训练循环阶段**: 在损失计算时应用权重，正则化图像的损失被 `prior_loss_weight` 加权

正则化图像与训练图像混合在同一个批次中，通过不同的损失权重来平衡它们对模型训练的影响。
