# 其他模型（Flux, SD3, Lumina, Hunyuan）的 reg_timestep 支持分析

## 概述

分析 Flux、SD3、Lumina、Hunyuan 模型是否可以使用与标准 SD/SDXL 模型相同的逻辑来支持 `--reg_min_timestep` 和 `--reg_max_timestep`。

---

## 模型分析

### 1. Flux 模型

**文件**: `library/flux_train_utils.py`
**函数**: `get_noisy_model_input_and_timesteps`
**调用位置**: `flux_train_network.py` 第 332 行

#### 当前实现

Flux 支持多种 timestep 采样模式：
- `uniform`: 均匀采样
- `sigmoid`: Sigmoid 采样
- `shift`: Shift 采样
- `flux_shift`: Flux shift 采样
- `qinglong_flux`: 青龙 Flux 采样（**唯一使用 min/max_timestep 的模式**）
- `else`: 使用 `compute_density_for_timestep_sampling`（**未使用 min/max_timestep**）

#### 关键代码位置

```python
# 第 536-542 行：只在 qinglong_flux 模式中使用
t_min = args.min_timestep if args.min_timestep is None else 0
t_max = args.max_timestep if args.max_timestep is None else 1000.0
t_min /= 1000.0
t_max /= 1000.0

t = t * (t_max - t_min) + t_min
timesteps = t * 1000.0
```

#### 修改可行性

✅ **可以修改**，但需要：
1. 为所有采样模式添加 `is_reg` 支持
2. 在 `qinglong_flux` 模式中，根据 `is_reg` 使用不同的 `t_min` 和 `t_max`
3. 在其他模式中，需要先应用 min/max 限制，然后根据 `is_reg` 调整

---

### 2. SD3 模型

**文件**: `library/sd3_train_utils.py`
**函数**: `get_noisy_model_input_and_timesteps`
**调用位置**: `sd3_train_network.py` 第 333 行

#### 当前实现

SD3 使用 `compute_density_for_timestep_sampling` 进行采样，并**已经应用了 min/max_timestep**：

```python
# 第 930-937 行
t_min = args.min_timestep if args.min_timestep is None else 0
t_max = args.max_timestep if args.max_timestep is None else 1000
shift = args.training_shift

u = (u * shift) / (1 + (shift - 1) * u)

indices = (u * (t_max - t_min) + t_min).long()
timesteps = indices.to(device=device, dtype=dtype)
```

#### 修改可行性

✅ **最容易修改**，因为：
1. 已经使用了 `min_timestep` 和 `max_timestep`
2. 只需要根据 `is_reg` 使用不同的 `t_min` 和 `t_max` 值
3. 逻辑简单清晰

---

### 3. Lumina 模型

**文件**: `library/lumina_train_util.py`
**函数**: `get_noisy_model_input_and_timesteps`
**调用位置**: `lumina_train_network.py` 第 252 行

#### 当前实现

Lumina 支持多种采样模式：
- `uniform`: 均匀采样，`timesteps = t * 1000.0`
- `sigmoid`: Sigmoid 采样，`timesteps = t * 1000.0`
- `shift`: Shift 采样，`timesteps = timesteps * 1000.0`
- `nextdit_shift`: NextDiT shift 采样，`timesteps = t * 1000.0`
- `else`: 使用 `compute_density_for_timestep_sampling`，**未使用 min/max_timestep**

#### 修改可行性

⚠️ **需要更多工作**，因为：
1. 大部分模式没有应用 min/max 限制
2. 需要先添加 min/max 限制逻辑
3. 然后根据 `is_reg` 调整

---

### 4. Hunyuan 模型

**文件**: `hunyuan_image_train_network.py`
**函数**: 使用 Flux 的 `get_noisy_model_input_and_timesteps`
**调用位置**: `hunyuan_image_train_network.py` 第 540 行

#### 当前实现

Hunyuan **直接使用 Flux 的函数**：
```python
noisy_model_input, _, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
)
```

#### 修改可行性

✅ **自动支持**，因为：
1. 如果 Flux 模型支持了 `is_reg`，Hunyuan 会自动获得支持
2. 只需要在 `get_noise_pred_and_target` 中传递 `is_reg` 参数

---

## 修改方案

### 方案 1：统一修改策略（推荐）

为所有模型的 `get_noisy_model_input_and_timesteps` 函数添加 `is_reg` 参数，并根据 `is_reg` 使用不同的 timestep 范围。

#### 通用修改模式

```python
def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents, noise, device, dtype, is_reg: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = latents.shape[0]

    # 确定训练和正则化图像的 timestep 范围
    train_min = 0 if args.min_timestep is None else args.min_timestep
    train_max = 1000 if args.max_timestep is None else args.max_timestep

    reg_min = args.reg_min_timestep if args.reg_min_timestep is not None else train_min
    reg_max = args.reg_max_timestep if args.reg_max_timestep is not None else train_max

    # 根据 is_reg 分别处理
    if is_reg is not None and is_reg.any():
        # 分别生成训练和正则化图像的 timesteps
        # ... 具体实现根据采样模式而定
    else:
        # 原有逻辑
        # ...
```

---

## 具体修改方案

### 1. Flux 模型修改

**文件**: `library/flux_train_utils.py`

#### 修改点 1：函数签名

```python
def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype,
    is_reg: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

#### 修改点 2：qinglong_flux 模式（第 536-542 行）

```python
elif args.timestep_sampling == "qinglong_flux":
    # ... 前面的代码保持不变 ...

    # 修改这里：根据 is_reg 使用不同的 t_min 和 t_max
    if is_reg is not None and is_reg.any():
        # 分别处理训练和正则化图像
        train_mask = ~is_reg
        reg_mask = is_reg

        t_train = torch.zeros((train_mask.sum().item(),), device=device)
        t_reg = torch.zeros((reg_mask.sum().item(),), device=device)

        # 训练图像的 t_min 和 t_max
        train_t_min = args.min_timestep if args.min_timestep is not None else 0
        train_t_max = args.max_timestep if args.max_timestep is not None else 1000.0

        # 正则化图像的 t_min 和 t_max
        reg_t_min = args.reg_min_timestep if args.reg_min_timestep is not None else train_t_min
        reg_t_max = args.reg_max_timestep if args.reg_max_timestep is not None else train_t_max

        train_t_min /= 1000.0
        train_t_max /= 1000.0
        reg_t_min /= 1000.0
        reg_t_max /= 1000.0

        # 对训练图像应用范围
        if train_mask.any():
            t_train = t[train_mask] * (train_t_max - train_t_min) + train_t_min

        # 对正则化图像应用范围
        if reg_mask.any():
            t_reg = t[reg_mask] * (reg_t_max - reg_t_min) + reg_t_min

        # 合并
        t = torch.zeros((bsz,), device=device)
        t[train_mask] = t_train
        t[reg_mask] = t_reg

        timesteps = t * 1000.0
        timesteps += 1
    else:
        # 原有逻辑
        t_min = args.min_timestep if args.min_timestep is None else 0
        t_max = args.max_timestep if args.max_timestep is None else 1000.0
        t_min /= 1000.0
        t_max /= 1000.0
        t = t * (t_max - t_min) + t_min
        timesteps = t * 1000.0
        timesteps += 1
```

#### 修改点 3：其他模式

对于其他模式（uniform, sigmoid, shift, flux_shift），需要先应用 min/max 限制，然后根据 `is_reg` 调整。但由于这些模式目前没有 min/max 限制，可能需要：

1. **选项 A**：只支持 `qinglong_flux` 模式的 `is_reg`（简单但功能有限）
2. **选项 B**：为所有模式添加 min/max 限制和 `is_reg` 支持（完整但工作量大）

#### 修改点 4：else 分支（compute_density_for_timestep_sampling）

```python
else:
    u = compute_density_for_timestep_sampling(...)

    if is_reg is not None and is_reg.any():
        # 分别处理训练和正则化图像
        train_mask = ~is_reg
        reg_mask = is_reg

        train_min = args.min_timestep if args.min_timestep is None else 0
        train_max = num_timesteps if args.max_timestep is None else args.max_timestep
        reg_min = args.reg_min_timestep if args.reg_min_timestep is not None else train_min
        reg_max = args.reg_max_timestep if args.reg_max_timestep is not None else train_max

        indices = torch.zeros((bsz,), dtype=torch.long, device=device)
        if train_mask.any():
            train_u = u[train_mask]
            train_indices = (train_u * (train_max - train_min) + train_min).long()
            indices[train_mask] = train_indices
        if reg_mask.any():
            reg_u = u[reg_mask]
            reg_indices = (reg_u * (reg_max - reg_min) + reg_min).long()
            indices[reg_mask] = reg_indices

        timesteps = noise_scheduler.timesteps[indices].to(device=device)
    else:
        # 原有逻辑
        indices = (u * num_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
```

---

### 2. SD3 模型修改

**文件**: `library/sd3_train_utils.py`

#### 修改点：函数签名和 timestep 计算（第 918-937 行）

```python
def get_noisy_model_input_and_timesteps(
    args, latents, noise, device, dtype, is_reg: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = latents.shape[0]

    u = compute_density_for_timestep_sampling(...)
    shift = args.training_shift
    u = (u * shift) / (1 + (shift - 1) * u)

    if is_reg is not None and is_reg.any():
        # 分别处理训练和正则化图像
        train_mask = ~is_reg
        reg_mask = is_reg

        train_min = args.min_timestep if args.min_timestep is not None else 0
        train_max = args.max_timestep if args.max_timestep is not None else 1000
        reg_min = args.reg_min_timestep if args.reg_min_timestep is not None else train_min
        reg_max = args.reg_max_timestep if args.reg_max_timestep is not None else train_max

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
        # 原有逻辑
        t_min = args.min_timestep if args.min_timestep is None else 0
        t_max = args.max_timestep if args.max_timestep is None else 1000
        indices = (u * (t_max - t_min) + t_min).long()
        timesteps = indices.to(device=device, dtype=dtype)

    # ... 后续代码保持不变 ...
```

---

### 3. Lumina 模型修改

**文件**: `library/lumina_train_util.py`

#### 修改策略

由于 Lumina 的大部分模式没有 min/max 限制，建议：

1. **为所有模式添加 min/max 限制**（如果需要）
2. **根据 `is_reg` 使用不同的范围**

具体修改类似于 Flux 模型，但需要为每个模式分别处理。

---

### 4. Hunyuan 模型修改

**文件**: `hunyuan_image_train_network.py`

#### 修改点：get_noise_pred_and_target 方法（第 522-542 行）

```python
def get_noise_pred_and_target(
    self,
    args,
    accelerator,
    noise_scheduler,
    latents,
    batch,
    text_encoder_conds,
    unet: hunyuan_image_models.HYImageDiffusionTransformer,
    network,
    weight_dtype,
    train_unet,
    is_train=True,
):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)

    # 獲取 is_reg 信息（如果可用）
    is_reg = batch.get("is_reg", None)

    # get noisy model input and timesteps
    noisy_model_input, _, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
        args, noise_scheduler, latents, noise, accelerator.device, weight_dtype, is_reg=is_reg
    )
    # ... 后续代码保持不变 ...
```

---

## 修改优先级建议

### 高优先级（推荐先实现）

1. **SD3** ✅
   - 已经使用 min/max_timestep
   - 修改最简单
   - 逻辑清晰

2. **Hunyuan** ✅
   - 使用 Flux 的函数
   - 只需修改调用处传递 `is_reg`
   - 如果 Flux 支持了，Hunyuan 自动支持

### 中优先级

3. **Flux (qinglong_flux 模式)** ⚠️
   - 已经使用 min/max_timestep
   - 只需修改 qinglong_flux 模式
   - 其他模式需要额外工作

### 低优先级（可选）

4. **Flux (其他模式)** ⚠️
   - 需要先添加 min/max 限制
   - 然后添加 `is_reg` 支持
   - 工作量大

5. **Lumina** ⚠️
   - 需要为所有模式添加支持
   - 工作量大

---

## 总结

### 可行性评估

| 模型 | 当前状态 | 修改难度 | 推荐优先级 |
|------|---------|---------|-----------|
| **SD3** | ✅ 已使用 min/max | ⭐ 简单 | 🔥 高 |
| **Hunyuan** | ✅ 使用 Flux | ⭐ 简单 | 🔥 高 |
| **Flux (qinglong_flux)** | ✅ 已使用 min/max | ⭐⭐ 中等 | ⚡ 中 |
| **Flux (其他模式)** | ❌ 未使用 min/max | ⭐⭐⭐ 困难 | 💡 低 |
| **Lumina** | ❌ 未使用 min/max | ⭐⭐⭐ 困难 | 💡 低 |

### 建议

1. **先实现 SD3 和 Hunyuan**：这两个最容易实现，可以快速验证功能
2. **然后实现 Flux (qinglong_flux)**：如果用户主要使用这个模式
3. **最后考虑其他模式**：根据实际需求决定是否实现

### 通用原则

所有模型的修改都遵循相同的原则：
1. 添加 `is_reg` 参数到 `get_noisy_model_input_and_timesteps` 函数
2. 根据 `is_reg` 使用不同的 `reg_min_timestep` 和 `reg_max_timestep`
3. 在 `get_noise_pred_and_target` 方法中从 batch 获取 `is_reg` 并传递
