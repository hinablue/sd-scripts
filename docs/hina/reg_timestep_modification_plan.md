# 為 is_reg 資料集添加獨立 timestep 範圍的修改方案

## 目標

為正則化（`is_reg`）資料集添加獨立的 `--reg_min_timestep` 和 `--reg_max_timestep` 參數，使其可以使用與訓練資料不同的 timestep 範圍。

---

## 修改步驟

### 步驟 1：添加命令行參數

**文件**: `library/train_util.py`
**位置**: 約第 4172-4183 行（在 `--max_timestep` 之後）

**修改**：
```python
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000) / U-Net学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）",
    )
    parser.add_argument(
        "--reg_min_timestep",
        type=int,
        default=None,
        help="set minimum time step for regularization images (0~999, default is same as min_timestep) / 正則化画像のtime stepの最小値を設定する（0~999で指定、省略時はmin_timestepと同じ）",
    )
    parser.add_argument(
        "--reg_max_timestep",
        type=int,
        default=None,
        help="set maximum time step for regularization images (1~1000, default is same as max_timestep) / 正則化画像のtime stepの最大値を設定する（1~1000で指定、省略時はmax_timestepと同じ）",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        ...
    )
```

---

### 步驟 2：修改 `get_timesteps` 函數以支持每個樣本不同的範圍

**文件**: `library/train_util.py`
**位置**: 約第 6547-6565 行

**修改前**：
```python
def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device, use_log_norm_timesteps=None, loss_type=None, global_step=None, max_train_steps=None) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    # ... 後續處理 ...
```

**修改後**：
```python
def get_timesteps(
    min_timestep: int,
    max_timestep: int,
    b_size: int,
    device: torch.device,
    use_log_norm_timesteps=None,
    loss_type=None,
    global_step=None,
    max_train_steps=None,
    is_reg: Optional[torch.Tensor] = None,
    reg_min_timestep: Optional[int] = None,
    reg_max_timestep: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample timesteps for a batch. If is_reg is provided, use different timestep ranges
    for regularization images.

    Args:
        is_reg: Boolean tensor of shape (b_size,) indicating which samples are regularization images
        reg_min_timestep: Minimum timestep for regularization images (if None, use min_timestep)
        reg_max_timestep: Maximum timestep for regularization images (if None, use max_timestep)
    """
    if is_reg is not None and reg_min_timestep is not None and reg_max_timestep is not None:
        # 為訓練圖像和正則化圖像分別生成 timesteps
        timesteps = torch.zeros((b_size,), dtype=torch.long, device="cpu")

        # 訓練圖像的 timesteps
        train_mask = ~is_reg
        if train_mask.any():
            train_count = train_mask.sum().item()
            if min_timestep < max_timestep:
                train_timesteps = torch.randint(min_timestep, max_timestep, (train_count,), device="cpu")
            else:
                train_timesteps = torch.full((train_count,), max_timestep, device="cpu")
            timesteps[train_mask] = train_timesteps

        # 正則化圖像的 timesteps
        reg_mask = is_reg
        if reg_mask.any():
            reg_count = reg_mask.sum().item()
            if reg_min_timestep < reg_max_timestep:
                reg_timesteps = torch.randint(reg_min_timestep, reg_max_timestep, (reg_count,), device="cpu")
            else:
                reg_timesteps = torch.full((reg_count,), reg_max_timestep, device="cpu")
            timesteps[reg_mask] = reg_timesteps
    else:
        # 原有邏輯：所有樣本使用相同的範圍
        if min_timestep < max_timestep:
            timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
        else:
            timesteps = torch.full((b_size,), max_timestep, device="cpu")

    # 處理 log_norm_timesteps（需要分別處理訓練和正則化圖像）
    if use_log_norm_timesteps and loss_type == "l2":
        if global_step and max_train_steps:
            m = torch.distributions.LogNormal(0 + (0.65 * global_step / max_train_steps), 1)
        else:
            m = torch.distributions.LogNormal(0.65, 1)

        if is_reg is not None and reg_min_timestep is not None and reg_max_timestep is not None:
            # 分別處理訓練和正則化圖像
            train_mask = ~is_reg
            reg_mask = is_reg

            if train_mask.any():
                train_timesteps_log = m.sample((train_mask.sum().item(),)).to(device) * 250
                while torch.any(train_timesteps_log > max_timestep - 1):
                    train_timesteps_log = m.sample((train_mask.sum().item(),)).to(device) * 250
                timesteps[train_mask] = torch.round(train_timesteps_log).long()

            if reg_mask.any():
                reg_timesteps_log = m.sample((reg_mask.sum().item(),)).to(device) * 250
                while torch.any(reg_timesteps_log > reg_max_timestep - 1):
                    reg_timesteps_log = m.sample((reg_mask.sum().item(),)).to(device) * 250
                timesteps[reg_mask] = torch.round(reg_timesteps_log).long()
        else:
            # 原有邏輯
            timesteps = m.sample((b_size,)).to(device) * 250
            while torch.any(timesteps > max_timestep - 1):
                timesteps = m.sample((b_size,)).to(device) * 250
            timesteps = torch.round(timesteps)

    timesteps = timesteps.long().to(device)
    return timesteps
```

---

### 步驟 3：修改 `get_noise_noisy_latents_and_timesteps` 函數

**文件**: `library/train_util.py`
**位置**: 約第 6568-6605 行

**修改前**：
```python
def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor, global_step=None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    # ...
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device, args.use_log_norm_timesteps, args.loss_type, global_step, args.max_train_steps)
    # ...
```

**修改後**：
```python
def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor, global_step=None, is_reg: Optional[torch.Tensor] = None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = torch.rand(1, device=latents.device) * args.noise_offset
        else:
            noise_offset = args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep

    # 處理正則化圖像的 timestep 範圍
    reg_min_timestep = None
    reg_max_timestep = None
    if is_reg is not None and is_reg.any():
        # 如果提供了 reg_min_timestep 或 reg_max_timestep，使用它們
        # 否則使用與訓練圖像相同的範圍
        reg_min_timestep = args.reg_min_timestep if args.reg_min_timestep is not None else min_timestep
        reg_max_timestep = args.reg_max_timestep if args.reg_max_timestep is not None else max_timestep

    timesteps = get_timesteps(
        min_timestep,
        max_timestep,
        b_size,
        latents.device,
        args.use_log_norm_timesteps,
        args.loss_type,
        global_step,
        args.max_train_steps,
        is_reg=is_reg,
        reg_min_timestep=reg_min_timestep,
        reg_max_timestep=reg_max_timestep,
    )

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # This moves the alphas_cumprod back to the CPU after it is moved in noise_scheduler.add_noise
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    return noise, noisy_latents, timesteps
```

---

### 步驟 4：修改 `get_noise_pred_and_target` 方法

**文件**: `train_network.py`
**位置**: 約第 261-278 行

**修改前**：
```python
    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        global_step=None,
        is_train=True
    ):
        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, global_step)
```

**修改後**：
```python
    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        global_step=None,
        is_train=True
    ):
        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        # 獲取 is_reg 信息（如果可用）
        is_reg = batch.get("is_reg", None)
        if is_reg is not None:
            # 確保 is_reg 在 CPU 上（因為 get_timesteps 在 CPU 上生成 timesteps）
            is_reg = is_reg.cpu()

        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
            args, noise_scheduler, latents, global_step, is_reg=is_reg
        )
```

---

### 步驟 5：處理其他模型的 timestep 採樣（可選）

對於 Flux、SD3、Lumina 等模型，它們使用各自的 `get_noisy_model_input_and_timesteps` 函數。如果需要支持這些模型，也需要類似的修改：

**文件**: `library/flux_train_utils.py`, `library/sd3_train_utils.py`, `library/lumina_train_util.py`

**修改策略**：
1. 在各自的 `get_noisy_model_input_and_timesteps` 函數中添加 `is_reg` 參數
2. 根據 `is_reg` 使用不同的 `t_min` 和 `t_max` 值
3. 在對應的 `NetworkTrainer.get_noise_pred_and_target` 方法中傳遞 `is_reg`

---

## 使用示例

### 示例 1：基本使用

```bash
python train_network.py \
    --train_data_dir ./train_data \
    --reg_data_dir ./reg_data \
    --min_timestep 0 \
    --max_timestep 1000 \
    --reg_min_timestep 200 \
    --reg_max_timestep 800 \
    ...
```

### 示例 2：只限制正則化圖像的最大 timestep

```bash
python train_network.py \
    --train_data_dir ./train_data \
    --reg_data_dir ./reg_data \
    --min_timestep 0 \
    --max_timestep 1000 \
    --reg_max_timestep 500 \
    ...
```

### 示例 3：在配置文件中使用

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

## 注意事項

1. **向後兼容性**：
   - 如果未提供 `--reg_min_timestep` 和 `--reg_max_timestep`，正則化圖像將使用與訓練圖像相同的 timestep 範圍
   - 這確保了現有代碼不會受到影響

2. **驗證邏輯**：
   - 應該驗證 `reg_min_timestep >= 0` 和 `reg_max_timestep <= 1000`
   - 應該驗證 `reg_min_timestep < reg_max_timestep`

3. **性能影響**：
   - 為每個樣本單獨生成 timestep 會略微增加計算開銷
   - 但由於只是隨機數生成，影響應該很小

4. **日誌記錄**：
   - 建議在訓練開始時記錄使用的 timestep 範圍
   - 可以在 metadata 中保存這些信息

---

## 測試建議

1. **單元測試**：
   - 測試 `get_timesteps` 函數在提供和不提供 `is_reg` 時的行為
   - 驗證 timestep 範圍的正確性

2. **集成測試**：
   - 使用包含正則化圖像的數據集進行訓練
   - 驗證正則化圖像和訓練圖像使用不同的 timestep 範圍

3. **邊界情況**：
   - 測試所有樣本都是正則化圖像的情況
   - 測試所有樣本都是訓練圖像的情況
   - 測試 `reg_min_timestep == reg_max_timestep` 的情況

---

## 總結

通過以上修改，可以實現為正則化圖像設置獨立的 timestep 範圍。主要修改點：

1. ✅ 添加命令行參數
2. ✅ 修改 `get_timesteps` 函數支持每個樣本不同的範圍
3. ✅ 修改 `get_noise_noisy_latents_and_timesteps` 函數傳遞 `is_reg` 信息
4. ✅ 修改 `get_noise_pred_and_target` 方法從 batch 中獲取 `is_reg`
5. ⚠️ （可選）為其他模型添加類似支持
