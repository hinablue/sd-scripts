### sd-scripts 的 Modal 訓練執行器

本文說明如何使用 `run_modal.py` 在 Modal 上執行 kohya-ss/sd-scripts 的訓練。內容包含安裝設定、必要的 Secret/Volume、指令參數，以及各訓練型別的使用範例。

### 概觀

- 應用程式會建置包含 CUDA PyTorch 的 Modal 映像，安裝本專案相依套件，並將你本機的 `sd-scripts` 掛載到容器內的 `/root/sd-scripts`。
- 使用 Modal Volume（預設：`flux-lora-models`），掛載至 `/root/sd-scripts/modal_output`，用於快取模型與儲存訓練輸出。
- 會自動下載必要的基底模型/輔助檔（VAE、CLIP、T5 等）到該 Volume，並將正確路徑傳入訓練腳本。

### 先決條件

- Modal 帳號與 CLI
  - 安裝：`pip install modal`
  - 登入：`modal token new`
- Modal 上的 GPU 使用權（預設 `A100`）。
- 確認本機專案路徑與 `run_modal.py` 中的 `LOCAL_SD_SCRIPTS_DIR` 一致。
  - 程式碼預設：`/Users/username/sd-scripts`
  - 請改成你的實際絕對路徑，例如：`/Users/hina/Workspace/sd-scripts`。

### 來自 run_modal.py 的重要預設值

- 專案掛載：本機 `LOCAL_SD_SCRIPTS_DIR = "/Users/username/sd-scripts"` → 容器 `/root/sd-scripts`
- App 名稱：`MODAL_APPLICATION_NAME = "flux-lora-training"`
- GPU：`MODAL_GPU = "A100"`（可在程式中調整）
- 逾時：`MODAL_APPLICATION_TIMEOUT = 7200` 單位：秒（2 小時）
- Volume 名稱：`MODAL_VOLUME_NAME = "flux-lora-models"`（不存在時自動建立）
- Volume 掛載點：`MOUNT_DIR = "/root/sd-scripts/modal_output"`
- Secrets：`HUGGINGFACE_SECRET_NAME = "huggingface-secret"` → 需要 `HF_TOKEN`
  - 建立名為 `huggingface-secret`（可調整）的 Modal Secret，內含環境變數 `HF_TOKEN`（你的 HF token）。
  - 範例：`modal secret create huggingface-secret HF_TOKEN=hf_xxx`
- 使用的 Accelerate 設定：`/root/sd-scripts/datasets/modal_accelerate_config.yaml`

### 環境變數

- Modal 函式內必須：`HF_TOKEN`（由 Modal Secret 提供）
- 選用：`DEBUG_TOOLKIT=1` 會啟用 `torch.autograd.set_detect_anomaly(True)`
- 腳本會自動設定下列參數：
  - `DISABLE_TELEMETRY=YES`
  - `TOKENIZERS_PARALLELISM=false`

### 輸出與儲存位置

- 所有輸出（模型檔、範例、最佳化器狀態等）都寫入 `/root/sd-scripts/modal_output`，對應到 Modal Volume `flux-lora-models`。
- 若提供 `--name`，輸出將寫入 `/root/sd-scripts/modal_output/<name>`。
- 可於 Modal Storage 瀏覽該 Volume：`https://modal.com/storage/<your-username>/main/flux-lora-models`。

### 預先準備模型檔案

- 若使用預設的 `flux-lora-models` 可以在執行前先建立。
  - 範例：`modal volume create flux-lora-models`
- 請在本地資料夾，例如 `Downloads/models` 當中，下載以下模型檔案。
  - `https://huggingface.co/hinablue/modal-for-sd-scripts/tree/main` 所有的檔案。
  - 根據你要訓練的模型，下載所需要的檔案。
    - [SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
    - [FLUX1.dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors)
    - [FLUX1.krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/resolve/main/flux1-krea-dev.safetensors)
    - [SD3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/resolve/main/sd3.5_medium.safetensors)
    - [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors)
    - [Lumina Image 2](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/resolve/3648256a142a83759fc9b8599113780a02f425c1/lumina-image-2.safetensors)
- 準備好模型之後，可以先將這些模型同步到 Modal Volume 當中，用以節省訓練器下載的時間。
  - 在你下載模型的資料夾當中執行以下命令：
  - `modal volume put flux-lora-models . models/`
- 前往 [Modal Storage](https://modal.com/storage) 察看是否建立成功。

### TOML 設定檔

- 以下項目不需要設定，請留空白即可，訓練開始前會預先準備。
  - ae = ""
  - clip_l = ""
  - clip_g = ""
  - t5xxl = ""
  - vae = ""
  - gemma2 = ""
  - pretrained_model_name_or_path = ""
- 若需要修改 `pretrained_model_name_or_path`，請使用 `-m/--pretrained-model` 指定。

### 訓練型別與自動下載模型檔案

使用 `-t/--train_type` 選擇訓練進入點，缺少時會自動下載依賴：

- `flux`（腳本：`flux_train_network.py`）
  - 下載：`clip_l.safetensors`、`flux_vae.safetensors`、`t5xxl_fp8_e4m3fn.safetensors`
  - 傳入參數：`--ae`、`--clip_l`、`--t5xxl`
  - 基底模型：若未提供，預設下載 `FLUX.1-dev` → `flux1-dev.safetensors`

- `flux_krea`（腳本：`flux_train_network.py`）
  - 與 `flux` 相同，但基底模型預設改為 `FLUX.1-Krea-dev` → `flux1-krea-dev.safetensors`

- `sdxl`（腳本：`sdxl_train_network.py`）
  - 下載：`sdxl_vae_fp16_fix.safetensors`
  - 傳入參數：`--vae`
  - 基底模型：若未提供，下載 `sd_xl_base_1.0.safetensors`

- `sd3` / `sd3_medium`（腳本：`sd3_train_network.py`）
  - 下載：`sd35_clip_l.safetensors`、`sd35_clip_g.safetensors`、`t5xxl_fp8_e4m3fn.safetensors`
  - 傳入參數：`--clip_l`、`--clip_g`、`--t5xxl`
  - 基底模型：若未提供，將下載
    - `sd3` → `sd3.5_large.safetensors`
    - `sd3_medium` → `sd3.5_medium.safetensors`

- `lumina`（腳本：`lumina_train_network.py`）
  - 下載：`flux_vae.safetensors`、`gemma-2-2b-fp16.safetensors`
  - 傳入參數：`--ae`、`--gemma2`
  - 基底模型：若未提供，下載 `lumina-image-2.safetensors`

若你以 HTTP URL 提供 `-m/--pretrained_model`，腳本會將其下載到 Volume，並以 `--pretrained_model_name_or_path` 使用。

### 設定檔與額外參數

必須提供設定檔或 `additional_args`（可同時提供）：

- `-c/--config_file`：訓練腳本會讀取的 TOML 設定檔。若提供，可能指定 `output_name` 與 `pretrained_model_name_or_path`（僅接受 URL）。
- `-a/--additional_args`：額外 CLI 旗標，用來覆蓋設定檔值。範例：`-a "--lr=1e-5 --network_dim=32"`
- `-n/--name`：若設定，腳本會加入 `--output_dir` 與 `--output_name` 指向 `/root/sd-scripts/modal_output/<name>`。

Accelerate 一律以以下方式啟動：

```bash
accelerate launch --config_file /root/sd-scripts/datasets/modal_accelerate_config.yaml /root/sd-scripts/<train_script>.py [--config_file=...] [additional_args]
```

### 使用方式

基本範例（flux）：

```bash
modal run run_modal.py -t flux -c /root/sd-scripts/datasets/config.toml
```

自訂執行名稱與額外覆蓋參數：

```bash
modal run run_modal.py -t flux -c /root/sd-scripts/datasets/config.toml -n my-flux-run -a "--lr=1e-4 --network_dim=32"
```

以 URL 指定特定基底模型：

```bash
modal run run_modal.py -t sdxl -c /root/sd-scripts/datasets/config.toml -m https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

其他訓練型別：

```bash
# FLUX.1-Krea-dev
modal run run_modal.py -t flux_krea -c /root/sd-scripts/datasets/config.toml

# SD3.5 Large
modal run run_modal.py -t sd3 -c /root/sd-scripts/datasets/config.toml

# SD3.5 Medium
modal run run_modal.py -t sd3_medium -c /root/sd-scripts/datasets/config.toml

# Lumina Image 2.0
modal run run_modal.py -t lumina -c /root/sd-scripts/datasets/config.toml
```

### 重要檔案路徑

- 設定檔範例：見 `datasets/example_flux_config.toml`
- Modal 使用的 Accelerate 設定：`datasets/modal_accelerate_config.yaml`
- 專案根目錄下的訓練入口：`flux_train_network.py`、`sdxl_train_network.py`、`sd3_train_network.py`、`lumina_train_network.py`

### 注意事項

- 為符合 Modal 的 Volume 掛載限制，Volume 會掛載在非空資料夾 `/root/sd-scripts/modal_output`。
- 若提供 TOML 設定且包含 `pretrained_model_name_or_path`，其值必須為 HTTP URL（此執行器不接受本機路徑）。
- 本機掛載會忽略大型或非必要資料夾，如 `tests`、`docs`、`wandb`、Windows 版 bitsandbytes 等。
- GPU 與逾時可在 `run_modal.py` 的 `main` 裝飾器中調整。

### 疑難排解

- 錯誤："Train type is required"
  - 請提供 `-t`，可用值：`flux`、`flux_krea`、`sdxl`、`sd3`、`sd3_medium`、`lumina`。

- 錯誤："Config file or additional_args are required"
  - 請提供 `-c /root/sd-scripts/path/to.toml` 或 `-a "..."`。

- 錯誤："Pretrained model name or path must be a valid URL"
  - 若 TOML 設定了 `pretrained_model_name_or_path`，請確保為 HTTP URL（或改用 `-m` 指定 URL）。

- 下載模型出現權限/404 問題
  - 確認 `HF_TOKEN` 有效，且 Modal Secret `huggingface-secret` 已包含該值。

- TOML 內引用的檔案找不到
  - 在 Modal 環境中請使用容器內的絕對路徑，例如 `/root/sd-scripts/...`，或直接依賴此執行器自動提供的路徑。


