# 傅立葉特徵損失 Latent Space 整合使用指南

## 🎯 概述

本指南介紹如何在 SD-Scripts 框架中使用新整合的傅立葉特徵損失功能。該實現專門針對 latent space 訓練環境設計，提供了多種模式和配置選項。

## ⚡ 快速開始

### 基本使用

```bash
# 使用傅立葉損失訓練 LoRA
python train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/path/to/training/data" \
  --output_dir="/path/to/output" \
  --loss_type="fourier" \
  --learning_rate=1e-4 \
  --max_train_steps=2000
```

### 使用預設配置

```bash
# 平衡配置（推薦初學者）
python train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/path/to/training/data" \
  --output_dir="/path/to/output" \
  --loss_type="fourier" \
  --fourier_config="balanced"

# 超解析度專用配置
python train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/path/to/training/data" \
  --output_dir="/path/to/output" \
  --loss_type="fourier" \
  --fourier_config="super_resolution"
```

## 🛠️ 配置選項

### 預設配置模式

| 模式 | 權重 | 描述 | 適用場景 |
|------|------|------|----------|
| `conservative` | 0.01 | 保守配置，小權重 | 穩定訓練，初次嘗試 |
| `balanced` | 0.05 | 平衡配置，中等權重 | 一般用途，推薦 |
| `aggressive` | 0.10 | 激進配置，大權重 | 追求最大效果 |
| `super_resolution` | 0.08 | 超解析度專用 | 圖像增強任務 |
| `fine_detail` | 0.12 | 細節增強 | 提升細節表現 |

### 傅立葉損失模式

#### 1. Basic 模式
```bash
--fourier_mode="basic" \
--fourier_weight=0.05
```
- **特點**：最簡單的傅立葉損失實現
- **優點**：計算速度快，記憶體佔用少
- **適用**：初學者，快速實驗

#### 2. Weighted 模式（推薦）
```bash
--fourier_mode="weighted" \
--fourier_weight=0.05 \
--fourier_high_freq_weight=2.0
```
- **特點**：對高頻成分給予更高權重
- **優點**：平衡低頻和高頻特徵
- **適用**：大多數訓練場景

#### 3. Multiscale 模式
```bash
--fourier_mode="multiscale" \
--fourier_weight=0.08 \
--fourier_scales="1,2,4"
```
- **特點**：多尺度頻率分析
- **優點**：捕捉不同尺度的特徵
- **適用**：複雜圖像，多尺度特徵重要的場景

#### 4. Adaptive 模式
```bash
--fourier_mode="adaptive" \
--fourier_weight=0.06 \
--fourier_adaptive_max_weight=3.0 \
--fourier_adaptive_min_weight=1.0
```
- **特點**：根據訓練進度自動調整權重
- **優點**：早期重視高頻，後期平衡
- **適用**：長期訓練，自動優化

## 🔧 高級配置

### 完整參數列表

```bash
python train_network.py \
  --loss_type="fourier" \
  --fourier_weight=0.05 \
  --fourier_mode="weighted" \
  --fourier_norm="l2" \
  --fourier_high_freq_weight=2.0 \
  --fourier_warmup_steps=300 \
  --fourier_eps=1e-8 \
  # ... 其他訓練參數
```

### 參數說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `fourier_weight` | float | 0.05 | 傅立葉損失權重 |
| `fourier_mode` | str | "weighted" | 傅立葉損失模式 |
| `fourier_norm` | str | "l2" | 損失範數類型 ("l1" 或 "l2") |
| `fourier_high_freq_weight` | float | 2.0 | 高頻權重倍數 |
| `fourier_warmup_steps` | int | 300 | 預熱步數 |
| `fourier_eps` | float | 1e-8 | 數值穩定性常數 |

### 自定義配置示例

```python
# 在訓練腳本中使用自定義配置
from library.train_util import get_fourier_loss_config

# 獲取基礎配置
config = get_fourier_loss_config("balanced")

# 修改特定參數
config["fourier_weight"] = 0.08
config["fourier_high_freq_weight"] = 2.5

# 應用配置
for key, value in config.items():
    setattr(args, key, value)
```

## 📊 性能優化

### 記憶體優化

```bash
# 減少 VRAM 使用
python train_network.py \
  --loss_type="fourier" \
  --fourier_mode="basic" \
  --fourier_weight=0.03 \
  --mixed_precision="fp16" \
  --gradient_checkpointing
```

### 計算優化

```bash
# 優化計算速度
python train_network.py \
  --loss_type="fourier" \
  --fourier_mode="weighted" \
  --fourier_warmup_steps=500 \  # 增加預熱期
  --dataloader_num_workers=4
```

## 🎨 不同模型類型的建議配置

### Stable Diffusion 1.x/2.x

```bash
python train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --loss_type="fourier" \
  --fourier_config="balanced" \
  --network_module="networks.lora" \
  --network_dim=128 \
  --learning_rate=1e-4
```

### SDXL

```bash
python sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --loss_type="fourier" \
  --fourier_config="super_resolution" \
  --network_module="networks.lora" \
  --network_dim=256 \
  --learning_rate=5e-5
```

### SD3

```bash
python sd3_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium" \
  --loss_type="fourier" \
  --fourier_mode="adaptive" \
  --fourier_weight=0.06 \
  --learning_rate=3e-5
```

### FLUX

```bash
python flux_train_network.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --loss_type="fourier" \
  --fourier_config="fine_detail" \
  --learning_rate=2e-5
```

## 📈 效果監控

### 訓練日誌監控

訓練過程中關注以下指標：

```
# 正常的損失日誌示例
step: 100, loss: 0.1234, fourier_component: 0.0234
step: 200, loss: 0.1156, fourier_component: 0.0198
step: 300, loss: 0.1089, fourier_component: 0.0167
```

### 異常情況處理

1. **損失爆炸**
   ```bash
   # 減少傅立葉權重
   --fourier_weight=0.01
   ```

2. **記憶體不足**
   ```bash
   # 使用 basic 模式
   --fourier_mode="basic"
   --fourier_weight=0.03
   ```

3. **訓練不穩定**
   ```bash
   # 增加預熱期
   --fourier_warmup_steps=500
   ```

## 🧪 實驗建議

### A/B 測試對比

1. **基礎測試**：
   ```bash
   # 不使用傅立葉損失
   --loss_type="l2"

   # 使用傅立葉損失
   --loss_type="fourier" --fourier_config="balanced"
   ```

2. **模式對比**：
   ```bash
   # 測試不同模式
   --fourier_mode="basic"
   --fourier_mode="weighted"
   --fourier_mode="adaptive"
   ```

3. **權重調優**：
   ```bash
   # 測試不同權重
   --fourier_weight=0.01
   --fourier_weight=0.05
   --fourier_weight=0.10
   ```

### 評估指標

- **FID 分數**：圖像品質整體評估
- **LPIPS 分數**：感知相似度
- **PSNR/SSIM**：像素級品質
- **人工評估**：主觀品質判斷

## 🔍 故障排除

### 常見問題

#### Q1: 出現 "傅立葉損失需要至少 3D 張量" 警告
**解決**：這是正常的，表示某些批次的張量維度不足，系統會自動回退到基礎損失。

#### Q2: 訓練速度明顯變慢
**解決**：
```bash
# 降低傅立葉損失權重或使用 basic 模式
--fourier_mode="basic" --fourier_weight=0.03
```

#### Q3: 損失值出現 NaN
**解決**：
```bash
# 增加數值穩定性
--fourier_eps=1e-6
# 或減少權重
--fourier_weight=0.01
```

#### Q4: 記憶體溢出
**解決**：
```bash
# 使用更簡單的配置
--fourier_config="conservative"
--mixed_precision="fp16"
```

### 除錯模式

```bash
# 啟用除錯日誌
export PYTHONPATH=/path/to/sd-scripts:$PYTHONPATH
python -u train_network.py \
  --loss_type="fourier" \
  --fourier_config="balanced" \
  --logging_dir="./logs" \
  --log_with="tensorboard"
```

## 🚀 最佳實踐

### 1. 漸進式調優
- 從 `conservative` 配置開始
- 逐漸增加 `fourier_weight`
- 根據效果選擇合適的模式

### 2. 配置文件管理
```toml
# config.toml
[fourier_loss]
weight = 0.05
mode = "weighted"
high_freq_weight = 2.0
warmup_steps = 300
```

### 3. 實驗記錄
```python
# 記錄實驗配置
experiment_config = {
    "fourier_weight": 0.05,
    "fourier_mode": "weighted",
    "dataset": "custom_art",
    "model": "sd15",
    "results": {
        "fid": 12.34,
        "training_time": "2h30m"
    }
}
```

### 4. 版本控制
- 記錄每次實驗的配置
- 追蹤最佳參數組合
- 建立配置模板庫

## 🔮 未來發展

### 計劃中的功能
- **頻率段選擇**：指定特定頻率範圍的權重
- **動態權重調整**：基於訓練指標自動調整
- **多模態支援**：文本-圖像聯合頻率分析
- **效率優化**：進一步降低計算開銷

### 社群貢獻
- 分享最佳配置
- 報告使用體驗
- 提出改進建議

---

**版本**: v1.0
**更新日期**: 2024
**適用範圍**: SD-Scripts 所有支援的模型
**作者**: Hina

如有問題或建議，請提交 Issue 或 Pull Request。