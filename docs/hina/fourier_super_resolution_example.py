#!/usr/bin/env python3
"""
HinaAdaptive 優化器正則化技術示例腳本

本腳本展示如何使用 HinaAdaptive 優化器的各種正則化技術，
包括邊緣感知、空間感知、背景正則化和 LoRA 低秩正則化。

注意：傅立葉特徵損失功能已被移除，因為它不適用於 SD-Scripts
的 latent space 訓練環境。

作者: Hina
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import sys
import os

# 添加庫路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from library.hina_adaptive import HinaAdaptive

class SuperResolutionModel(nn.Module):
    """
    簡單的超解析度模型示例
    """
    def __init__(self, scale_factor: int = 4, num_channels: int = 3):
        super().__init__()
        self.scale_factor = scale_factor

        # 特徵提取層
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 殘差塊
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(64) for _ in range(6)
        ])

        # 上採樣層
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64 * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, num_channels, 3, padding=1),
        )

    def forward(self, x):
        # 特徵提取
        features = self.feature_extraction(x)

        # 殘差處理
        residual = self.residual_blocks(features)

        # 上採樣
        sr_image = self.upsampling(residual)

        return sr_image

class ResidualBlock(nn.Module):
    """殘差塊"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

def generate_synthetic_data(batch_size: int = 8, image_size: int = 64,
                          scale_factor: int = 4, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成用於測試的合成數據
    """
    # 生成高解析度圖像
    hr_size = image_size * scale_factor
    hr_images = torch.randn(batch_size, 3, hr_size, hr_size, device=device)

    # 通過下採樣創建低解析度圖像
    lr_images = F.interpolate(hr_images, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # 添加一些細節結構
    for i in range(batch_size):
        # 添加邊緣結構
        hr_images[i, 0, 50:150, 50:150] = 1.0
        hr_images[i, 1, 100:200, 100:200] = 1.0
        hr_images[i, 2, 150:250, 150:250] = 1.0

        # 重新生成對應的低解析度版本
        lr_images[i] = F.interpolate(hr_images[i:i+1], size=(image_size, image_size), mode='bilinear', align_corners=False)[0]

    return lr_images, hr_images

def train_with_edge_suppression(scale_factor: int = 4):
    """
    使用邊緣感知正則化訓練示例
    """
    print("=" * 60)
    print(f"🔍 邊緣感知正則化訓練示例 (Scale: {scale_factor}x)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建模型
    model = SuperResolutionModel(scale_factor=scale_factor).to(device)
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters())}")

    # 創建優化器 - 使用邊緣感知正則化
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-4,
        # 邊緣感知正則化
        edge_suppression=True,
        edge_penalty=0.1,
        edge_threshold=0.6,
        # 結合其他功能
        use_tam=True,
        use_cautious=True,
        memory_efficient=True,
        vram_budget_gb=8.0
    )

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"優化器類型: {info['optimizer_type']}")
    print(f"邊緣感知正則化: {info['features']['edge_suppression']}")
    print(f"TAM 阻尼: {info['features']['tam']}")

    # 訓練循環
    model.train()
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # 生成訓練數據
        lr_images, hr_images = generate_synthetic_data(
            batch_size=4,
            image_size=32,
            scale_factor=scale_factor,
            device=device
        )

        # 前向傳播
        optimizer.zero_grad()
        sr_images = model(lr_images)

        # 計算損失
        loss = F.mse_loss(sr_images, hr_images)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 輸出訓練信息
        with torch.no_grad():
            psnr = 20 * torch.log10(2.0 / torch.sqrt(F.mse_loss(sr_images, hr_images)))
            print(f"Loss: {loss.item():.4f}, PSNR: {psnr.item():.2f}dB")

    print("✅ 邊緣感知正則化訓練完成!")
    return model, optimizer

def train_with_spatial_awareness(scale_factor: int = 4):
    """
    使用空間感知正則化訓練示例
    """
    print("=" * 60)
    print(f"🔍 空間感知正則化訓練示例 (Scale: {scale_factor}x)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建模型
    model = SuperResolutionModel(scale_factor=scale_factor).to(device)

    # 創建優化器 - 使用空間感知正則化
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-4,
        # 空間感知正則化
        spatial_awareness=True,
        frequency_penalty=0.05,
        detail_preservation=0.8,
        # 結合其他功能
        use_dynamic_adaptation=True,
        memory_efficient=True,
        vram_budget_gb=8.0
    )

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"空間感知正則化: {info['features']['spatial_awareness']}")
    print(f"動態自適應: {info['features']['dynamic_adaptation']}")

    # 訓練循環
    model.train()
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # 生成訓練數據
        lr_images, hr_images = generate_synthetic_data(
            batch_size=4,
            image_size=32,
            scale_factor=scale_factor,
            device=device
        )

        # 前向傳播
        optimizer.zero_grad()
        sr_images = model(lr_images)

        # 計算損失
        loss = F.mse_loss(sr_images, hr_images)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 輸出訓練信息
        with torch.no_grad():
            psnr = 20 * torch.log10(2.0 / torch.sqrt(F.mse_loss(sr_images, hr_images)))
            print(f"Loss: {loss.item():.4f}, PSNR: {psnr.item():.2f}dB")

    print("✅ 空間感知正則化訓練完成!")
    return model, optimizer

def train_with_background_regularization(scale_factor: int = 4):
    """
    使用背景正則化訓練示例
    """
    print("=" * 60)
    print(f"🔍 背景正則化訓練示例 (Scale: {scale_factor}x)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建模型
    model = SuperResolutionModel(scale_factor=scale_factor).to(device)

    # 創建優化器 - 使用背景正則化
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-4,
        # 背景正則化
        background_regularization=True,
        # 結合其他功能
        use_spd=True,
        use_orthogonal_grad=True,
        memory_efficient=True,
        vram_budget_gb=8.0
    )

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"背景正則化: {info['features']['background_regularization']}")
    print(f"SPD 正則化: {info['features']['spd']}")
    print(f"正交梯度投影: {info['features']['orthogonal_grad']}")

    # 訓練循環
    model.train()
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # 生成訓練數據
        lr_images, hr_images = generate_synthetic_data(
            batch_size=4,
            image_size=32,
            scale_factor=scale_factor,
            device=device
        )

        # 前向傳播
        optimizer.zero_grad()
        sr_images = model(lr_images)

        # 計算損失
        loss = F.mse_loss(sr_images, hr_images)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 輸出訓練信息
        with torch.no_grad():
            psnr = 20 * torch.log10(2.0 / torch.sqrt(F.mse_loss(sr_images, hr_images)))
            print(f"Loss: {loss.item():.4f}, PSNR: {psnr.item():.2f}dB")

    print("✅ 背景正則化訓練完成!")
    return model, optimizer

def train_with_lora_regularization():
    """
    使用 LoRA 低秩正則化訓練示例
    """
    print("=" * 60)
    print("🔍 LoRA 低秩正則化訓練示例")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建簡單的線性模型（適合 LoRA）
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)

    # 創建優化器 - 使用 LoRA 低秩正則化
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-4,
        # LoRA 低秩正則化
        lora_rank_penalty=True,
        rank_penalty_strength=0.01,
        low_rank_emphasis=1.2,
        # 結合其他功能
        use_lr_mask=True,
        dynamic_weight_decay=True,
        memory_efficient=True,
        vram_budget_gb=8.0
    )

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print(f"LoRA 低秩正則化: {info['features']['lora_rank_penalty']}")
    print(f"學習率遮罩: {info['features']['lr_mask']}")
    print(f"動態權重衰減: {info['features']['dynamic_weight_decay']}")

    # 訓練循環
    model.train()
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # 生成訓練數據
        batch_size = 32
        x = torch.randn(batch_size, 128, device=device)
        y = torch.randn(batch_size, 64, device=device)

        # 前向傳播
        optimizer.zero_grad()
        pred = model(x)

        # 計算損失
        loss = F.mse_loss(pred, y)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 輸出訓練信息
        print(f"Loss: {loss.item():.4f}")

    print("✅ LoRA 低秩正則化訓練完成!")
    return model, optimizer

def train_with_combined_regularization(scale_factor: int = 4):
    """
    使用組合正則化技術訓練示例
    """
    print("=" * 60)
    print(f"🔍 組合正則化技術訓練示例 (Scale: {scale_factor}x)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建模型
    model = SuperResolutionModel(scale_factor=scale_factor).to(device)

    # 創建優化器 - 使用組合正則化技術
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-4,
        # 組合多種正則化技術
        edge_suppression=True,
        edge_penalty=0.1,
        spatial_awareness=True,
        frequency_penalty=0.05,
        background_regularization=True,
        lora_rank_penalty=True,
        rank_penalty_strength=0.01,
        # 其他功能
        use_dynamic_adaptation=True,
        use_tam=True,
        use_cautious=True,
        use_spd=True,
        memory_efficient=True,
        vram_budget_gb=8.0
    )

    # 獲取優化器信息
    info = optimizer.get_optimization_info()
    print("啟用的正則化技術:")
    print(f"  - 邊緣感知正則化: {info['features']['edge_suppression']}")
    print(f"  - 空間感知正則化: {info['features']['spatial_awareness']}")
    print(f"  - 背景正則化: {info['features']['background_regularization']}")
    print(f"  - LoRA 低秩正則化: {info['features']['lora_rank_penalty']}")
    print(f"  - 動態自適應: {info['features']['dynamic_adaptation']}")

    # 訓練循環
    model.train()
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # 生成訓練數據
        lr_images, hr_images = generate_synthetic_data(
            batch_size=4,
            image_size=32,
            scale_factor=scale_factor,
            device=device
        )

        # 前向傳播
        optimizer.zero_grad()
        sr_images = model(lr_images)

        # 計算損失
        loss = F.mse_loss(sr_images, hr_images)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 輸出訓練信息
        with torch.no_grad():
            psnr = 20 * torch.log10(2.0 / torch.sqrt(F.mse_loss(sr_images, hr_images)))
            print(f"Loss: {loss.item():.4f}, PSNR: {psnr.item():.2f}dB")

        # 顯示記憶體統計
        memory_stats = optimizer.get_memory_stats()
        print(f"記憶體壓力: {memory_stats['memory_pressure']:.2%}")

    print("✅ 組合正則化技術訓練完成!")
    return model, optimizer

def compare_regularization_techniques():
    """
    比較不同正則化技術的效果
    """
    print("=" * 60)
    print("🔍 正則化技術比較")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 測試數據
    lr_images, hr_images = generate_synthetic_data(
        batch_size=4,
        image_size=32,
        scale_factor=4,
        device=device
    )

    techniques = [
        ("基礎優化器", {}),
        ("邊緣感知", {"edge_suppression": True, "edge_penalty": 0.1}),
        ("空間感知", {"spatial_awareness": True, "frequency_penalty": 0.05}),
        ("背景正則化", {"background_regularization": True}),
        ("組合技術", {
            "edge_suppression": True,
            "edge_penalty": 0.1,
            "spatial_awareness": True,
            "frequency_penalty": 0.05,
            "background_regularization": True,
        }),
    ]

    results = []

    for name, config in techniques:
        print(f"\n--- 測試 {name} ---")

        # 創建模型
        model = SuperResolutionModel(scale_factor=4).to(device)

        # 創建優化器
        optimizer_config = {
            "lr": 1e-4,
            "memory_efficient": True,
            "vram_budget_gb": 8.0
        }
        optimizer_config.update(config)

        optimizer = HinaAdaptive(model.parameters(), **optimizer_config)

        # 快速訓練
        model.train()
        initial_loss = None
        final_loss = None

        for step in range(10):
            optimizer.zero_grad()
            sr_images = model(lr_images)
            loss = F.mse_loss(sr_images, hr_images)

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if step == 9:
                final_loss = loss.item()

        # 計算 PSNR
        with torch.no_grad():
            sr_images = model(lr_images)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(F.mse_loss(sr_images, hr_images)))

        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        results.append((name, improvement, psnr.item()))

        print(f"改善率: {improvement:.1f}%, PSNR: {psnr.item():.2f}dB")

    # 顯示比較結果
    print("\n" + "=" * 60)
    print("📊 結果比較")
    print("=" * 60)

    for name, improvement, psnr in results:
        print(f"{name:<12}: 改善率 {improvement:>6.1f}%, PSNR {psnr:>5.2f}dB")

    return results

def main():
    """
    主函數 - 運行所有示例
    """
    print("🚀 HinaAdaptive 正則化技術示例")
    print("=" * 60)

    try:
        # 運行各種示例
        print("\n1. 邊緣感知正則化示例")
        train_with_edge_suppression(scale_factor=4)

        print("\n2. 空間感知正則化示例")
        train_with_spatial_awareness(scale_factor=4)

        print("\n3. 背景正則化示例")
        train_with_background_regularization(scale_factor=4)

        print("\n4. LoRA 低秩正則化示例")
        train_with_lora_regularization()

        print("\n5. 組合正則化技術示例")
        train_with_combined_regularization(scale_factor=4)

        print("\n6. 正則化技術比較")
        compare_regularization_techniques()

    except Exception as e:
        print(f"❌ 執行過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ 所有示例執行完成!")

if __name__ == "__main__":
    main()