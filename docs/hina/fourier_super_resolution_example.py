#!/usr/bin/env python3
"""
傅立葉特徵損失超解析度優化範例腳本

本腳本展示如何使用 HinaAdaptive 優化器的傅立葉特徵損失功能
來優化超解析度模型的訓練，特別針對細節保持和模糊抑制。

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
            nn.Tanh()
        )

    def forward(self, x):
        features = self.feature_extraction(x)
        residual = self.residual_blocks(features)
        output = self.upsampling(features + residual)
        return output

class ResidualBlock(nn.Module):
    """殘差塊"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

def create_synthetic_data(batch_size: int = 8, lr_size: int = 32, scale: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    創建合成的超解析度訓練數據

    Args:
        batch_size: 批次大小
        lr_size: 低解析度圖像大小
        scale: 放大倍數

    Returns:
        低解析度和高解析度圖像對
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建高解析度圖像（含有豐富細節）
    hr_size = lr_size * scale
    hr_images = torch.randn(batch_size, 3, hr_size, hr_size, device=device)

    # 添加一些紋理和邊緣
    for i in range(batch_size):
        # 添加網格紋理
        x = torch.arange(hr_size, device=device, dtype=torch.float32)
        y = torch.arange(hr_size, device=device, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # 創建不同頻率的正弦波紋理
        texture = (torch.sin(xx * 0.2) * torch.cos(yy * 0.15) +
                  torch.sin(xx * 0.5) * torch.cos(yy * 0.3) * 0.5)

        hr_images[i] += texture.unsqueeze(0) * 0.3

    # 創建低解析度圖像（下採樣）
    lr_images = F.interpolate(hr_images, size=(lr_size, lr_size), mode='bicubic', align_corners=False)

    # 正規化到 [-1, 1]
    hr_images = torch.tanh(hr_images)
    lr_images = torch.tanh(lr_images)

    return lr_images, hr_images

def compute_image_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    計算圖像品質指標

    Args:
        pred: 預測圖像
        target: 目標圖像

    Returns:
        包含各種指標的字典
    """
    with torch.no_grad():
        # PSNR
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))

        # 高頻能量比較
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)

        # 計算高頻能量
        h, w = pred.shape[-2:]
        freq_y = torch.fft.fftfreq(h, device=pred.device).unsqueeze(1)
        freq_x = torch.fft.fftfreq(w, device=pred.device).unsqueeze(0)
        freq_radius = torch.sqrt(freq_y**2 + freq_x**2)

        high_freq_mask = freq_radius > 0.3
        pred_hf_energy = torch.sum(pred_magnitude * high_freq_mask.float())
        target_hf_energy = torch.sum(target_magnitude * high_freq_mask.float())

        hf_preservation = pred_hf_energy / (target_hf_energy + 1e-8)

        return {
            'psnr': psnr.item(),
            'mse': mse.item(),
            'high_freq_preservation': hf_preservation.item()
        }

def train_with_fourier_loss(scale_factor: int = 4):
    """
    使用傅立葉特徵損失訓練超解析度模型

    Args:
        scale_factor: 超解析度放大倍數
    """
    print(f"🚀 開始使用傅立葉特徵損失訓練 {scale_factor}x 超解析度模型")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 創建模型
    model = SuperResolutionModel(scale_factor=scale_factor).to(device)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 創建傅立葉特徵損失優化器
    optimizer = HinaAdaptive(
        model.parameters(),
        lr=1e-4,
        # 啟用傅立葉特徵損失
        fourier_feature_loss=True,
        super_resolution_mode=True,
        super_resolution_scale=scale_factor,
        # 傅立葉特徵參數調整
        fourier_high_freq_preservation=0.3,     # 高頻細節保持
        fourier_detail_enhancement=0.25,        # 細節增強
        fourier_blur_suppression=0.2,           # 模糊抑制
        texture_coherence_penalty=0.1,          # 紋理一致性
        adaptive_frequency_weighting=True,      # 自適應頻率權重
        frequency_domain_lr_scaling=True,       # 頻域學習率縮放
        # 記憶體優化
        memory_efficient=True,
        vram_budget_gb=8.0,
        reduce_precision=True,
        # 其他功能
        use_dynamic_adaptation=True,
        edge_suppression=True,
        spatial_awareness=True
    )

    print("✅ HinaAdaptive 優化器配置:")
    optimizer_info = optimizer.get_optimization_info()
    for key, value in optimizer_info['fourier_super_resolution_config'].items():
        print(f"   {key}: {value}")

    print("\n🔧 訓練配置:")
    print(f"   放大倍數: {scale_factor}x")
    print(f"   學習率: {optimizer.defaults['lr']}")
    print(f"   記憶體預算: {optimizer.vram_budget_gb}GB")

    # 訓練循環
    model.train()
    training_history = {
        'losses': [],
        'psnr': [],
        'high_freq_preservation': []
    }

    print("\n🎯 開始訓練...")
    print("-" * 60)

    for epoch in range(20):
        epoch_losses = []
        epoch_metrics = {'psnr': [], 'high_freq_preservation': []}

        for step in range(10):  # 每個 epoch 10 個步驟
            # 生成訓練數據
            lr_images, hr_images = create_synthetic_data(batch_size=4, lr_size=32, scale=scale_factor)

            # 前向傳播
            sr_images = model(lr_images)

            # 計算損失
            loss = F.mse_loss(sr_images, hr_images)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 記錄指標
            epoch_losses.append(loss.item())

            if step % 5 == 0:  # 每 5 步計算一次詳細指標
                metrics = compute_image_metrics(sr_images, hr_images)
                epoch_metrics['psnr'].append(metrics['psnr'])
                epoch_metrics['high_freq_preservation'].append(metrics['high_freq_preservation'])

        # 統計本 epoch 結果
        avg_loss = np.mean(epoch_losses)
        avg_psnr = np.mean(epoch_metrics['psnr']) if epoch_metrics['psnr'] else 0
        avg_hf_preservation = np.mean(epoch_metrics['high_freq_preservation']) if epoch_metrics['high_freq_preservation'] else 0

        training_history['losses'].append(avg_loss)
        training_history['psnr'].append(avg_psnr)
        training_history['high_freq_preservation'].append(avg_hf_preservation)

        # 顯示進度
        print(f"Epoch {epoch+1:2d}/20 | "
              f"Loss: {avg_loss:.6f} | "
              f"PSNR: {avg_psnr:.2f}dB | "
              f"HF保持: {avg_hf_preservation:.3f}")

        # 每 5 個 epoch 顯示記憶體狀態
        if (epoch + 1) % 5 == 0:
            memory_stats = optimizer.get_memory_stats()
            print(f"   💾 記憶體壓力: {memory_stats['memory_pressure']:.1%}")

    print("\n✅ 訓練完成！")

    # 顯示最終結果
    final_loss = training_history['losses'][-1]
    final_psnr = training_history['psnr'][-1]
    final_hf_preservation = training_history['high_freq_preservation'][-1]

    print(f"\n📊 最終結果:")
    print(f"   最終損失: {final_loss:.6f}")
    print(f"   最終 PSNR: {final_psnr:.2f}dB")
    print(f"   高頻保持率: {final_hf_preservation:.3f}")

    # 改善分析
    initial_loss = training_history['losses'][0]
    initial_psnr = training_history['psnr'][0] if training_history['psnr'][0] > 0 else 20

    loss_improvement = (initial_loss - final_loss) / initial_loss * 100
    psnr_improvement = final_psnr - initial_psnr

    print(f"\n🎯 改善幅度:")
    print(f"   損失降低: {loss_improvement:.1f}%")
    print(f"   PSNR 提升: {psnr_improvement:.2f}dB")

    return training_history, optimizer

def compare_with_baseline(scale_factor: int = 4):
    """
    與基準優化器進行比較

    Args:
        scale_factor: 超解析度放大倍數
    """
    print(f"\n🔬 與基準優化器比較 ({scale_factor}x 超解析度)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    for optimizer_type in ['baseline_adam', 'hina_adaptive_fourier']:
        print(f"\n訓練使用: {optimizer_type}")

        # 創建新模型
        model = SuperResolutionModel(scale_factor=scale_factor).to(device)

        if optimizer_type == 'baseline_adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = HinaAdaptive(
                model.parameters(),
                lr=1e-4,
                fourier_feature_loss=True,
                super_resolution_mode=True,
                super_resolution_scale=scale_factor,
                fourier_high_freq_preservation=0.3,
                fourier_detail_enhancement=0.25,
                fourier_blur_suppression=0.2,
                memory_efficient=True
            )

        # 訓練
        model.train()
        final_metrics = {'loss': 0, 'psnr': 0, 'hf_preservation': 0}

        for epoch in range(10):  # 較短的比較訓練
            for step in range(5):
                lr_images, hr_images = create_synthetic_data(batch_size=4, lr_size=32, scale=scale_factor)

                sr_images = model(lr_images)
                loss = F.mse_loss(sr_images, hr_images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch == 9 and step == 4:  # 最後一步
                    metrics = compute_image_metrics(sr_images, hr_images)
                    final_metrics['loss'] = loss.item()
                    final_metrics['psnr'] = metrics['psnr']
                    final_metrics['hf_preservation'] = metrics['high_freq_preservation']

        results[optimizer_type] = final_metrics
        print(f"   最終 PSNR: {final_metrics['psnr']:.2f}dB")
        print(f"   高頻保持: {final_metrics['hf_preservation']:.3f}")

    # 比較結果
    print(f"\n📈 比較結果:")
    baseline = results['baseline_adam']
    fourier = results['hina_adaptive_fourier']

    psnr_improvement = fourier['psnr'] - baseline['psnr']
    hf_improvement = (fourier['hf_preservation'] - baseline['hf_preservation']) / baseline['hf_preservation'] * 100

    print(f"   PSNR 改善: {psnr_improvement:+.2f}dB")
    print(f"   高頻保持改善: {hf_improvement:+.1f}%")

    if psnr_improvement > 0:
        print("   ✅ 傅立葉特徵損失優化器表現更好！")
    else:
        print("   ⚠️  需要調整參數以獲得更好效果")

def visualize_frequency_analysis():
    """
    可視化頻率分析過程
    """
    print(f"\n🔍 傅立葉頻率分析可視化")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建測試圖像
    lr_images, hr_images = create_synthetic_data(batch_size=1, lr_size=64, scale=4)

    # 創建帶有傅立葉特徵的優化器
    model = SuperResolutionModel(scale_factor=4).to(device)
    optimizer = HinaAdaptive(
        model.parameters(),
        fourier_feature_loss=True,
        super_resolution_mode=True,
        super_resolution_scale=4
    )

    # 進行一次前向和反向傳播
    model.train()
    sr_images = model(lr_images)
    loss = F.mse_loss(sr_images, hr_images)

    optimizer.zero_grad()
    loss.backward()

    # 分析第一個卷積層的梯度
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is not None and first_conv.weight.grad is not None:
        grad = first_conv.weight.grad[0, 0]  # 取第一個通道的梯度

        # 計算傅立葉變換
        grad_fft = torch.fft.fft2(grad)
        magnitude = torch.abs(grad_fft)

        # 分析頻率分佈
        h, w = grad.shape
        freq_y = torch.fft.fftfreq(h, device=device).unsqueeze(1)
        freq_x = torch.fft.fftfreq(w, device=device).unsqueeze(0)
        freq_radius = torch.sqrt(freq_y**2 + freq_x**2)

        # 計算不同頻段的能量
        low_freq_mask = freq_radius <= 0.1
        mid_freq_mask = (freq_radius > 0.1) & (freq_radius <= 0.3)
        high_freq_mask = freq_radius > 0.3

        low_energy = torch.sum(magnitude * low_freq_mask.float()).item()
        mid_energy = torch.sum(magnitude * mid_freq_mask.float()).item()
        high_energy = torch.sum(magnitude * high_freq_mask.float()).item()

        total_energy = low_energy + mid_energy + high_energy

        print(f"頻率能量分佈:")
        print(f"   低頻 (≤0.1): {low_energy/total_energy*100:.1f}%")
        print(f"   中頻 (0.1-0.3): {mid_energy/total_energy*100:.1f}%")
        print(f"   高頻 (>0.3): {high_energy/total_energy*100:.1f}%")

        # 模糊指標
        blur_indicator = low_energy / (high_energy + 1e-8)
        print(f"   模糊指標: {blur_indicator:.2f} {'(偏模糊)' if blur_indicator > 2.0 else '(正常)'}")

def main():
    """主函數"""
    print("🎨 傅立葉特徵損失超解析度優化演示")
    print("=" * 60)

    try:
        # 1. 基本訓練演示
        print("📍 1. 基本訓練演示 (4x 超解析度)")
        train_with_fourier_loss(scale_factor=4)

        # 2. 不同放大倍數比較
        print("\n📍 2. 不同放大倍數比較")
        for scale in [2, 4, 8]:
            print(f"\n--- {scale}x 超解析度 ---")
            history, _ = train_with_fourier_loss(scale_factor=scale)
            final_psnr = history['psnr'][-1] if history['psnr'] else 0
            print(f"{scale}x 最終 PSNR: {final_psnr:.2f}dB")

        # 3. 與基準比較
        print("\n📍 3. 與基準優化器比較")
        compare_with_baseline(scale_factor=4)

        # 4. 頻率分析可視化
        print("\n📍 4. 頻率分析可視化")
        visualize_frequency_analysis()

        print("\n✅ 所有演示完成！")
        print("\n💡 使用建議:")
        print("   - 對於 2x 超解析度：使用較溫和的參數")
        print("   - 對於 4x 超解析度：平衡細節保持和穩定性")
        print("   - 對於 8x+ 超解析度：加強高頻保持和模糊抑制")
        print("   - 根據 VRAM 限制調整 memory_efficient 和 vram_budget_gb")

    except Exception as e:
        print(f"❌ 演示過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理資源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()