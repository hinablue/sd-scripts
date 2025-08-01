#!/usr/bin/env python3
"""
ANLO 優化器在 LoRA 訓練中的實際應用示例

這個腳本展示了如何在真實的 Stable Diffusion LoRA 訓練中使用 ANLO 優化器，
包括與現有訓練腳本的集成、參數配置、監控等功能。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import json

# 添加項目根目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from library.hina_ano import ANLO
from library.utils import setup_logging

# 設置日誌
setup_logging()
logger = logging.getLogger(__name__)


class MockLoRADataset(Dataset):
    """
    模擬 LoRA 訓練數據集
    """

    def __init__(self, num_samples: int = 1000, image_size: int = 512):
        self.num_samples = num_samples
        self.image_size = image_size

        # 模擬圖像數據（實際應用中會是真實的圖像）
        self.images = torch.randn(num_samples, 3, image_size, image_size)

        # 模擬文本提示
        self.prompts = [
            f"a beautiful landscape, high quality, detailed, step {i}"
            for i in range(num_samples)
        ]

        # 模擬負面提示
        self.negative_prompts = [
            "low quality, blurry, distorted, step {i}"
            for i in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'prompt': self.prompts[idx],
            'negative_prompt': self.negative_prompts[idx],
            'index': idx
        }


class MockStableDiffusionModel(nn.Module):
    """
    模擬 Stable Diffusion 模型，用於演示 ANLO 優化器的使用
    """

    def __init__(self, latent_dim: int = 64, text_dim: int = 768):
        super().__init__()

        # 模擬 UNet 的某些層
        self.unet_layers = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(12)
        ])

        # 模擬 Text Encoder 的某些層
        self.text_layers = nn.ModuleList([
            nn.Linear(text_dim, text_dim) for _ in range(6)
        ])

        # 模擬 VAE
        self.vae_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 3, padding=1)
        )

        self.vae_decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def encode_text(self, text_embeddings):
        """模擬文本編碼"""
        x = text_embeddings
        for layer in self.text_layers:
            x = layer(x)
        return x

    def encode_image(self, images):
        """模擬圖像編碼"""
        return self.vae_encoder(images)

    def decode_latent(self, latents):
        """模擬潛在空間解碼"""
        return self.vae_decoder(latents)

    def forward(self, images, text_embeddings, timesteps):
        """前向傳播"""
        # 編碼圖像
        latents = self.encode_image(images)

        # 編碼文本
        text_features = self.encode_text(text_embeddings)

        # 模擬 UNet 處理
        x = latents.view(latents.size(0), -1)  # 展平
        for layer in self.unet_layers:
            x = layer(x)

        # 添加文本條件
        x = x + text_features.mean(dim=1, keepdim=True)

        # 解碼
        x = x.view(latents.size(0), latents.size(1), latents.size(2), latents.size(3))
        output = self.decode_latent(x)

        return output


class LoRATrainer:
    """
    LoRA 訓練器，使用 ANLO 優化器
    """

    def __init__(
        self,
        model: MockStableDiffusionModel,
        train_dataset: MockLoRADataset,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config

        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 創建數據加載器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=True,
            num_workers=config.get('num_workers', 2)
        )

        # 準備優化器參數
        self.optimizer_params = self._prepare_optimizer_params()

        # 創建 ANLO 優化器
        self.optimizer = ANLO(
            params=self.optimizer_params,
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-2),
            normalize_frequency=config.get('normalize_frequency', 1),
            global_norm_weight=config.get('global_norm_weight', 1.0),
            layer_norm_weight=config.get('layer_norm_weight', 1.0),
            adaptive_eps=config.get('adaptive_eps', True),
            verbose=config.get('verbose', True)
        )

        # 創建學習率調度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * config.get('epochs', 10)
        )

        # 訓練狀態
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')

        # 記錄訓練歷史
        self.training_history = {
            'losses': [],
            'learning_rates': [],
            'memory_usage': [],
            'normalization_stats': []
        }

        logger.info(f"LoRA 訓練器初始化完成，使用 ANLO 優化器")
        logger.info(f"設備: {self.device}")
        logger.info(f"批次大小: {config.get('batch_size', 4)}")
        logger.info(f"學習率: {config.get('learning_rate', 1e-4)}")

    def _prepare_optimizer_params(self) -> List[Dict[str, Any]]:
        """
        準備優化器參數，模擬 LoRA 網絡的參數組織方式
        """
        param_groups = []

        # 第一組：UNet LoRA 參數
        unet_params = []
        unet_names = []
        for i, layer in enumerate(self.model.unet_layers):
            unet_params.extend(layer.parameters())
            unet_names.extend([f'unet_layer_{i}_weight', f'unet_layer_{i}_bias'])

        param_groups.append({
            'params': unet_params,
            'named': unet_names,
            'lr': self.config.get('unet_lr', 1e-4)
        })

        # 第二組：Text Encoder LoRA 參數
        text_params = []
        text_names = []
        for i, layer in enumerate(self.model.text_layers):
            text_params.extend(layer.parameters())
            text_names.extend([f'text_layer_{i}_weight', f'text_layer_{i}_bias'])

        param_groups.append({
            'params': text_params,
            'named': text_names,
            'lr': self.config.get('text_encoder_lr', 1e-5)
        })

        # 第三組：VAE 參數
        vae_params = []
        vae_names = []

        # VAE 編碼器參數
        for i, layer in enumerate(self.model.vae_encoder):
            if hasattr(layer, 'weight'):
                vae_params.append(layer.weight)
                vae_names.append(f'vae_encoder_{i}_weight')
            if hasattr(layer, 'bias') and layer.bias is not None:
                vae_params.append(layer.bias)
                vae_names.append(f'vae_encoder_{i}_bias')

        # VAE 解碼器參數
        for i, layer in enumerate(self.model.vae_decoder):
            if hasattr(layer, 'weight'):
                vae_params.append(layer.weight)
                vae_names.append(f'vae_decoder_{i}_weight')
            if hasattr(layer, 'bias') and layer.bias is not None:
                vae_params.append(layer.bias)
                vae_names.append(f'vae_decoder_{i}_bias')

        param_groups.append({
            'params': vae_params,
            'named': vae_names,
            'lr': self.config.get('vae_lr', 1e-4)
        })

        return param_groups

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        計算訓練損失
        """
        images = batch['image'].to(self.device)

        # 模擬文本嵌入
        batch_size = images.size(0)
        text_embeddings = torch.randn(batch_size, 77, 768).to(self.device)  # 模擬 CLIP 嵌入

        # 模擬時間步
        timesteps = torch.randint(0, 1000, (batch_size,)).to(self.device)

        # 前向傳播
        output = self.model(images, text_embeddings, timesteps)

        # 計算重建損失
        reconstruction_loss = nn.MSELoss()(output, images)

        # 模擬其他損失（如 KL 散度、感知損失等）
        kl_loss = torch.mean(torch.sum(output ** 2, dim=[1, 2, 3]))

        # 總損失
        total_loss = reconstruction_loss + 0.01 * kl_loss

        return total_loss

    def _log_training_info(self, loss: float, step: int):
        """
        記錄訓練信息
        """
        # 記錄損失
        self.training_history['losses'].append(loss)

        # 記錄學習率
        current_lr = self.optimizer.get_lr()[0]  # 取第一個參數組的學習率
        self.training_history['learning_rates'].append(current_lr)

        # 記錄記憶體使用
        memory_info = self.optimizer.get_memory_usage()
        self.training_history['memory_usage'].append(memory_info)

        # 記錄正規化統計
        norm_stats = self.optimizer.get_normalization_stats()
        self.training_history['normalization_stats'].append(norm_stats)

        # 每 100 步輸出一次詳細信息
        if step % 100 == 0:
            logger.info(f"Step {step}: Loss = {loss:.6f}, LR = {current_lr:.2e}")
            logger.info(f"  記憶體使用: {memory_info['total_memory_mb']:.2f} MB")
            logger.info(f"  正規化模式: {norm_stats['normalization_mode']}")

    def train_epoch(self) -> float:
        """
        訓練一個 epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        logger.info(f"開始訓練 Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(self.train_loader):
            # 前向傳播和損失計算
            loss = self._compute_loss(batch)

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()

            # 優化器步驟
            self.optimizer.step()

            # 學習率調度
            self.scheduler.step()

            # 更新計數器
            self.current_step += 1

            # 記錄損失
            epoch_loss += loss.item()

            # 記錄訓練信息
            self._log_training_info(loss.item(), self.current_step)

            # 進度輸出
            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                logger.info(f"  Progress: {progress:.1f}% ({batch_idx + 1}/{num_batches})")

        # 計算平均損失
        avg_loss = epoch_loss / num_batches

        logger.info(f"Epoch {self.current_epoch + 1} 完成，平均損失: {avg_loss:.6f}")

        return avg_loss

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        完整訓練流程
        """
        logger.info(f"開始訓練，總共 {num_epochs} 個 epochs")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 訓練一個 epoch
            epoch_loss = self.train_epoch()

            # 檢查是否是最佳模型
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                logger.info(f"新的最佳損失: {self.best_loss:.6f}")

            # 每 5 個 epochs 保存一次檢查點
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # 計算總訓練時間
        total_time = time.time() - start_time

        # 保存最終模型
        self.save_checkpoint("final_model.pt")

        # 保存訓練歷史
        self.save_training_history()

        logger.info(f"訓練完成！總時間: {total_time:.2f} 秒")
        logger.info(f"最佳損失: {self.best_loss:.6f}")

        return {
            'best_loss': self.best_loss,
            'total_time': total_time,
            'training_history': self.training_history
        }

    def save_checkpoint(self, filename: str):
        """
        保存檢查點
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_loss': self.best_loss,
            'config': self.config
        }

        torch.save(checkpoint, filename)
        logger.info(f"檢查點已保存: {filename}")

    def save_training_history(self):
        """
        保存訓練歷史
        """
        # 轉換為可序列化的格式
        history = {}
        for key, value in self.training_history.items():
            if key == 'memory_usage':
                # 簡化記憶體使用信息
                history[key] = [
                    {
                        'total_memory_mb': item['total_memory_mb'],
                        'parameter_memory_mb': item['parameter_memory_mb'],
                        'gradient_memory_mb': item['gradient_memory_mb']
                    }
                    for item in value
                ]
            elif key == 'normalization_stats':
                # 簡化正規化統計信息
                history[key] = [
                    {
                        'step_count': item['step_count'],
                        'normalization_mode': item['normalization_mode']
                    }
                    for item in value
                ]
            else:
                history[key] = value

        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        logger.info("訓練歷史已保存: training_history.json")


def main():
    """
    主函數：演示 ANLO 優化器在 LoRA 訓練中的使用
    """
    print("ANLO 優化器 LoRA 訓練示例")
    print("=" * 50)

    # 配置參數
    config = {
        'batch_size': 4,
        'num_workers': 2,
        'epochs': 10,
        'learning_rate': 1e-4,
        'unet_lr': 1e-4,
        'text_encoder_lr': 1e-5,
        'vae_lr': 1e-4,
        'weight_decay': 1e-2,
        'normalize_frequency': 1,
        'global_norm_weight': 1.0,
        'layer_norm_weight': 1.0,
        'adaptive_eps': True,
        'verbose': True
    }

    # 創建模型和數據集
    model = MockStableDiffusionModel()
    dataset = MockLoRADataset(num_samples=1000)

    # 創建訓練器
    trainer = LoRATrainer(model, dataset, config)

    # 開始訓練
    results = trainer.train(num_epochs=config['epochs'])

    # 輸出結果摘要
    print("\n" + "=" * 50)
    print("訓練結果摘要:")
    print(f"最佳損失: {results['best_loss']:.6f}")
    print(f"總訓練時間: {results['total_time']:.2f} 秒")
    print(f"平均每 epoch 時間: {results['total_time'] / config['epochs']:.2f} 秒")

    # 分析記憶體使用
    memory_usage = results['training_history']['memory_usage']
    if memory_usage:
        avg_memory = sum(item['total_memory_mb'] for item in memory_usage) / len(memory_usage)
        max_memory = max(item['total_memory_mb'] for item in memory_usage)
        print(f"平均記憶體使用: {avg_memory:.2f} MB")
        print(f"最大記憶體使用: {max_memory:.2f} MB")

    print("\n訓練完成！檢查以下文件:")
    print("- final_model.pt: 最終模型")
    print("- training_history.json: 訓練歷史")
    print("- checkpoint_epoch_*.pt: 檢查點文件")


if __name__ == "__main__":
    main()