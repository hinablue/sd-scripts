import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Optional, Union
import math
import array
import random
import concurrent.futures
from threading import Thread
from collections import defaultdict
import time

from library.utils import setup_logging

setup_logging()
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ANLO(torch.optim.Optimizer):
    """
    Alternating Norm LoRA Optimizer (ANLO)

    基於「梯度多重正規化」概念的無狀態優化器，專為 LoRA 訓練設計。

    核心特點：
    1. 無狀態設計：不儲存動量緩衝區，記憶體佔用與 SGD 相同
    2. 交替正規化：全局正規化與層級正規化交替進行
    3. 自適應性：通過正規化實現類似 Adam 的自適應效果
    4. 記憶體高效：極致的 VRAM 節省
    5. 謹慎更新：整合對齊度檢測，提高訓練穩定性

    設計原理：
    - 全局正規化：控制整體更新步長，防止梯度爆炸
    - 層級正規化：平衡不同層之間的學習速度
    - 交替機制：模擬自適應優化器的動態調整效果
    - 謹慎更新：檢測梯度與更新方向的對齊度，防止不一致更新
    """

    def __init__(
        self,
        params: Union[torch.Tensor, List[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-4,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        normalize_frequency: int = 1,
        global_norm_weight: float = 1.0,
        layer_norm_weight: float = 1.0,
        adaptive_eps: bool = True,
        use_cautious_update: bool = True,
        cautious_threshold: float = 0.1,
        cautious_scale: float = 0.5,
        verbose: bool = False,
        **kwargs
    ):
        """
        初始化 ANLO 優化器

        Args:
            params: 要優化的參數
            lr: 學習率
            eps: 數值穩定性常數
            weight_decay: 權重衰減係數
            normalize_frequency: 正規化頻率（每 N 步進行一次正規化）
            global_norm_weight: 全局正規化權重
            layer_norm_weight: 層級正規化權重
            adaptive_eps: 是否使用自適應 eps
            use_cautious_update: 是否啟用謹慎更新策略
            cautious_threshold: 謹慎更新對齊度閾值
            cautious_scale: 謹慎更新縮放因子
            verbose: 是否輸出詳細信息
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0 < normalize_frequency:
            raise ValueError(f"Invalid normalize_frequency: {normalize_frequency}")
        if not 0.0 <= cautious_threshold <= 1.0:
            raise ValueError(f"Invalid cautious_threshold: {cautious_threshold}")
        if not 0.0 <= cautious_scale <= 1.0:
            raise ValueError(f"Invalid cautious_scale: {cautious_scale}")

        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            normalize_frequency=normalize_frequency,
            global_norm_weight=global_norm_weight,
            layer_norm_weight=layer_norm_weight,
            adaptive_eps=adaptive_eps,
            use_cautious_update=use_cautious_update,
            cautious_threshold=cautious_threshold,
            cautious_scale=cautious_scale,
            verbose=verbose
        )

        super().__init__(params, defaults)

        # 初始化步數計數器
        self.step_count = 0

        if verbose:
            logger.info(f"ANLO Optimizer initialized with {len(self.param_groups)} parameter groups")
            logger.info(f"Total parameters: {sum(p.numel() for group in self.param_groups for p in group['params'])}")
            logger.info(f"Memory efficient: No momentum buffers required")
            logger.info(f"Cautious update: {'Enabled' if use_cautious_update else 'Disabled'}")

    @staticmethod
    @torch.jit.script
    def _compute_global_norm(params: List[torch.Tensor]) -> torch.Tensor:
        """
        計算所有參數梯度的全局 L2 範數

        Args:
            params: 參數列表

        Returns:
            全局 L2 範數
        """
        grad_norms = []
        for param in params:
            if param.grad is not None:
                grad_norms.append(param.grad.view(-1))

        if not grad_norms:
            return torch.tensor(0.0, device=params[0].device, dtype=params[0].dtype)

        # 連接所有梯度並計算 L2 範數
        all_grads = torch.cat(grad_norms)
        global_norm = torch.norm(all_grads, p=2)

        return global_norm

    @staticmethod
    @torch.jit.script
    def _compute_layer_norm(param_group: Dict[str, Any]) -> torch.Tensor:
        """
        計算單個參數組內所有梯度的 L2 範數

        Args:
            param_group: 參數組字典

        Returns:
            層級 L2 範數
        """
        grad_norms = []
        for param in param_group['params']:
            if param.grad is not None:
                grad_norms.append(param.grad.view(-1))

        if not grad_norms:
            return torch.tensor(0.0, device=param_group['params'][0].device, dtype=param_group['params'][0].dtype)

        # 連接該組內所有梯度並計算 L2 範數
        group_grads = torch.cat(grad_norms)
        layer_norm = torch.norm(group_grads, p=2)

        return layer_norm

    @staticmethod
    @torch.jit.script
    def _adaptive_eps(group: Dict[str, Any], step: int) -> float:
        """
        計算自適應 eps 值

        Args:
            group: 參數組配置
            step: 當前步數

        Returns:
            自適應 eps 值
        """
        if not group['adaptive_eps']:
            return group['eps']

        # 基於步數動態調整 eps，防止早期訓練不穩定
        base_eps = group['eps']
        warmup_factor = min(1.0, step / 1000.0)  # 前 1000 步逐漸增加穩定性
        adaptive_eps = base_eps * (1.0 + 9.0 * (1.0 - warmup_factor))

        return adaptive_eps

    @staticmethod
    @torch.jit.script
    def _apply_cautious_update_optimized(update: torch.Tensor, grad: torch.Tensor,
                                       threshold: float = 0.1, scale: float = 0.5) -> torch.Tensor:
        """
        應用謹慎更新策略（JIT 優化版本）

        檢查更新向量與梯度的對齊度，當對齊度低於閾值時縮放更新步長。
        這有助於防止梯度方向不一致導致的訓練不穩定。

        Args:
            update: 更新向量
            grad: 梯度向量
            threshold: 對齊度閾值
            scale: 縮放因子

        Returns:
            調整後的更新向量
        """
        update_flat = update.view(-1)
        grad_flat = grad.view(-1)

        update_norm = torch.norm(update_flat)
        grad_norm = torch.norm(grad_flat)

        if update_norm > 0 and grad_norm > 0:
            alignment = torch.dot(update_flat, grad_flat) / (update_norm * grad_norm)
            if alignment < threshold:
                return update * scale

        return update

    def _apply_global_normalization(self, params: List[torch.Tensor], group: Dict[str, Any]) -> None:
        """
        應用全局正規化

        Args:
            params: 參數列表
            group: 參數組配置
        """
        global_norm = self._compute_global_norm(params)
        eps = group['eps']
        weight = group['global_norm_weight']

        if global_norm > 0:
            # 計算正規化係數
            norm_factor = weight / (global_norm + eps)

            # 對所有參數的梯度進行正規化
            for param in params:
                if param.grad is not None:
                    param.grad.mul_(norm_factor)

    def _apply_layer_normalization(self, param_group: Dict[str, Any]) -> None:
        """
        應用層級正規化

        Args:
            param_group: 參數組字典
        """
        layer_norm = self._compute_layer_norm(param_group)
        eps = param_group['eps']
        weight = param_group['layer_norm_weight']

        if layer_norm > 0:
            # 計算正規化係數
            norm_factor = weight / (layer_norm + eps)

            # 對該組內所有參數的梯度進行正規化
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.mul_(norm_factor)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        執行優化步驟

        Args:
            closure: 可選的閉包函數，用於重新計算損失

        Returns:
            損失值（如果提供了 closure）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 更新步數計數器
        self.step_count += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            normalize_frequency = group['normalize_frequency']
            verbose = group['verbose']
            use_cautious_update = group['use_cautious_update']
            cautious_threshold = group['cautious_threshold']
            cautious_scale = group['cautious_scale']

            # 獲取自適應 eps
            adaptive_eps = self._adaptive_eps(group, self.step_count)

            # 決定是否進行正規化
            should_normalize = (self.step_count % normalize_frequency == 0)

            if should_normalize:
                # 交替正規化決策
                if self.step_count % 2 == 0:
                    # 偶數步：全局正規化
                    if verbose and self.step_count % 100 == 0:
                        logger.info(f"Step {self.step_count}: Applying global normalization")

                    # 收集所有參數進行全局正規化
                    all_params = []
                    for g in self.param_groups:
                        all_params.extend(g['params'])

                    self._apply_global_normalization(all_params, group)

                else:
                    # 奇數步：層級正規化
                    if verbose and self.step_count % 100 == 0:
                        logger.info(f"Step {self.step_count}: Applying layer normalization")

                    self._apply_layer_normalization(group)

            # 參數更新
            for param in group['params']:
                if param.grad is not None:
                    # 計算更新向量
                    update = param.grad.clone()

                    # 應用謹慎更新
                    if use_cautious_update:
                        update = self._apply_cautious_update_optimized(update, param.grad, cautious_threshold, cautious_scale)

                    # 應用更新
                    param.add_(update, alpha=-lr)
                    param.grad.zero_()

            # 權重衰減（在參數更新之後直接應用）
            if weight_decay != 0:
                for param in group['params']:
                    param.data.add_(param.data, alpha=-lr * weight_decay)

        return loss

    def get_lr(self) -> List[float]:
        """
        獲取當前學習率

        Returns:
            學習率列表
        """
        return [group['lr'] for group in self.param_groups]

    def set_lr(self, lr: Union[float, List[float]]) -> None:
        """
        設置學習率

        Args:
            lr: 新的學習率或學習率列表
        """
        if isinstance(lr, (int, float)):
            lr = [lr] * len(self.param_groups)

        for group, new_lr in zip(self.param_groups, lr):
            group['lr'] = new_lr

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        獲取記憶體使用情況

        Returns:
            記憶體使用信息
        """
        total_params = 0
        total_grads = 0

        for group in self.param_groups:
            for param in group['params']:
                total_params += param.numel()
                if param.grad is not None:
                    total_grads += param.grad.numel()

        # ANLO 是無狀態的，只計算參數和梯度佔用
        param_memory = total_params * 4  # 假設 float32
        grad_memory = total_grads * 4

        return {
            'total_parameters': total_params,
            'parameter_memory_mb': param_memory / (1024 * 1024),
            'gradient_memory_mb': grad_memory / (1024 * 1024),
            'optimizer_state_memory_mb': 0.0,  # 無狀態設計
            'total_memory_mb': (param_memory + grad_memory) / (1024 * 1024)
        }

    def get_normalization_stats(self) -> Dict[str, Any]:
        """
        獲取正規化統計信息

        Returns:
            正規化統計信息
        """
        stats = {
            'step_count': self.step_count,
            'normalization_mode': 'global' if self.step_count % 2 == 0 else 'layer',
            'cautious_update_enabled': any(group['use_cautious_update'] for group in self.param_groups),
            'param_groups': {}
        }

        for group_idx, group in enumerate(self.param_groups):
            group_stats = {
                'param_count': len(group['params']),
                'total_params': sum(p.numel() for p in group['params']),
                'learning_rate': group['lr'],
                'eps': group['eps'],
                'cautious_update': group['use_cautious_update'],
                'cautious_threshold': group['cautious_threshold'],
                'cautious_scale': group['cautious_scale']
            }

            # 計算當前梯度統計
            grad_norms = []
            for param in group['params']:
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad, p=2).item())

            if grad_norms:
                group_stats['grad_norm_mean'] = sum(grad_norms) / len(grad_norms)
                group_stats['grad_norm_max'] = max(grad_norms)
                group_stats['grad_norm_min'] = min(grad_norms)

            stats['param_groups'][f'group_{group_idx}'] = group_stats

        return stats

    def get_cautious_update_stats(self) -> Dict[str, Any]:
        """
        獲取謹慎更新統計信息

        Returns:
            謹慎更新統計信息
        """
        stats = {
            'enabled_groups': 0,
            'total_groups': len(self.param_groups),
            'cautious_update_config': {}
        }

        for group_idx, group in enumerate(self.param_groups):
            if group['use_cautious_update']:
                stats['enabled_groups'] += 1

            stats['cautious_update_config'][f'group_{group_idx}'] = {
                'enabled': group['use_cautious_update'],
                'threshold': group['cautious_threshold'],
                'scale': group['cautious_scale']
            }

        return stats
