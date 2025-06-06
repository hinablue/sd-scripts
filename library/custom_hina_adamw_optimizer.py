"""
Hina Custom AdamW Optimizer with Enhanced Features
基於 AdamW8bit 的增強優化器，整合多種先進優化技術

This optimizer integrates multiple state-of-the-art optimization techniques:
- Generalization enhancement (SPD, Cautious Optimizer, Orthogonal Gradient)
- Adaptive learning rates (ADOPT, Grams, AGR, TAM)
- LoRA-specific optimizations (ALoRA, dynamic weight decay)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import re
from collections import defaultdict

# Import bitsandbytes optimizer
try:
    import bitsandbytes as bnb
    from bitsandbytes.optim import AdamW8bit
except ImportError:
    raise ImportError("bitsandbytes is required for this optimizer")

from .utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)


class HinaAdamWOptimizer(AdamW8bit):
    """
    Hina Custom AdamW Optimizer with Enhanced Features

    基於 AdamW8bit 的增強優化器，整合以下功能：

    1. 避免過擬合的技術：
       - SPD (Selective Projection Decay)
       - Cautious Optimizer 機制
       - Orthogonal Gradient 投影

    2. 自適應學習率調整：
       - ADOPT 穩定性機制
       - Grams 自適應動量縮放
       - AGR (Adaptive Gradient Regularization)
       - TAM (Torque-Aware Momentum)

    3. LoRA 專屬優化：
       - ALoRA 風格自適應學習率
       - 動態權重衰減
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        optim_bits: int = 32,
        args: Any = None,
        min_8bit_size: int = 4096,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        is_paged: bool = False,
        # Enhanced features configuration
        use_spd: bool = True,
        spd_lambda: float = 0.1,
        use_cautious: bool = True,
        use_orthogonal_grad: bool = False,
        use_adopt_stability: bool = True,
        use_grams: bool = True,
        use_agr: bool = True,
        use_tam: bool = True,
        tam_beta: float = 0.999,
        use_alora: bool = True,
        alora_ratio: float = 21.0,
        dynamic_weight_decay: bool = True,
        **kwargs
    ):
        """
        Args:
            params: 模型參數
            lr: 學習率
            betas: Adam 的 beta 參數
            eps: 數值穩定性常數
            weight_decay: 權重衰減係數
            use_spd: 啟用 Selective Projection Decay
            spd_lambda: SPD 懲罰強度
            use_cautious: 啟用謹慎優化器機制
            use_orthogonal_grad: 啟用正交梯度投影
            use_adopt_stability: 啟用 ADOPT 穩定性機制
            use_grams: 啟用 Grams 自適應動量縮放
            use_agr: 啟用自適應梯度正則化
            use_tam: 啟用 Torque-Aware Momentum
            tam_beta: TAM 的 beta 參數
            use_alora: 啟用 ALoRA 風格學習率
            alora_ratio: ALoRA 學習率比例（ηB/ηA）
            dynamic_weight_decay: 啟用動態權重衰減
        """
        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, optim_bits=optim_bits, args=args,
            min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
            block_wise=block_wise, is_paged=is_paged, **kwargs
        )

        # Enhanced features configuration
        self.use_spd = use_spd
        self.spd_lambda = spd_lambda
        self.use_cautious = use_cautious
        self.use_orthogonal_grad = use_orthogonal_grad
        self.use_adopt_stability = use_adopt_stability
        self.use_grams = use_grams
        self.use_agr = use_agr
        self.use_tam = use_tam
        self.tam_beta = tam_beta
        self.use_alora = use_alora
        self.alora_ratio = alora_ratio
        self.dynamic_weight_decay = dynamic_weight_decay

        # Initialize parameter groups and metadata
        self._initialize_parameter_groups()

        # Store initial parameters for SPD
        if self.use_spd:
            self._store_initial_parameters()

        logger.info(f"CustomAdamWOptimizer initialized with enhanced features:")
        logger.info(f"  SPD: {use_spd}, Cautious: {use_cautious}, Orthogonal Grad: {use_orthogonal_grad}")
        logger.info(f"  ADOPT: {use_adopt_stability}, Grams: {use_grams}, AGR: {use_agr}")
        logger.info(f"  TAM: {use_tam}, ALoRA: {use_alora}, Dynamic WD: {dynamic_weight_decay}")

    def _initialize_parameter_groups(self):
        """初始化參數分組，識別 LoRA 參數和其他參數類型"""
        self.param_groups_metadata = {}

        for group_idx, group in enumerate(self.param_groups):
            group_metadata = {
                'lora_a_params': [],
                'lora_b_params': [],
                'lora_pairs': {},  # 存儲 A-B 配對
                'regular_params': [],
                'param_names': {},
                'param_shapes': {}
            }

            # 分析參數名稱和形狀
            for param_idx, param in enumerate(group['params']):
                if hasattr(param, 'param_name'):
                    param_name = param.param_name
                else:
                    param_name = f"param_{group_idx}_{param_idx}"

                group_metadata['param_names'][param] = param_name
                group_metadata['param_shapes'][param] = param.shape

                # 識別 LoRA 參數
                if 'lora_down' in param_name or 'lora_A' in param_name:
                    group_metadata['lora_a_params'].append(param)
                elif 'lora_up' in param_name or 'lora_B' in param_name:
                    group_metadata['lora_b_params'].append(param)
                else:
                    group_metadata['regular_params'].append(param)

            # 建立 LoRA A-B 配對
            self._pair_lora_parameters(group_metadata)

            self.param_groups_metadata[group_idx] = group_metadata

    def _pair_lora_parameters(self, group_metadata):
        """建立 LoRA A 和 B 參數的配對"""
        lora_a_params = group_metadata['lora_a_params']
        lora_b_params = group_metadata['lora_b_params']
        param_names = group_metadata['param_names']

        for a_param in lora_a_params:
            a_name = param_names[a_param]
            # 尋找對應的 B 參數
            base_name = a_name.replace('lora_down', '').replace('lora_A', '').replace('.weight', '')

            for b_param in lora_b_params:
                b_name = param_names[b_param]
                b_base_name = b_name.replace('lora_up', '').replace('lora_B', '').replace('.weight', '')

                if base_name == b_base_name:
                    group_metadata['lora_pairs'][a_param] = b_param
                    break

    def _store_initial_parameters(self):
        """存儲初始參數以供 SPD 使用"""
        self.initial_params = {}
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param.requires_grad:
                    self.initial_params[param] = param.data.clone().detach()

    def _apply_spd_regularization(self, param, group, state):
        """應用 Selective Projection Decay 正則化"""
        if param not in self.initial_params:
            return 0

        initial_param = self.initial_params[param]
        param_diff = param.data - initial_param

        # 計算偏差比率 rt
        param_norm = torch.norm(param.data)
        diff_norm = torch.norm(param_diff)

        if param_norm > 0:
            bias_ratio = diff_norm / param_norm
        else:
            bias_ratio = 0

        # SPD 懲罰項
        spd_penalty = self.spd_lambda * bias_ratio * param_diff
        return spd_penalty

    def _apply_orthogonal_gradient(self, grad, param):
        """應用正交梯度投影"""
        if param.dim() < 2:
            return grad

        param_flat = param.view(-1)
        grad_flat = grad.view(-1)

        # 計算正交投影
        dot_product = torch.dot(param_flat, grad_flat)
        param_norm_sq = torch.dot(param_flat, param_flat)

        if param_norm_sq > 0:
            projection = (dot_product / param_norm_sq) * param_flat
            orthogonal_grad = grad_flat - projection
            return orthogonal_grad.view(grad.shape)

        return grad

    def _apply_agr_regularization(self, grad):
        """應用自適應梯度正則化"""
        if grad.dim() == 0:
            return grad

        # 計算梯度向量的總和歸一化係數
        grad_sum = torch.sum(grad)
        grad_norm = torch.norm(grad)

        if grad_norm > 0:
            agr_coeff = torch.abs(grad_sum) / grad_norm
            return grad * (1.0 + agr_coeff)

        return grad

    def _apply_cautious_update(self, update, grad):
        """應用謹慎優化器的對齊檢查"""
        # 計算對齊遮罩：僅當更新方向與梯度對齊時才應用
        alignment = update * grad
        alignment_mask = (alignment > 0).float()
        return update * alignment_mask

    def _compute_alora_lr_scale(self, lora_a_param, lora_b_param):
        """計算 ALoRA 風格的學習率縮放因子"""
        try:
            # 計算 BA 矩陣
            if lora_a_param.dim() == 2 and lora_b_param.dim() == 2:
                ba_matrix = torch.matmul(lora_b_param, lora_a_param)
            else:
                # 處理卷積層的情況
                ba_matrix = torch.matmul(
                    lora_b_param.view(lora_b_param.size(0), -1),
                    lora_a_param.view(lora_a_param.size(0), -1)
                )

            # 計算行向量的 L2 範數
            row_norms = torch.norm(ba_matrix, dim=1)
            avg_row_norm = torch.mean(row_norms)

            # 自適應學習率與範數成反比
            if avg_row_norm > 0:
                lr_scale = 1.0 / (1.0 + avg_row_norm)
            else:
                lr_scale = 1.0

            return lr_scale.item()
        except Exception as e:
            logger.warning(f"Failed to compute ALoRA lr scale: {e}")
            return 1.0

    def _apply_tam_damping(self, momentum, grad, state):
        """應用 Torque-Aware Momentum 阻尼"""
        if 'momentum_alignment' not in state:
            state['momentum_alignment'] = 0.0

        # 計算梯度與動量的對齊程度
        if torch.norm(momentum) > 0 and torch.norm(grad) > 0:
            alignment = torch.dot(momentum.view(-1), grad.view(-1)) / (
                torch.norm(momentum) * torch.norm(grad)
            )
        else:
            alignment = 0.0

        # 平滑對齊估計
        state['momentum_alignment'] = (
            self.tam_beta * state['momentum_alignment'] +
            (1 - self.tam_beta) * alignment
        )

        # 計算阻尼因子
        damping_factor = (1 + state['momentum_alignment']) / 2
        return damping_factor

    def step(self, closure=None):
        """執行優化步驟"""
        loss = None
        if closure is not None:
            loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            group_metadata = self.param_groups_metadata[group_idx]

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CustomAdamWOptimizer does not support sparse gradients')

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)
                    if self.use_adopt_stability:
                        state['exp_avg_sq_prev'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if self.use_adopt_stability and 'exp_avg_sq_prev' in state:
                    exp_avg_sq_prev = state['exp_avg_sq_prev']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # 應用正交梯度投影
                if self.use_orthogonal_grad:
                    grad = self._apply_orthogonal_gradient(grad, param)

                # 應用自適應梯度正則化
                if self.use_agr:
                    grad = self._apply_agr_regularization(grad)

                # ADOPT 穩定性：使用前一步的二階矩進行動量歸一化
                if self.use_adopt_stability:
                    # 更新二階矩（移除當前梯度）
                    exp_avg_sq_prev.copy_(exp_avg_sq)
                    exp_avg_sq.mul_(beta2).addcmulsq_(grad, grad, value=1 - beta2)

                    # 使用前一步的二階矩進行動量更新
                    if state['step'] > 1:
                        denom = exp_avg_sq_prev.sqrt().add_(group['eps'])
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    # 標準 Adam 更新
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmulsq_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # 計算基礎更新
                update = exp_avg / denom

                # TAM 阻尼
                if self.use_tam:
                    damping_factor = self._apply_tam_damping(exp_avg, grad, state)
                    update = update * damping_factor

                # Grams 自適應幅度縮放
                if self.use_grams:
                    update = torch.abs(update) * torch.sign(grad)

                # 謹慎優化器機制
                if self.use_cautious:
                    update = self._apply_cautious_update(update, grad)

                # ALoRA 學習率調整
                current_step_size = step_size
                if self.use_alora and param in group_metadata.get('lora_pairs', {}):
                    paired_param = group_metadata['lora_pairs'][param]
                    lr_scale = self._compute_alora_lr_scale(param, paired_param)
                    current_step_size *= lr_scale
                elif self.use_alora and param in group_metadata.get('lora_b_params', []):
                    # 對 B 參數應用比例調整
                    current_step_size *= self.alora_ratio

                # 應用更新
                param.data.add_(update, alpha=-current_step_size)

                # 權重衰減
                current_weight_decay = group['weight_decay']

                # 動態權重衰減
                if self.dynamic_weight_decay:
                    # 根據訓練進度調整權重衰減
                    if param in group_metadata.get('lora_a_params', []) + group_metadata.get('lora_b_params', []):
                        # 對 LoRA 參數可能減少權重衰減
                        if state['step'] > 100:  # 在訓練後期可能減少權重衰減
                            current_weight_decay *= 0.5

                if current_weight_decay != 0:
                    param.data.add_(param.data, alpha=-group['lr'] * current_weight_decay)

                # SPD 正則化
                if self.use_spd:
                    spd_penalty = self._apply_spd_regularization(param, group, state)
                    if isinstance(spd_penalty, torch.Tensor):
                        param.data.add_(spd_penalty, alpha=-group['lr'])

        return loss

    def get_optimization_info(self) -> Dict[str, Any]:
        """獲取優化器的詳細信息，用於監控和調試"""
        info = {
            'optimizer_type': 'CustomAdamWOptimizer',
            'total_params': sum(len(group['params']) for group in self.param_groups),
            'features': {
                'spd': self.use_spd,
                'cautious': self.use_cautious,
                'orthogonal_grad': self.use_orthogonal_grad,
                'adopt_stability': self.use_adopt_stability,
                'grams': self.use_grams,
                'agr': self.use_agr,
                'tam': self.use_tam,
                'alora': self.use_alora,
                'dynamic_weight_decay': self.dynamic_weight_decay
            }
        }

        # 添加 LoRA 參數統計
        total_lora_a = sum(len(meta.get('lora_a_params', [])) for meta in self.param_groups_metadata.values())
        total_lora_b = sum(len(meta.get('lora_b_params', [])) for meta in self.param_groups_metadata.values())
        total_lora_pairs = sum(len(meta.get('lora_pairs', {})) for meta in self.param_groups_metadata.values())

        info['lora_stats'] = {
            'lora_a_params': total_lora_a,
            'lora_b_params': total_lora_b,
            'lora_pairs': total_lora_pairs
        }

        return info
