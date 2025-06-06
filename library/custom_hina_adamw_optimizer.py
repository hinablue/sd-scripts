"""
Hina Custom AdamW Optimizer with Enhanced Features
基於 AdamW8bit 的增強優化器，整合多種先進優化技術

This optimizer integrates multiple state-of-the-art optimization techniques:
- Generalization enhancement (SPD, Cautious Optimizer, Orthogonal Gradient)
- Adaptive learning rates (ADOPT, Grams, AGR, TAM)
- LoRA-specific optimizations (ALoRA, dynamic weight decay)
- LoKr-specific optimizations (Kronecker product-aware learning rates, specialized weight decay)
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

    4. LoKr 專屬優化：
       - Kronecker 積結構感知的學習率縮放
       - LoKr 專屬的動態權重衰減策略
       - 支援多種 LoKr 參數命名模式
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
        # Dynamic weight decay configuration
        wd_transition_steps: int = 1000,  # 權重衰減過渡的步數閾值
        wd_decay_factor: float = 0.7,    # 權重衰減減少係數
        wd_min_ratio: float = 0.1,       # 最小權重衰減比例
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
            wd_transition_steps: 權重衰減過渡的步數閾值
            wd_decay_factor: 權重衰減減少係數
            wd_min_ratio: 最小權重衰減比例
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
        # Dynamic weight decay configuration
        self.wd_transition_steps = wd_transition_steps
        self.wd_decay_factor = wd_decay_factor
        self.wd_min_ratio = wd_min_ratio

        # Initialize parameter groups and metadata
        self._initialize_parameter_groups()

        # Store initial parameters for SPD
        if self.use_spd:
            self._store_initial_parameters()

        logger.info(f"HinaAdamWOptimizer initialized with enhanced features:")
        logger.info(f"  SPD: {use_spd}, Cautious: {use_cautious}, Orthogonal Grad: {use_orthogonal_grad}")
        logger.info(f"  ADOPT: {use_adopt_stability}, Grams: {use_grams}, AGR: {use_agr}")
        logger.info(f"  TAM: {use_tam}, ALoRA: {use_alora}, Dynamic WD: {dynamic_weight_decay}")
        logger.info(f"  LoRA/LoKr Support: Automatic parameter type detection and specialized optimization")

    def _initialize_parameter_groups(self):
        """初始化參數分組，識別 LoRA/LoKr 參數和其他參數類型"""
        self.param_groups_metadata = {}

        for group_idx, group in enumerate(self.param_groups):
            group_metadata = {
                # LoRA 參數
                'lora_a_params': [],
                'lora_b_params': [],
                'lora_pairs': {},  # 存儲 A-B 配對
                # LoKr 參數
                'lokr_w1_params': [],
                'lokr_w2_params': [],
                'lokr_w1_a_params': [],
                'lokr_w1_b_params': [],
                'lokr_w2_a_params': [],
                'lokr_w2_b_params': [],
                'lokr_pairs': {},  # 存儲 LoKr 配對關係
                'lokr_groups': {},  # 存儲 LoKr 組別（同一層的不同參數）
                # 通用參數
                'regular_params': [],
                'param_names': {},
                'param_shapes': {},
                'param_types': {}  # 記錄參數類型：'lora_a', 'lora_b', 'lokr_w1', etc.
            }

            # 分析參數名稱和形狀
            for param_idx, param in enumerate(group['params']):
                if hasattr(param, 'param_name'):
                    param_name = param.param_name
                else:
                    param_name = f"param_{group_idx}_{param_idx}"

                group_metadata['param_names'][param] = param_name
                group_metadata['param_shapes'][param] = param.shape

                # 識別參數類型並分類
                param_type = self._classify_parameter(param_name)
                group_metadata['param_types'][param] = param_type

                if param_type == 'lora_a':
                    group_metadata['lora_a_params'].append(param)
                elif param_type == 'lora_b':
                    group_metadata['lora_b_params'].append(param)
                elif param_type == 'lokr_w1':
                    group_metadata['lokr_w1_params'].append(param)
                elif param_type == 'lokr_w2':
                    group_metadata['lokr_w2_params'].append(param)
                elif param_type == 'lokr_w1_a':
                    group_metadata['lokr_w1_a_params'].append(param)
                elif param_type == 'lokr_w1_b':
                    group_metadata['lokr_w1_b_params'].append(param)
                elif param_type == 'lokr_w2_a':
                    group_metadata['lokr_w2_a_params'].append(param)
                elif param_type == 'lokr_w2_b':
                    group_metadata['lokr_w2_b_params'].append(param)
                else:
                    group_metadata['regular_params'].append(param)

            # 建立 LoRA A-B 配對
            self._pair_lora_parameters(group_metadata)

            # 建立 LoKr 參數配對和分組
            self._pair_lokr_parameters(group_metadata)

            self.param_groups_metadata[group_idx] = group_metadata

    def _classify_parameter(self, param_name):
        """分類參數類型"""
        param_name_lower = param_name.lower()

        # LoKr 參數識別（需要先於 LoRA 檢查，避免誤判）
        if 'lokr_w1_a' in param_name_lower or 'lokr.w1_a' in param_name_lower:
            return 'lokr_w1_a'
        elif 'lokr_w1_b' in param_name_lower or 'lokr.w1_b' in param_name_lower:
            return 'lokr_w1_b'
        elif 'lokr_w2_a' in param_name_lower or 'lokr.w2_a' in param_name_lower:
            return 'lokr_w2_a'
        elif 'lokr_w2_b' in param_name_lower or 'lokr.w2_b' in param_name_lower:
            return 'lokr_w2_b'
        elif 'lokr_w1' in param_name_lower or 'lokr.w1' in param_name_lower:
            return 'lokr_w1'
        elif 'lokr_w2' in param_name_lower or 'lokr.w2' in param_name_lower:
            return 'lokr_w2'
        elif 'lokr' in param_name_lower:
            # 通用 LoKr 參數（可能有其他命名變體）
            return 'lokr_generic'

        # LoRA 參數識別
        elif 'lora_down' in param_name_lower or 'lora_a' in param_name_lower:
            return 'lora_a'
        elif 'lora_up' in param_name_lower or 'lora_b' in param_name_lower:
            return 'lora_b'

        # 其他參數
        else:
            return 'regular'

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

    def _pair_lokr_parameters(self, group_metadata):
        """建立 LoKr 參數的配對和分組關係"""
        param_names = group_metadata['param_names']

        # 提取基礎名稱（去除 lokr 特定後綴）
        def extract_base_name(param_name):
            # 移除常見的 LoKr 後綴
            base_name = param_name
            suffixes_to_remove = [
                '.lokr_w1_a.weight', '.lokr_w1_b.weight',
                '.lokr_w2_a.weight', '.lokr_w2_b.weight',
                '.lokr_w1.weight', '.lokr_w2.weight',
                '.lokr.w1_a.weight', '.lokr.w1_b.weight',
                '.lokr.w2_a.weight', '.lokr.w2_b.weight',
                '.lokr.w1.weight', '.lokr.w2.weight',
                'lokr_w1_a', 'lokr_w1_b', 'lokr_w2_a', 'lokr_w2_b',
                'lokr_w1', 'lokr_w2', '.weight'
            ]

            for suffix in suffixes_to_remove:
                if suffix in base_name:
                    base_name = base_name.replace(suffix, '')
                    break

            return base_name.strip('.')

        # 收集所有 LoKr 參數的基礎名稱
        lokr_base_names = {}
        all_lokr_params = (
            group_metadata['lokr_w1_params'] + group_metadata['lokr_w2_params'] +
            group_metadata['lokr_w1_a_params'] + group_metadata['lokr_w1_b_params'] +
            group_metadata['lokr_w2_a_params'] + group_metadata['lokr_w2_b_params']
        )

        for param in all_lokr_params:
            param_name = param_names[param]
            base_name = extract_base_name(param_name)

            if base_name not in lokr_base_names:
                lokr_base_names[base_name] = {
                    'w1': None, 'w2': None,
                    'w1_a': None, 'w1_b': None,
                    'w2_a': None, 'w2_b': None
                }

            param_type = group_metadata['param_types'][param]
            if param_type == 'lokr_w1':
                lokr_base_names[base_name]['w1'] = param
            elif param_type == 'lokr_w2':
                lokr_base_names[base_name]['w2'] = param
            elif param_type == 'lokr_w1_a':
                lokr_base_names[base_name]['w1_a'] = param
            elif param_type == 'lokr_w1_b':
                lokr_base_names[base_name]['w1_b'] = param
            elif param_type == 'lokr_w2_a':
                lokr_base_names[base_name]['w2_a'] = param
            elif param_type == 'lokr_w2_b':
                lokr_base_names[base_name]['w2_b'] = param

        # 建立 LoKr 配對關係
        for base_name, params_dict in lokr_base_names.items():
            # 儲存組別信息
            group_metadata['lokr_groups'][base_name] = params_dict

            # 建立配對關係
            if params_dict['w1_a'] and params_dict['w1_b']:
                group_metadata['lokr_pairs'][params_dict['w1_a']] = params_dict['w1_b']
            if params_dict['w2_a'] and params_dict['w2_b']:
                group_metadata['lokr_pairs'][params_dict['w2_a']] = params_dict['w2_b']
            if params_dict['w1'] and params_dict['w2']:
                group_metadata['lokr_pairs'][params_dict['w1']] = params_dict['w2']

    def _store_initial_parameters(self):
        """存儲初始參數以供 SPD 使用"""
        self.initial_params = {}
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param.requires_grad:
                    self.initial_params[param] = param.data.clone().detach()

    def update_device(self, device):
        """
        當模型被移動到新設備時，更新優化器內部存儲的張量設備

        Args:
            device: 新的設備（如 'cuda:0', 'cpu' 等）
        """
        if hasattr(self, 'initial_params'):
            for param, initial_param in self.initial_params.items():
                if initial_param.device != device:
                    self.initial_params[param] = initial_param.to(device)

        # 更新所有狀態中的張量
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    state[key] = value.to(device)

    def _apply_spd_regularization(self, param, group, state):
        """應用 Selective Projection Decay 正則化"""
        if param not in self.initial_params:
            return 0

        initial_param = self.initial_params[param]
        # 確保 initial_param 與 param.data 在同一個設備上
        if initial_param.device != param.data.device:
            initial_param = initial_param.to(param.data.device)
            # 更新存儲的初始參數到正確的設備
            self.initial_params[param] = initial_param

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

    def _compute_lokr_lr_scale(self, lokr_group):
        """
        計算 LoKr 風格的學習率縮放因子
        LoKr 使用 Kronecker 積分解，需要特殊的處理方式
        """
        try:
            # 獲取 LoKr 參數
            w1_a = lokr_group.get('w1_a')
            w1_b = lokr_group.get('w1_b')
            w2_a = lokr_group.get('w2_a')
            w2_b = lokr_group.get('w2_b')
            w1 = lokr_group.get('w1')
            w2 = lokr_group.get('w2')

            total_norm = 0.0
            param_count = 0

            # 處理 w1_a, w1_b 配對
            if w1_a is not None and w1_b is not None:
                w1_product = torch.matmul(w1_b.data, w1_a.data)
                total_norm += torch.norm(w1_product).item()
                param_count += 1

            # 處理 w2_a, w2_b 配對
            if w2_a is not None and w2_b is not None:
                w2_product = torch.matmul(w2_b.data, w2_a.data)
                total_norm += torch.norm(w2_product).item()
                param_count += 1

            # 處理直接的 w1, w2 參數
            if w1 is not None:
                total_norm += torch.norm(w1.data).item()
                param_count += 1
            if w2 is not None:
                total_norm += torch.norm(w2.data).item()
                param_count += 1

            if param_count > 0:
                avg_norm = total_norm / param_count
                # LoKr 的學習率縮放策略，考慮 Kronecker 積結構
                lr_scale = 1.0 / (1.0 + avg_norm * 0.5)  # 比 LoRA 更溫和的縮放
            else:
                lr_scale = 1.0

            return lr_scale
        except Exception as e:
            logger.warning(f"Failed to compute LoKr lr scale: {e}")
            return 1.0

    def _get_lokr_dynamic_weight_decay(self, param, group_metadata, state):
        """
        計算 LoKr 參數的動態權重衰減
        LoKr 由於使用 Kronecker 積結構，需要更精細的權重衰減策略
        """
        if not self.dynamic_weight_decay:
            return 1.0

        param_type = group_metadata['param_types'].get(param, 'regular')

        # LoKr 參數的動態權重衰減策略
        if param_type.startswith('lokr_'):
            # 對於 LoKr 參數，使用更保守的權重衰減減少策略
            if state['step'] > self.wd_transition_steps:
                progress = (state['step'] - self.wd_transition_steps) / self.wd_transition_steps
                # LoKr 使用更溫和的衰減曲線
                decay_multiplier = max(
                    self.wd_min_ratio * 1.5,  # LoKr 保持更高的最小權重衰減
                    (self.wd_decay_factor ** 0.7) ** min(progress, 1.5)  # 更溫和的衰減
                )
                return decay_multiplier

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
                    raise RuntimeError('HinaAdamWOptimizer does not support sparse gradients')

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
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

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
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
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

                # 學習率調整（ALoRA + LoKr）
                current_step_size = step_size
                param_type = group_metadata['param_types'].get(param, 'regular')

                # LoRA 學習率調整
                if self.use_alora and param in group_metadata.get('lora_pairs', {}):
                    paired_param = group_metadata['lora_pairs'][param]
                    lr_scale = self._compute_alora_lr_scale(param, paired_param)
                    current_step_size *= lr_scale
                elif self.use_alora and param in group_metadata.get('lora_b_params', []):
                    # 對 B 參數應用比例調整
                    current_step_size *= self.alora_ratio

                # LoKr 學習率調整
                elif self.use_alora and param_type.startswith('lokr_'):
                    # 為 LoKr 參數尋找對應的組別
                    for base_name, lokr_group in group_metadata.get('lokr_groups', {}).items():
                        if param in lokr_group.values():
                            lr_scale = self._compute_lokr_lr_scale(lokr_group)
                            current_step_size *= lr_scale
                            break

                    # LoKr 的特殊學習率比例調整
                    if param_type in ['lokr_w1_b', 'lokr_w2_b', 'lokr_w2']:
                        # 對於 LoKr 的"上層"參數，應用較高的學習率
                        current_step_size *= (self.alora_ratio * 0.8)  # 比 LoRA 稍微保守一些

                # 應用更新
                param.data.add_(update, alpha=-current_step_size)

                # 權重衰減
                current_weight_decay = group['weight_decay']

                # 動態權重衰減
                if self.dynamic_weight_decay:
                    # LoRA 參數的動態權重衰減
                    if param in group_metadata.get('lora_a_params', []) + group_metadata.get('lora_b_params', []):
                        # 對 LoRA 參數進行漸進式權重衰減調整
                        if state['step'] > self.wd_transition_steps:
                            # 計算漸進式衰減係數
                            progress = (state['step'] - self.wd_transition_steps) / self.wd_transition_steps
                            # 使用指數衰減曲線，避免權重衰減降得過低
                            decay_multiplier = max(
                                self.wd_min_ratio,
                                self.wd_decay_factor ** min(progress, 2.0)  # 限制進度最大為 2.0
                            )
                            current_weight_decay *= decay_multiplier

                    # LoKr 參數的動態權重衰減
                    elif param_type.startswith('lokr_'):
                        lokr_decay_multiplier = self._get_lokr_dynamic_weight_decay(param, group_metadata, state)
                        current_weight_decay *= lokr_decay_multiplier

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
            'optimizer_type': 'HinaAdamWOptimizer',
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
                'dynamic_weight_decay': self.dynamic_weight_decay,
                'wd_transition_steps': self.wd_transition_steps,
                'wd_decay_factor': self.wd_decay_factor,
                'wd_min_ratio': self.wd_min_ratio
            }
        }

        # 添加 LoRA 參數統計
        total_lora_a = sum(len(meta.get('lora_a_params', [])) for meta in self.param_groups_metadata.values())
        total_lora_b = sum(len(meta.get('lora_b_params', [])) for meta in self.param_groups_metadata.values())
        total_lora_pairs = sum(len(meta.get('lora_pairs', {})) for meta in self.param_groups_metadata.values())

        # 添加 LoKr 參數統計
        total_lokr_w1 = sum(len(meta.get('lokr_w1_params', [])) for meta in self.param_groups_metadata.values())
        total_lokr_w2 = sum(len(meta.get('lokr_w2_params', [])) for meta in self.param_groups_metadata.values())
        total_lokr_w1_a = sum(len(meta.get('lokr_w1_a_params', [])) for meta in self.param_groups_metadata.values())
        total_lokr_w1_b = sum(len(meta.get('lokr_w1_b_params', [])) for meta in self.param_groups_metadata.values())
        total_lokr_w2_a = sum(len(meta.get('lokr_w2_a_params', [])) for meta in self.param_groups_metadata.values())
        total_lokr_w2_b = sum(len(meta.get('lokr_w2_b_params', [])) for meta in self.param_groups_metadata.values())
        total_lokr_pairs = sum(len(meta.get('lokr_pairs', {})) for meta in self.param_groups_metadata.values())
        total_lokr_groups = sum(len(meta.get('lokr_groups', {})) for meta in self.param_groups_metadata.values())

        info['lora_stats'] = {
            'lora_a_params': total_lora_a,
            'lora_b_params': total_lora_b,
            'lora_pairs': total_lora_pairs
        }

        info['lokr_stats'] = {
            'lokr_w1_params': total_lokr_w1,
            'lokr_w2_params': total_lokr_w2,
            'lokr_w1_a_params': total_lokr_w1_a,
            'lokr_w1_b_params': total_lokr_w1_b,
            'lokr_w2_a_params': total_lokr_w2_a,
            'lokr_w2_b_params': total_lokr_w2_b,
            'lokr_pairs': total_lokr_pairs,
            'lokr_groups': total_lokr_groups
        }

        return info
