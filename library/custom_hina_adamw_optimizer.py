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
        spd_lambda: float = 0.06,
        use_cautious: bool = True,
        use_orthogonal_grad: bool = False,
        use_adopt_stability: bool = True,
        use_grams: bool = True,
        use_agr: bool = True,
        use_tam: bool = True,
        tam_beta: float = 0.999,
        use_alora: bool = True,
        alora_ratio: float = 16.0,
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
                'lokr_pairs': {},  # 存儲 LoKr w1-w2 配對關係
                'lokr_groups': {},  # 存儲 LoKr 組別（同一層的不同參數）
                # Norm 參數
                'w_norm_params': [],
                'b_norm_params': [],
                'norm_pairs': {},  # 存儲 w_norm-b_norm 配對關係
                # 通用參數
                'regular_params': [],
                'param_names': {},
                'param_shapes': {},
                'param_types': {}  # 記錄參數類型：'lora_a', 'lora_b', 'lokr_w1', etc.
            }

            # 建立參數名稱映射
            param_name_map = []
            if 'named' in group and group['named'] is not None:
                param_name_map = group['named']

            # 分析參數名稱和形狀
            for param_idx, param in enumerate(group['params']):
                # 從 group['named'] 中獲取參數名稱
                if param_idx < len(param_name_map) and param_name_map[param_idx] is not None:
                    param_name = param_name_map[param_idx]
                else:
                    # 如果沒有就不處理，避免出現錯誤
                    continue

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
                elif param_type == 'w_norm':
                    group_metadata['w_norm_params'].append(param)
                elif param_type == 'b_norm':
                    group_metadata['b_norm_params'].append(param)
                else:
                    group_metadata['regular_params'].append(param)

            # 建立 LoRA A-B 配對
            self._pair_lora_parameters(group_metadata)

            # 建立 LoKr 參數配對和分組
            self._pair_lokr_parameters(group_metadata)

            # 建立 Norm 參數配對
            self._pair_norm_parameters(group_metadata)

            self.param_groups_metadata[group_idx] = group_metadata

    def _classify_parameter(self, param_name):
        """分類參數類型"""
        param_name_lower = param_name.lower()

        # LoKr 參數識別（需要先於 LoRA 檢查，避免誤判）
        if '.lokr_w1' in param_name_lower or 'lokr_w1' in param_name_lower:
            return 'lokr_w1'
        elif '.lokr_w2' in param_name_lower or 'lokr_w2' in param_name_lower:
            return 'lokr_w2'
        elif 'lokr' in param_name_lower:
            # 通用 LoKr 參數（可能有其他命名變體）
            return 'lokr_generic'

        # LoRA 參數識別
        elif 'lora_down' in param_name_lower or 'lora_a' in param_name_lower:
            return 'lora_a'
        elif 'lora_up' in param_name_lower or 'lora_b' in param_name_lower:
            return 'lora_b'

        # Norm 參數識別
        elif '.w_norm' in param_name_lower or 'w_norm' in param_name_lower:
            return 'w_norm'
        elif '.b_norm' in param_name_lower or 'b_norm' in param_name_lower:
            return 'b_norm'

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
                    # 檢查維度兼容性
                    a_shape = a_param.shape
                    b_shape = b_param.shape

                    # 檢查是否可以進行矩陣乘法
                    can_multiply = False
                    if len(a_shape) >= 2 and len(b_shape) >= 2:
                        # 檢查各種可能的矩陣乘法組合
                        if (b_shape[1] == a_shape[0] or
                            b_shape[0] == a_shape[1] or
                            b_shape[1] == a_shape[1] or
                            b_shape[0] == a_shape[0]):
                            can_multiply = True

                    if can_multiply:
                        group_metadata['lora_pairs'][a_param] = b_param
                        logger.debug(f"LoRA pair matched: {base_name} - A shape: {a_shape}, B shape: {b_shape}")
                    else:
                        logger.warning(f"LoRA parameters have incompatible shapes: {base_name} - A shape: {a_shape}, B shape: {b_shape}")
                    break

    def _pair_lokr_parameters(self, group_metadata):
        """建立 LoKr 參數的配對和分組關係"""
        param_names = group_metadata['param_names']

        # 提取基礎名稱（去除 lokr 特定後綴）
        def extract_base_name(param_name):
            # 移除 LoKr 後綴，保留基礎名稱
            base_name = param_name
            if '.lokr_w1' in base_name:
                base_name = base_name.replace('.lokr_w1', '')
            elif '.lokr_w2' in base_name:
                base_name = base_name.replace('.lokr_w2', '')
            return base_name.strip('.')

        # 收集所有 LoKr 參數的基礎名稱
        lokr_base_names = {}
        all_lokr_params = group_metadata['lokr_w1_params'] + group_metadata['lokr_w2_params']

        for param in all_lokr_params:
            param_name = param_names[param]
            base_name = extract_base_name(param_name)

            if base_name not in lokr_base_names:
                lokr_base_names[base_name] = {'w1': None, 'w2': None}

            param_type = group_metadata['param_types'][param]
            if param_type == 'lokr_w1':
                lokr_base_names[base_name]['w1'] = param
            elif param_type == 'lokr_w2':
                lokr_base_names[base_name]['w2'] = param

        # 建立 LoKr w1-w2 配對關係
        for base_name, params_dict in lokr_base_names.items():
            # 儲存組別信息
            group_metadata['lokr_groups'][base_name] = params_dict

            # 建立 w1-w2 配對關係（LoKr 的核心結構）
            if params_dict['w1'] is not None and params_dict['w2'] is not None:
                group_metadata['lokr_pairs'][params_dict['w1']] = params_dict['w2']
                logger.debug(f"LoKr pair matched: {base_name} - w1 shape: {params_dict['w1'].shape}, w2 shape: {params_dict['w2'].shape}")

    def _pair_norm_parameters(self, group_metadata):
        """建立 Norm 參數的 w_norm-b_norm 配對關係"""
        param_names = group_metadata['param_names']

        # 提取基礎名稱（去除 norm 特定後綴）
        def extract_base_name(param_name):
            base_name = param_name
            if '.w_norm' in base_name:
                base_name = base_name.replace('.w_norm', '')
            elif '.b_norm' in base_name:
                base_name = base_name.replace('.b_norm', '')
            return base_name.strip('.')

        # 收集所有 Norm 參數並按基礎名稱分組
        norm_base_names = {}
        all_norm_params = group_metadata['w_norm_params'] + group_metadata['b_norm_params']

        for param in all_norm_params:
            param_name = param_names[param]
            base_name = extract_base_name(param_name)
            param_type = group_metadata['param_types'][param]

            if base_name not in norm_base_names:
                norm_base_names[base_name] = {'w_norm': None, 'b_norm': None}

            if param_type == 'w_norm':
                norm_base_names[base_name]['w_norm'] = param
            elif param_type == 'b_norm':
                norm_base_names[base_name]['b_norm'] = param

        # 建立 w_norm-b_norm 配對關係
        for base_name, params_dict in norm_base_names.items():
            if params_dict['w_norm'] is not None and params_dict['b_norm'] is not None:
                group_metadata['norm_pairs'][params_dict['w_norm']] = params_dict['b_norm']
                logger.debug(f"Norm pair matched: {base_name}")

    def _store_initial_parameters(self):
        """存儲初始參數以供 SPD 使用"""
        self.initial_params = {}
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param.requires_grad:
                    self.initial_params[param] = param.data.clone().detach()

    def update_device(self, device):
        """
        當模型被移動到新裝置時，更新優化器內部存儲的張量裝置

        Args:
            device: 新的裝置（如 'cuda:0', 'cpu' 等）
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
        # 確保 initial_param 與 param.data 在同一個裝置上
        if initial_param.device != param.data.device:
            initial_param = initial_param.to(param.data.device)
            # 更新存儲的初始參數到正確的裝置
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

        try:
            param_flat = param.view(-1)
            grad_flat = grad.view(-1)

            # 檢查張量大小是否匹配
            if param_flat.size() != grad_flat.size():
                logger.warning(f"Orthogonal gradient: Parameter and gradient size mismatch: {param_flat.size()} vs {grad_flat.size()}. Skipping orthogonal projection.")
                return grad

            # 計算正交投影
            dot_product = torch.dot(param_flat, grad_flat)
            param_norm_sq = torch.dot(param_flat, param_flat)

            if param_norm_sq > 0:
                projection = (dot_product / param_norm_sq) * param_flat
                orthogonal_grad = grad_flat - projection
                return orthogonal_grad.view(grad.shape)

            return grad
        except Exception as e:
            logger.warning(f"Orthogonal gradient: Error computing orthogonal projection: {e}. Returning original gradient.")
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
        try:
            # 檢查張量形狀是否匹配
            if update.shape != grad.shape:
                logger.warning(f"Cautious update: Update and gradient shape mismatch: {update.shape} vs {grad.shape}. Returning original update.")
                return update

            # 計算對齊遮罩：僅當更新方向與梯度對齊時才應用
            alignment = update * grad
            alignment_mask = (alignment > 0).float()
            return update * alignment_mask
        except Exception as e:
            logger.warning(f"Cautious update: Error computing alignment mask: {e}. Returning original update.")
            return update

    def _compute_alora_lr_scale(self, lora_a_param, lora_b_param):
        """計算 ALoRA 風格的學習率縮放因子"""
        try:
            # 確保參數是 2D 張量，如果不是就進行重構
            if lora_a_param.dim() == 2 and lora_b_param.dim() == 2:
                # LoRA 的標準結構：A 是下投影 (rank, input_dim)，B 是上投影 (output_dim, rank)
                # 要計算 B @ A，需要確保維度匹配
                a_shape = lora_a_param.shape  # (rank, input_dim) 或 (input_dim, rank)
                b_shape = lora_b_param.shape  # (output_dim, rank) 或 (rank, output_dim)

                # 確定正確的矩陣乘法順序
                # 通常 LoRA 的結構是 B @ A，其中 B.shape[1] == A.shape[0]
                if b_shape[1] == a_shape[0]:
                    # B @ A：(output_dim, rank) @ (rank, input_dim) = (output_dim, input_dim)
                    ba_matrix = torch.matmul(lora_b_param, lora_a_param)
                elif b_shape[0] == a_shape[1]:
                    # A^T @ B^T：(input_dim, rank) @ (rank, output_dim) = (input_dim, output_dim)
                    ba_matrix = torch.matmul(lora_a_param, lora_b_param)
                elif b_shape[1] == a_shape[1]:
                    # B @ A^T：(output_dim, rank) @ (input_dim, rank)^T = (output_dim, input_dim)
                    ba_matrix = torch.matmul(lora_b_param, lora_a_param.T)
                elif b_shape[0] == a_shape[0]:
                    # B^T @ A：(output_dim, rank)^T @ (rank, input_dim) = (rank, input_dim)
                    ba_matrix = torch.matmul(lora_b_param.T, lora_a_param)
                else:
                    # 維度完全不匹配，使用參數範數作為替代計算
                    logger.warning(f"LoRA parameter dimension mismatch: A {a_shape}, B {b_shape}. Using norm-based scaling.")
                    a_norm = torch.norm(lora_a_param)
                    b_norm = torch.norm(lora_b_param)
                    combined_norm = (a_norm + b_norm) / 2
                    if combined_norm > 0:
                        lr_scale = 1.0 / (1.0 + combined_norm)
                    else:
                        lr_scale = 1.0
                    return lr_scale.item()
            else:
                # 處理非 2D 張量（如卷積層的情況）
                # 重構為 2D 張量進行計算
                a_flat = lora_a_param.view(lora_a_param.size(0), -1)
                b_flat = lora_b_param.view(lora_b_param.size(0), -1)

                # 嘗試矩陣乘法的不同組合
                if b_flat.shape[1] == a_flat.shape[0]:
                    ba_matrix = torch.matmul(b_flat, a_flat)
                elif b_flat.shape[0] == a_flat.shape[1]:
                    ba_matrix = torch.matmul(a_flat, b_flat)
                elif b_flat.shape[1] == a_flat.shape[1]:
                    ba_matrix = torch.matmul(b_flat, a_flat.T)
                elif b_flat.shape[0] == a_flat.shape[0]:
                    ba_matrix = torch.matmul(b_flat.T, a_flat)
                else:
                    # 使用參數範數作為替代
                    logger.warning(f"Cannot perform matrix multiplication between shapes {a_flat.shape} and {b_flat.shape}. Using norm-based scaling.")
                    a_norm = torch.norm(a_flat)
                    b_norm = torch.norm(b_flat)
                    combined_norm = (a_norm + b_norm) / 2
                    if combined_norm > 0:
                        lr_scale = 1.0 / (1.0 + combined_norm)
                    else:
                        lr_scale = 1.0
                    return lr_scale.item()

            # 計算行向量的 L2 範數
            if ba_matrix.dim() == 2 and ba_matrix.size(0) > 0:
                row_norms = torch.norm(ba_matrix, dim=1)
                avg_row_norm = torch.mean(row_norms)
            else:
                avg_row_norm = torch.norm(ba_matrix)

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
        LoKr 使用 Kronecker 積分解，w1 和 w2 的 Kronecker 積重建原始權重
        """
        try:
            # 獲取 LoKr w1 和 w2 參數
            w1 = lokr_group.get('w1')
            w2 = lokr_group.get('w2')

            if w1 is not None and w2 is not None:
                # 計算 Kronecker 積的近似範數
                w1_norm = torch.norm(w1.data).item()
                w2_norm = torch.norm(w2.data).item()

                # LoKr 的總體影響近似為兩個矩陣範數的乘積
                combined_norm = w1_norm * w2_norm

                # 學習率縮放策略：與組合範數成反比
                lr_scale = 1.0 / (1.0 + combined_norm * 0.3)  # 比 LoRA 稍微保守
                return lr_scale
            else:
                return 1.0

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
        try:
            if torch.norm(momentum) > 0 and torch.norm(grad) > 0:
                # 確保兩個張量的形狀匹配
                momentum_flat = momentum.view(-1)
                grad_flat = grad.view(-1)

                # 檢查張量大小是否匹配
                if momentum_flat.size() != grad_flat.size():
                    logger.warning(f"TAM: Momentum and gradient size mismatch: {momentum_flat.size()} vs {grad_flat.size()}. Skipping alignment computation.")
                    alignment = 0.0
                else:
                    alignment = torch.dot(momentum_flat, grad_flat) / (
                        torch.norm(momentum) * torch.norm(grad)
                    )
            else:
                alignment = 0.0
        except Exception as e:
            logger.warning(f"TAM: Error computing alignment: {e}. Using zero alignment.")
            alignment = 0.0

        # 平滑對齊估計
        state['momentum_alignment'] = (
            self.tam_beta * state['momentum_alignment'] +
            (1 - self.tam_beta) * alignment
        )

        # 計算阻尼因子
        damping_factor = (1 + state['momentum_alignment']) / 2
        return damping_factor

    @torch.no_grad()
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
                else:
                    # 檢查現有狀態張量是否與參數形狀匹配，如果不匹配則重新初始化
                    if state['exp_avg'].shape != param.data.shape:
                        logger.warning(f"State tensor shape mismatch detected. Reinitializing state for parameter with shape {param.data.shape}")
                        state['exp_avg'] = torch.zeros_like(param.data)
                        state['exp_avg_sq'] = torch.zeros_like(param.data)
                        if self.use_adopt_stability:
                            state['exp_avg_sq_prev'] = torch.zeros_like(param.data)

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
                try:
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
                except Exception as e:
                    logger.error(f"Error in Adam update computation: {e}. Falling back to standard Adam.")
                    # 重新初始化狀態並使用標準更新
                    state['exp_avg'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
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

                # LoKr/Lora 學習率調整
                if self.use_alora:
                    if param_type.startswith('lokr_'):
                        # 為 LoKr 參數尋找對應的組別
                        for base_name, lokr_group in group_metadata.get('lokr_groups', {}).items():
                            # 使用 id() 比較避免張量比較導致的尺寸錯誤
                            param_found = any(id(param) == id(p) for p in lokr_group.values() if p is not None)
                            if param_found:
                                try:
                                    lr_scale = self._compute_lokr_lr_scale(lokr_group)
                                    current_step_size *= lr_scale
                                    logger.debug(f"Applied LoKr lr_scale {lr_scale} to parameter in group {base_name}")
                                except Exception as e:
                                    logger.warning(f"Failed to compute LoKr lr scale for group {base_name}: {e}")
                                break

                        # LoKr 的特殊學習率比例調整
                        # 對於 LoKr w2 參數，通常應用較高的學習率（類似 LoRA B）
                        if param_type == 'lokr_w2':
                            current_step_size *= (self.alora_ratio * 0.6)  # 比 LoRA 更保守的調整
                            logger.debug(f"Applied LoKr w2 ratio adjustment to parameter")
                    else:
                        # 檢查是否為配對的 LoRA 參數 - 使用 id() 比較避免張量比較錯誤
                        lora_pairs = group_metadata.get('lora_pairs', {})
                        param_in_lora_pairs = any(id(param) == id(p) for p in lora_pairs.keys())
                        if param_in_lora_pairs:
                            # 找到對應的配對參數
                            paired_param = None
                            for p, paired in lora_pairs.items():
                                if id(param) == id(p):
                                    paired_param = paired
                                    break

                            if paired_param is not None:
                                try:
                                    lr_scale = self._compute_alora_lr_scale(param, paired_param)
                                    current_step_size *= lr_scale
                                    logger.debug(f"Applied ALoRA lr_scale {lr_scale} to paired LoRA parameter")
                                except Exception as e:
                                    logger.warning(f"Failed to compute ALoRA lr scale for paired parameter: {e}")

                        # 對未配對的 LoRA B 參數應用比例調整
                        else:
                            lora_b_params = group_metadata.get('lora_b_params', [])
                            param_in_lora_b = any(id(param) == id(p) for p in lora_b_params)
                            if param_in_lora_b:
                                # 檢查是否已經在配對中（避免重複處理）
                                param_in_paired_values = any(id(param) == id(p) for p in lora_pairs.values())
                                if not param_in_paired_values:
                                    current_step_size *= self.alora_ratio
                                    logger.debug(f"Applied ALoRA ratio {self.alora_ratio} to unpaired LoRA B parameter")

                # 應用更新
                param.data.add_(update, alpha=-current_step_size)

                # 權重衰減
                current_weight_decay = group['weight_decay']

                # 動態權重衰減
                if self.dynamic_weight_decay:
                    # LoRA 參數的動態權重衰減 - 使用 id() 比較避免張量比較錯誤
                    lora_a_params = group_metadata.get('lora_a_params', [])
                    lora_b_params = group_metadata.get('lora_b_params', [])
                    param_in_lora_a = any(id(param) == id(p) for p in lora_a_params)
                    param_in_lora_b = any(id(param) == id(p) for p in lora_b_params)

                    if param_in_lora_a or param_in_lora_b:
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
        total_lokr_pairs = sum(len(meta.get('lokr_pairs', {})) for meta in self.param_groups_metadata.values())
        total_lokr_groups = sum(len(meta.get('lokr_groups', {})) for meta in self.param_groups_metadata.values())

        # 添加 Norm 參數統計
        total_w_norm = sum(len(meta.get('w_norm_params', [])) for meta in self.param_groups_metadata.values())
        total_b_norm = sum(len(meta.get('b_norm_params', [])) for meta in self.param_groups_metadata.values())
        total_norm_pairs = sum(len(meta.get('norm_pairs', {})) for meta in self.param_groups_metadata.values())

        info['lora_stats'] = {
            'lora_a_params': total_lora_a,
            'lora_b_params': total_lora_b,
            'lora_pairs': total_lora_pairs
        }

        info['lokr_stats'] = {
            'lokr_w1_params': total_lokr_w1,
            'lokr_w2_params': total_lokr_w2,
            'lokr_pairs': total_lokr_pairs,
            'lokr_groups': total_lokr_groups,
            'w_norm_params': total_w_norm,
            'b_norm_params': total_b_norm,
            'norm_pairs': total_norm_pairs
        }

        return info

    def diagnose_lora_pairing(self) -> Dict[str, Any]:
        """診斷 LoRA 參數配對狀況，幫助調試維度不匹配問題"""
        diagnosis = {
            'total_groups': len(self.param_groups_metadata),
            'groups': {}
        }

        for group_idx, group_metadata in self.param_groups_metadata.items():
            group_diagnosis = {
                'lora_a_count': len(group_metadata.get('lora_a_params', [])),
                'lora_b_count': len(group_metadata.get('lora_b_params', [])),
                'pairs_count': len(group_metadata.get('lora_pairs', {})),
                'unpaired_a': [],
                'unpaired_b': [],
                'parameter_details': []
            }

            # 檢查未配對的參數
            lora_a_params = set(group_metadata.get('lora_a_params', []))
            lora_b_params = set(group_metadata.get('lora_b_params', []))
            paired_a = set(group_metadata.get('lora_pairs', {}).keys())
            paired_b = set(group_metadata.get('lora_pairs', {}).values())

            unpaired_a = lora_a_params - paired_a
            unpaired_b = lora_b_params - paired_b

            param_names = group_metadata.get('param_names', {})

            # 記錄未配對的 A 參數
            for param in unpaired_a:
                param_name = param_names.get(param, 'unknown')
                group_diagnosis['unpaired_a'].append({
                    'name': param_name,
                    'shape': list(param.shape)
                })

            # 記錄未配對的 B 參數
            for param in unpaired_b:
                param_name = param_names.get(param, 'unknown')
                group_diagnosis['unpaired_b'].append({
                    'name': param_name,
                    'shape': list(param.shape)
                })

            # 記錄所有 LoRA 參數的詳細信息
            for param in lora_a_params.union(lora_b_params):
                param_name = param_names.get(param, 'unknown')
                param_type = group_metadata.get('param_types', {}).get(param, 'unknown')
                is_paired = param in paired_a or param in paired_b
                paired_with = None

                if param in paired_a:
                    paired_with = param_names.get(group_metadata['lora_pairs'][param], 'unknown')
                elif param in paired_b:
                    # 找到配對的 A 參數
                    for a_param, b_param in group_metadata.get('lora_pairs', {}).items():
                        if b_param == param:
                            paired_with = param_names.get(a_param, 'unknown')
                            break

                group_diagnosis['parameter_details'].append({
                    'name': param_name,
                    'type': param_type,
                    'shape': list(param.shape),
                    'is_paired': is_paired,
                    'paired_with': paired_with
                })

            diagnosis['groups'][f'group_{group_idx}'] = group_diagnosis

        return diagnosis
