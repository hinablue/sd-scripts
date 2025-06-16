import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Tuple, Optional
import math
from collections import defaultdict

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# 引入隨機舍入功能
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    使用隨機舍入將源張量複製到目標張量

    Args:
        target: dtype=bfloat16 的目標張量
        source: dtype=float32 的源張量
    """
    # 確保輸入類型正確
    if target.dtype != torch.bfloat16:
        raise ValueError(f"目標張量必須是 bfloat16，但得到 {target.dtype}")
    if source.dtype != torch.float32:
        raise ValueError(f"源張量必須是 float32，但得到 {source.dtype}")

    # 創建一個隨機 16 位整數
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # 將隨機數添加到尾數的低 16 位
    result.add_(source.view(dtype=torch.int32))

    # 屏蔽尾數的低 16 位
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 作為有符號 int32

    # 將高 16 位複製到目標張量（正確轉換為 bfloat16）
    target.copy_(result.view(dtype=torch.float32).to(dtype=torch.bfloat16))

    del result


def add_stochastic_(_input: torch.Tensor, other: torch.Tensor, alpha: float = 1.0):
    """
    使用隨機舍入將 other 添加到 input

    Args:
        _input: dtype=bfloat16 的輸入張量
        other: 其他張量
        alpha: other 的乘數
    """
    # 確保輸入張量是 bfloat16
    if _input.dtype != torch.bfloat16:
        raise ValueError(f"輸入張量必須是 bfloat16，但得到 {_input.dtype}")

    if other.dtype == torch.float32:
        result = other.clone()
    else:
        result = other.to(dtype=torch.float32)

    # 將 bfloat16 輸入轉換為 float32 進行計算
    _input_fp32 = _input.to(dtype=torch.float32)
    result.add_(_input_fp32, alpha=alpha)

    copy_stochastic_(_input, result)


def addcdiv_stochastic_(
    _input: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor, value: float = 1.0
):
    """
    使用隨機舍入將 (tensor1 / tensor2 * value) 添加到 input

    Args:
        _input: dtype=bfloat16 的輸入張量
        tensor1: 分子張量
        tensor2: 分母張量
        value: tensor1/tensor2 的乘數
    """
    # 確保輸入張量是 bfloat16
    if _input.dtype != torch.bfloat16:
        raise ValueError(f"輸入張量必須是 bfloat16，但得到 {_input.dtype}")

    # 將輸入轉換為 float32 進行計算
    result = _input.to(dtype=torch.float32)

    # 確保 tensor1 和 tensor2 也是正確的精度
    if tensor1.dtype != torch.float32:
        tensor1 = tensor1.to(dtype=torch.float32)
    if tensor2.dtype != torch.float32:
        tensor2 = tensor2.to(dtype=torch.float32)

    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(_input, result)


class AdaptiveHinaAdamWBF16(Optimizer):
    """
    自適應 HinaAdamW BF16 專用優化器

    結合了 AdamW_HinaAdaptive 的智能自適應功能和 AdamWBF16 的 bfloat16 優化，
    專為現代混合精度訓練設計。

    主要特色：
    1. 動態參數重要性評估與學習率自適應調整
    2. 自動參數關係發現與配對優化
    3. bfloat16 精度的隨機舍入與補償式累加
    4. 延遲累積權重衰減機制
    5. 多種進階優化技術（SPD、AGR、TAM、正交梯度等）
    6. 記憶體優化的緩衝區池管理
    """

    decay_threshold = 5e-3  # 權重衰減累積閾值

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        # 增強功能配置
        use_spd: bool = True,
        spd_lambda: float = 0.06,
        use_cautious: bool = True,
        use_orthogonal_grad: bool = False,
        use_adopt_stability: bool = True,
        use_grams: bool = True,
        use_agr: bool = True,
        use_tam: bool = True,
        tam_beta: float = 0.999,
        # 動態自適應學習率功能
        use_dynamic_adaptation: bool = True,
        adaptation_strength: float = 1.0,
        relationship_discovery_interval: int = 100,
        importance_decay: float = 0.95,
        compatibility_threshold: float = 0.3,
        # 動態權重衰減配置
        dynamic_weight_decay: bool = True,
        wd_transition_steps: int = 1000,
        wd_decay_factor: float = 0.7,
        wd_min_ratio: float = 0.1,
        **kwargs
    ):
        """
        初始化 AdamWBF16_HinaAdaptive 優化器

        Args:
            params: 模型參數
            lr: 學習率
            betas: Adam 的 beta 參數
            eps: 數值穩定性常數
            weight_decay: 權重衰減係數
            use_spd: 啟用選擇性投影衰減
            spd_lambda: SPD 懲罰強度
            use_cautious: 啟用謹慎優化器機制
            use_orthogonal_grad: 啟用正交梯度投影
            use_adopt_stability: 啟用 ADOPT 穩定性機制
            use_grams: 啟用 Grams 自適應動量縮放
            use_agr: 啟用自適應梯度正則化
            use_tam: 啟用力矩感知動量
            tam_beta: TAM 的 beta 參數
            use_dynamic_adaptation: 是否啟用動態自適應學習率
            adaptation_strength: 自適應調整的強度係數
            relationship_discovery_interval: 參數關係重新發現的間隔步數
            importance_decay: 重要性分數的時間衰減係數
            compatibility_threshold: 判斷參數相容性的閾值
            dynamic_weight_decay: 啟用動態權重衰減
            wd_transition_steps: 權重衰減過渡的步數閾值
            wd_decay_factor: 權重衰減減少係數
            wd_min_ratio: 最小權重衰減比例
        """

        if not 0.0 <= eps:
            raise ValueError(f"無效的 epsilon 值：{eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"無效的 beta 參數（索引 0）：{betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"無效的 beta 參數（索引 1）：{betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"無效的 weight_decay 值：{weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        # 原有功能開關
        self.use_spd = use_spd
        self.spd_lambda = spd_lambda
        self.use_cautious = use_cautious
        self.use_orthogonal_grad = use_orthogonal_grad
        self.use_adopt_stability = use_adopt_stability
        self.use_grams = use_grams
        self.use_agr = use_agr
        self.use_tam = use_tam
        self.tam_beta = tam_beta

        # 動態自適應功能配置
        self.use_dynamic_adaptation = use_dynamic_adaptation
        self.adaptation_strength = adaptation_strength
        self.relationship_discovery_interval = relationship_discovery_interval
        self.importance_decay = importance_decay
        self.compatibility_threshold = compatibility_threshold

        # 動態權重衰減配置
        self.dynamic_weight_decay = dynamic_weight_decay
        self.wd_transition_steps = wd_transition_steps
        self.wd_decay_factor = wd_decay_factor
        self.wd_min_ratio = wd_min_ratio

        # 自適應優化器內部狀態
        self.parameter_relationships = {}  # 參數關係映射
        self.importance_scores = {}        # 參數重要性分數
        self.last_relationship_update = 0  # 上次關係更新的步數

        # 初始化參數組的元數據
        self._initialize_adaptive_metadata()

        # 初始化緩衝區池（記憶體優化）
        self._buffer_pool = {}
        self._max_buffers_per_shape = 3

        # 存儲初始參數（用於 SPD）
        if self.use_spd:
            self._store_initial_parameters()

        logger.info(f"HinaAdaptiveAdamWBF16 初始化完成，動態自適應: {use_dynamic_adaptation}")

    def _get_buffer(self, shape, dtype, device):
        """從緩衝區池獲取張量，如果不存在則創建新的"""
        key = (shape, dtype, device)

        if key in self._buffer_pool and self._buffer_pool[key]:
            return self._buffer_pool[key].pop()

        return torch.empty(shape, dtype=dtype, device=device)

    def _return_buffer(self, tensor):
        """將使用完的緩衝區歸還到池中"""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)

        if key not in self._buffer_pool:
            self._buffer_pool[key] = []

        if len(self._buffer_pool[key]) < self._max_buffers_per_shape:
            tensor.zero_()
            self._buffer_pool[key].append(tensor)

    def _initialize_adaptive_metadata(self):
        """初始化自適應版本的元數據結構"""
        self.param_groups_metadata = {}

        for group_idx, group in enumerate(self.param_groups):
            self.param_groups_metadata[group_idx] = {
                'param_count': len(group['params']),
                'param_list': list(group['params']),
                'grad_history': {},
                'adaptation_history': {}
            }

            for param in group['params']:
                param_id = id(param)
                self.importance_scores[param_id] = 1.0

                group_metadata = self.param_groups_metadata[group_idx]
                group_metadata['grad_history'][param_id] = []
                group_metadata['adaptation_history'][param_id] = {
                    'initial_norm': None,
                    'change_rate': 0.0,
                    'stability': 1.0
                }

    def _store_initial_parameters(self):
        """存儲初始參數以供 SPD 使用"""
        self.initial_params = {}
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param.requires_grad:
                    self.initial_params[param] = param.data.clone().detach()

    def _compute_parameter_contribution_score(self, param, state, group_metadata):
        """計算參數的實際貢獻度分數"""
        param_id = id(param)

        # 1. 梯度相關的貢獻度分析
        grad_contribution = 0.0
        if param.grad is not None:
            current_grad_norm = torch.norm(param.grad).item()
            grad_contribution = current_grad_norm

            grad_history = group_metadata['grad_history'].get(param_id, [])
            if len(grad_history) > 1:
                recent_norms = grad_history[-5:]
                mean_grad = torch.mean(torch.tensor(recent_norms))
                std_grad = torch.std(torch.tensor(recent_norms))
                grad_consistency = 1.0 - (std_grad / (mean_grad + 1e-8)).item()
                grad_contribution *= max(0.1, grad_consistency)

            grad_history.append(current_grad_norm)
            if len(grad_history) > 10:
                grad_history.pop(0)
            group_metadata['grad_history'][param_id] = grad_history

        # 2. 參數變化相關的貢獻度分析
        change_contribution = 0.0
        adaptation_info = group_metadata['adaptation_history'].get(param_id, {})

        if adaptation_info.get('initial_norm') is None:
            adaptation_info['initial_norm'] = torch.norm(param.data).item()
        else:
            current_norm = torch.norm(param.data).item()
            initial_norm = adaptation_info['initial_norm']
            if initial_norm > 0:
                change_rate = abs(current_norm - initial_norm) / initial_norm
                adaptation_info['change_rate'] = change_rate
                change_contribution = change_rate

        # 3. 參數內在特性分析
        param_variance = torch.var(param.data).item()
        param_sparsity = (param.data.abs() < 1e-6).float().mean().item()
        intrinsic_contribution = param_variance * (1.0 - param_sparsity)

        # 綜合貢獻度分數
        total_contribution = (
            grad_contribution * 0.4 +
            change_contribution * 0.3 +
            intrinsic_contribution * 0.3
        )

        return max(0.01, total_contribution)

    def _discover_parameter_relationships(self, group_metadata):
        """自動發現參數之間的潛在關聯"""
        param_list = group_metadata['param_list']
        new_relationships = {}

        for i, param1 in enumerate(param_list):
            if param1.dim() != 2:
                continue

            for j, param2 in enumerate(param_list[i+1:], i+1):
                if param2.dim() != 2:
                    continue

                compatibility = self._compute_parameter_compatibility(param1, param2)

                if compatibility > self.compatibility_threshold:
                    param1_id = id(param1)
                    param2_id = id(param2)

                    joint_importance = (
                        self.importance_scores.get(param1_id, 1.0) +
                        self.importance_scores.get(param2_id, 1.0)
                    ) / 2

                    new_relationships[param1_id] = {
                        'partner': param2,
                        'partner_id': param2_id,
                        'compatibility': compatibility,
                        'joint_importance': joint_importance,
                        'interaction_type': self._determine_interaction_type(param1, param2)
                    }

        return new_relationships

    def _compute_parameter_compatibility(self, param1, param2):
        """計算兩個參數的相容性"""
        if param1.dim() != 2 or param2.dim() != 2:
            return 0.0

        shape1, shape2 = param1.shape, param2.shape

        multiplication_checks = [
            shape1[1] == shape2[0],
            shape1[0] == shape2[1],
            shape1[1] == shape2[1],
            shape1[0] == shape2[0]
        ]

        if not any(multiplication_checks):
            return 0.0

        try:
            flat1 = param1.data.flatten()
            flat2 = param2.data.flatten()

            min_size = min(len(flat1), len(flat2))
            flat1 = flat1[:min_size]
            flat2 = flat2[:min_size]

            if min_size > 1:
                correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
                if torch.isnan(correlation):
                    correlation = 0.0
                else:
                    correlation = abs(correlation.item())
            else:
                correlation = 0.0

            shape_compatibility = sum(multiplication_checks) / len(multiplication_checks)
            total_compatibility = (shape_compatibility * 0.7 + correlation * 0.3)

            return total_compatibility

        except Exception as e:
            return 0.0

    def _determine_interaction_type(self, param1, param2):
        """確定兩個參數的最佳交互類型"""
        shape1, shape2 = param1.shape, param2.shape

        if shape1[1] == shape2[0]:
            return 'matmul_12'
        elif shape1[0] == shape2[1]:
            return 'matmul_21'
        elif shape1[1] == shape2[1]:
            return 'matmul_12t'
        elif shape1[0] == shape2[0]:
            return 'matmul_1t2'
        else:
            return 'norm_based'

    def _compute_adaptive_lr_scale(self, param, group_metadata, state):
        """基於動態關係和重要性的學習率調整"""
        if not self.use_dynamic_adaptation:
            return 1.0

        param_id = id(param)
        base_scale = 1.0

        # 基於重要性的調整
        importance = self.importance_scores.get(param_id, 1.0)
        importance_factor = min(3.0, max(0.1, importance * self.adaptation_strength))

        # 基於參數關係的調整
        if param_id in self.parameter_relationships:
            rel_info = self.parameter_relationships[param_id]
            partner = rel_info['partner']
            interaction_type = rel_info['interaction_type']

            try:
                interaction_matrix = self._compute_interaction_matrix(
                    param, partner, interaction_type
                )

                if interaction_matrix is not None:
                    interaction_norm = torch.norm(interaction_matrix).item()
                    compatibility_bonus = rel_info['compatibility']

                    interaction_scale = 1.0 / (1.0 + interaction_norm * 0.1)
                    compatibility_scale = 1.0 + compatibility_bonus * 0.2

                    paired_scale = interaction_scale * compatibility_scale
                    base_scale *= paired_scale

            except Exception as e:
                pass

        final_scale = base_scale * importance_factor
        final_scale = max(0.01, min(5.0, final_scale))

        return final_scale

    def _compute_interaction_matrix(self, param1, param2, interaction_type):
        """根據交互類型計算交互矩陣"""
        try:
            if interaction_type == 'matmul_12':
                return torch.matmul(param1, param2)
            elif interaction_type == 'matmul_21':
                return torch.matmul(param2, param1)
            elif interaction_type == 'matmul_12t':
                return torch.matmul(param1, param2.T)
            elif interaction_type == 'matmul_1t2':
                return torch.matmul(param1.T, param2)
            else:
                norm1 = torch.norm(param1)
                norm2 = torch.norm(param2)
                return torch.tensor(norm1 * norm2)

        except Exception as e:
            return None

    def _update_importance_scores(self, group_metadata):
        """更新所有參數的重要性分數"""
        for param in group_metadata['param_list']:
            param_id = id(param)

            current_contribution = self._compute_parameter_contribution_score(
                param, self.state.get(param, {}), group_metadata
            )

            old_importance = self.importance_scores.get(param_id, 1.0)
            new_importance = (
                self.importance_decay * old_importance +
                (1 - self.importance_decay) * current_contribution
            )

            self.importance_scores[param_id] = new_importance

    def _apply_spd_regularization(self, param, group, state):
        """應用選擇性投影衰減正則化"""
        if param not in self.initial_params:
            return 0

        initial_param = self.initial_params[param]
        if initial_param.device != param.data.device:
            initial_param = initial_param.to(param.data.device)
            self.initial_params[param] = initial_param

        param_diff = param.data - initial_param

        param_norm = torch.norm(param.data)
        diff_norm = torch.norm(param_diff)

        if param_norm > 0:
            bias_ratio = diff_norm / param_norm
        else:
            bias_ratio = 0

        spd_penalty = self.spd_lambda * bias_ratio * param_diff
        return spd_penalty

    def _apply_orthogonal_gradient(self, grad, param, temp_buffer=None):
        """應用正交梯度投影 - 記憶體優化版本"""
        param_norm = torch.norm(param.data, p=2)
        if param_norm <= 1e-30:
            return grad

        try:
            with torch.no_grad():
                original_shape = grad.shape

                param_flat = param.data.view(-1)
                grad_flat = grad.view(-1)

                if param_flat.shape != grad_flat.shape:
                    return grad

                grad_norm = torch.norm(grad_flat, p=2)
                if grad_norm <= 1e-30:
                    return grad

                dot_product = torch.dot(param_flat, grad_flat)
                param_norm_sq = torch.dot(param_flat, param_flat).add(1e-30)
                proj_coeff = dot_product / param_norm_sq

                if temp_buffer is not None and temp_buffer.shape == grad_flat.shape:
                    orthogonal_grad_flat = temp_buffer
                    orthogonal_grad_flat.copy_(grad_flat)
                    orthogonal_grad_flat.sub_(param_flat, alpha=proj_coeff)
                else:
                    orthogonal_grad_flat = grad_flat - proj_coeff * param_flat

                orth_norm = torch.norm(orthogonal_grad_flat, p=2).add(1e-30)
                scale_factor = grad_norm / orth_norm
                orthogonal_grad_flat.mul_(scale_factor)

                return orthogonal_grad_flat.view(original_shape)

        except Exception as e:
            return grad

    def _apply_agr_regularization(self, grad):
        """應用自適應梯度正則化"""
        grad_norm = torch.norm(grad)

        if grad_norm > 1.0:
            clip_factor = 1.0 / grad_norm
            return grad * clip_factor

        return grad

    def _apply_cautious_update(self, update, grad):
        """應用謹慎更新策略"""
        update_flat = update.view(-1)
        grad_flat = grad.view(-1)

        if torch.norm(update_flat) > 0 and torch.norm(grad_flat) > 0:
            alignment = torch.dot(update_flat, grad_flat) / (
                torch.norm(update_flat) * torch.norm(grad_flat)
            )

            if alignment < 0.1:
                return update * 0.5

        return update

    def _apply_tam_damping(self, momentum, grad, state):
        """應用力矩感知動量阻尼"""
        if 'momentum_alignment' not in state:
            state['momentum_alignment'] = 0.0

        try:
            if torch.norm(momentum) > 0 and torch.norm(grad) > 0:
                momentum_flat = momentum.view(-1)
                grad_flat = grad.view(-1)

                if momentum_flat.size() == grad_flat.size():
                    alignment = torch.dot(momentum_flat, grad_flat) / (
                        torch.norm(momentum) * torch.norm(grad)
                    )
                else:
                    alignment = 0.0
            else:
                alignment = 0.0
        except Exception as e:
            alignment = 0.0

        state['momentum_alignment'] = (
            self.tam_beta * state['momentum_alignment'] +
            (1 - self.tam_beta) * alignment
        )

        damping_factor = (1 + state['momentum_alignment']) / 2
        return damping_factor

    @torch.no_grad()
    def step(self, closure=None, zero_grad: bool = False):
        """執行優化步驟 - 結合 BF16 優化與自適應功能"""
        loss = None
        if closure is not None:
            loss = closure()

        # 全局步數計數
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        self.global_step += 1

        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            group_metadata = self.param_groups_metadata[group_idx]

            # 定期更新參數關係和重要性分數
            if (self.global_step - self.last_relationship_update >=
                self.relationship_discovery_interval):

                self._update_importance_scores(group_metadata)

                if self.use_dynamic_adaptation:
                    new_relationships = self._discover_parameter_relationships(group_metadata)
                    self.parameter_relationships.update(new_relationships)

                self.last_relationship_update = self.global_step

            # 為此參數群組創建臨時緩衝區池
            temp_buffers = {}

            for param in group["params"]:
                if param.grad is None:
                    continue

                # 確保參數為 bfloat16
                assert param.dtype == torch.bfloat16, "只支援 bfloat16 精度"

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamWBF16_HinaAdaptive 不支援稀疏梯度')

                # 確保梯度也是正確的精度
                if grad.dtype != torch.bfloat16:
                    grad = grad.to(dtype=torch.bfloat16)

                state = self.state[param]

                # 狀態初始化 - 確保所有狀態張量都是 bfloat16
                if len(state) == 0:
                    state["step"] = 0.0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format, dtype=torch.bfloat16
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format, dtype=torch.bfloat16
                    )
                    state["shift"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format, dtype=torch.bfloat16
                    )
                    state["accumulated_decay"] = float(
                        torch.rand([]) * self.decay_threshold
                    )

                    if self.use_adopt_stability:
                        state["exp_avg_sq_prev"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format, dtype=torch.bfloat16
                        )

                state["step"] += 1
                lr = group["lr"]

                # AGR 正則化
                if self.use_agr:
                    grad = self._apply_agr_regularization(grad)

                # 正交梯度投影
                if self.use_orthogonal_grad:
                    grad_flat_shape = grad.view(-1).shape
                    buffer_key = f"ortho_buffer_{grad_flat_shape}"
                    if buffer_key not in temp_buffers:
                        temp_buffers[buffer_key] = self._get_buffer(
                            grad_flat_shape, grad.dtype, grad.device
                        )

                    grad = self._apply_orthogonal_gradient(grad, param, temp_buffers[buffer_key])

                # 動態自適應學習率調整
                current_lr = lr
                if self.use_dynamic_adaptation:
                    lr_scale = self._compute_adaptive_lr_scale(param, group_metadata, state)
                    current_lr *= lr_scale

                # 累積權重衰減
                state["accumulated_decay"] += group["weight_decay"] * current_lr
                accum_decay = state["accumulated_decay"]
                decay_this_iteration = (
                    accum_decay > self.decay_threshold
                ) * accum_decay
                state["accumulated_decay"] -= decay_this_iteration

                # 動態權重衰減調整
                if self.dynamic_weight_decay:
                    if state["step"] > self.wd_transition_steps:
                        progress = (state["step"] - self.wd_transition_steps) / self.wd_transition_steps
                        decay_multiplier = max(
                            self.wd_min_ratio,
                            self.wd_decay_factor ** min(progress, 2.0)
                        )
                        decay_this_iteration *= decay_multiplier

                # Adam 動量更新（使用隨機舍入）
                if self.use_adopt_stability and "exp_avg_sq_prev" in state:
                    state["exp_avg_sq_prev"] = state["exp_avg_sq"].clone()

                # 確保所有計算都在正確的精度下進行
                state["exp_avg"].mul_(beta1)
                add_stochastic_(state["exp_avg"], grad, alpha=1 - beta1)

                # 對於二階矩，需要特別處理 bfloat16 的乘法
                grad_squared = grad * grad.conj()
                state["exp_avg_sq"].mul_(beta2)
                add_stochastic_(state["exp_avg_sq"], grad_squared, alpha=1 - beta2)

                # 計算分母校正
                denom_correction = (1 - beta2 ** state["step"]) ** 0.5

                # ADOPT 穩定性機制
                if self.use_adopt_stability and "exp_avg_sq_prev" in state:
                    denom = (torch.sqrt(torch.maximum(state["exp_avg_sq"], state["exp_avg_sq_prev"])) /
                            math.sqrt(1 - beta2 ** state["step"])).add_(group["eps"])
                else:
                    denom = (state["exp_avg_sq"].sqrt() /
                            math.sqrt(1 - beta2 ** state["step"])).add_(group["eps"])

                # 計算更新向量
                update = state["exp_avg"] / denom

                # TAM 阻尼
                if self.use_tam:
                    damping_factor = self._apply_tam_damping(state["exp_avg"], grad, state)
                    update = update * damping_factor

                # 謹慎更新
                if self.use_cautious:
                    update = self._apply_cautious_update(update, grad)

                # 使用隨機舍入更新 shift
                addcdiv_stochastic_(
                    state["shift"],
                    state["exp_avg"],
                    denom,
                    value=-current_lr * denom_correction,
                )

                # 應用參數更新（使用補償式累加）
                buffer = param.clone()
                add_stochastic_(param, state["shift"])
                add_stochastic_(state["shift"], buffer.sub_(param))

                # 權重衰減 - 確保正確的 bfloat16 處理
                if decay_this_iteration > 0:
                    # 創建權重衰減項並使用隨機舍入
                    weight_decay_term = param * decay_this_iteration
                    add_stochastic_(state["shift"], weight_decay_term, alpha=-1.0)

                # SPD 正則化
                if self.use_spd:
                    spd_penalty = self._apply_spd_regularization(param, group, state)
                    if isinstance(spd_penalty, torch.Tensor):
                        # 確保 SPD 懲罰項也使用隨機舍入
                        spd_scaled = spd_penalty * current_lr
                        add_stochastic_(state["shift"], spd_scaled, alpha=-1.0)

                if zero_grad:
                    grad.zero_()

            # 歸還此參數群組使用的緩衝區
            for buffer in temp_buffers.values():
                self._return_buffer(buffer)

        return loss

    def get_optimization_info(self) -> Dict[str, Any]:
        """獲取優化器的詳細信息，用於監控和調試"""
        info = {
            'optimizer_type': 'AdamWBF16_HinaAdaptive',
            'version': '自適應 BF16 專用版本',
            'precision': 'bfloat16',
            'total_params': sum(len(group['params']) for group in self.param_groups),
            'features': {
                'spd': self.use_spd,
                'cautious': self.use_cautious,
                'orthogonal_grad': self.use_orthogonal_grad,
                'adopt_stability': self.use_adopt_stability,
                'grams': self.use_grams,
                'agr': self.use_agr,
                'tam': self.use_tam,
                'dynamic_adaptation': self.use_dynamic_adaptation,
                'dynamic_weight_decay': self.dynamic_weight_decay,
                'stochastic_rounding': True,
                'compensated_summation': True,
                'delayed_weight_decay': True
            },
            'adaptation_config': {
                'adaptation_strength': self.adaptation_strength,
                'relationship_discovery_interval': self.relationship_discovery_interval,
                'importance_decay': self.importance_decay,
                'compatibility_threshold': self.compatibility_threshold,
                'decay_threshold': self.decay_threshold
            }
        }

        if hasattr(self, 'global_step'):
            info['training_stats'] = {
                'global_step': self.global_step,
                'total_relationships': len(self.parameter_relationships),
                'avg_importance_score': sum(self.importance_scores.values()) / max(1, len(self.importance_scores))
            }

        if hasattr(self, '_buffer_pool'):
            total_buffers = sum(len(buffers) for buffers in self._buffer_pool.values())
            info['memory_optimization'] = {
                'buffer_pool_types': len(self._buffer_pool),
                'total_cached_buffers': total_buffers,
                'max_buffers_per_shape': self._max_buffers_per_shape
            }

        return info

    def get_relationship_summary(self) -> Dict[str, Any]:
        """獲取參數關係的摘要信息"""
        if not self.parameter_relationships:
            return {'message': '尚未發現任何參數關係'}

        summary = {
            'total_relationships': len(self.parameter_relationships),
            'relationships': []
        }

        for param_id, rel_info in self.parameter_relationships.items():
            param_shape = None
            partner_shape = None

            for group_metadata in self.param_groups_metadata.values():
                for param in group_metadata['param_list']:
                    if id(param) == param_id:
                        param_shape = list(param.shape)
                    if id(param) == rel_info['partner_id']:
                        partner_shape = list(rel_info['partner'].shape)

            summary['relationships'].append({
                'param_shape': param_shape,
                'partner_shape': partner_shape,
                'compatibility': rel_info['compatibility'],
                'joint_importance': rel_info['joint_importance'],
                'interaction_type': rel_info['interaction_type']
            })

        return summary

    def get_importance_analysis(self) -> Dict[str, Any]:
        """獲取參數重要性分析報告"""
        if not self.importance_scores:
            return {'message': '尚未計算參數重要性分數'}

        scores = list(self.importance_scores.values())

        analysis = {
            'total_parameters': len(scores),
            'importance_statistics': {
                'mean': sum(scores) / len(scores),
                'max': max(scores),
                'min': min(scores),
                'std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
            },
            'high_importance_params': 0,
            'low_importance_params': 0
        }

        mean_importance = analysis['importance_statistics']['mean']
        for score in scores:
            if score > mean_importance * 1.5:
                analysis['high_importance_params'] += 1
            elif score < mean_importance * 0.5:
                analysis['low_importance_params'] += 1

        return analysis

    def clear_buffer_pool(self):
        """清理緩衝區池，釋放記憶體"""
        if hasattr(self, '_buffer_pool'):
            self._buffer_pool.clear()

    def get_buffer_pool_stats(self) -> Dict[str, Any]:
        """獲取緩衝區池的詳細統計信息"""
        if not hasattr(self, '_buffer_pool'):
            return {'message': '緩衝區池未初始化'}

        stats = {
            'total_buffer_types': len(self._buffer_pool),
            'buffer_details': {},
            'total_buffers': 0,
            'estimated_memory_mb': 0.0
        }

        for (shape, dtype, device), buffers in self._buffer_pool.items():
            buffer_count = len(buffers)
            if buffer_count > 0:
                element_size = 4 if dtype == torch.float32 else 8 if dtype == torch.float64 else 2
                buffer_size_mb = (torch.prod(torch.tensor(shape)).item() * element_size) / (1024 * 1024)
                total_size_mb = buffer_size_mb * buffer_count

                stats['buffer_details'][f"{shape}_{dtype}_{device}"] = {
                    'count': buffer_count,
                    'shape': shape,
                    'dtype': str(dtype),
                    'device': str(device),
                    'estimated_mb_per_buffer': buffer_size_mb,
                    'total_estimated_mb': total_size_mb
                }
                stats['total_buffers'] += buffer_count
                stats['estimated_memory_mb'] += total_size_mb

        return stats

    def update_device(self, device):
        """當模型被移動到新裝置時，更新優化器內部狀態"""
        if hasattr(self, 'initial_params'):
            for param, initial_param in self.initial_params.items():
                if initial_param.device != device:
                    self.initial_params[param] = initial_param.to(device)

        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    state[key] = value.to(device)

    def validate_bf16_compatibility(self) -> Dict[str, Any]:
        """驗證優化器的 BF16 兼容性並返回診斷信息"""
        issues = []
        warnings = []

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                param_info = f"參數組 {group_idx}, 參數 {param_idx}"

                # 檢查參數類型
                if param.dtype != torch.bfloat16:
                    issues.append(f"{param_info}: 參數不是 bfloat16 類型 ({param.dtype})")

                # 檢查梯度類型（如果存在）
                if param.grad is not None and param.grad.dtype != torch.bfloat16:
                    warnings.append(f"{param_info}: 梯度不是 bfloat16 類型 ({param.grad.dtype})")

                # 檢查狀態張量類型（如果已初始化）
                if param in self.state:
                    state = self.state[param]
                    for state_name, state_tensor in state.items():
                        if isinstance(state_tensor, torch.Tensor) and state_tensor.dtype != torch.bfloat16:
                            issues.append(f"{param_info}: 狀態 '{state_name}' 不是 bfloat16 類型 ({state_tensor.dtype})")

        return {
            'is_compatible': len(issues) == 0,
            'critical_issues': issues,
            'warnings': warnings,
            'total_params': sum(len(group['params']) for group in self.param_groups),
            'bf16_features_enabled': {
                'stochastic_rounding': True,
                'compensated_summation': True,
                'delayed_weight_decay': True
            }
        }

    def convert_to_bf16(self) -> None:
        """將優化器的所有參數和狀態轉換為 bfloat16"""
        for group in self.param_groups:
            for param in group['params']:
                if param.dtype != torch.bfloat16:
                    param.data = param.data.to(dtype=torch.bfloat16)
                    logger.info(f"將參數從 {param.dtype} 轉換為 bfloat16")

                if param.grad is not None and param.grad.dtype != torch.bfloat16:
                    param.grad.data = param.grad.data.to(dtype=torch.bfloat16)

                # 轉換狀態張量
                if param in self.state:
                    state = self.state[param]
                    for state_name, state_tensor in state.items():
                        if isinstance(state_tensor, torch.Tensor) and state_tensor.dtype != torch.bfloat16:
                            state[state_name] = state_tensor.to(dtype=torch.bfloat16)

        logger.info("已將所有優化器狀態轉換為 bfloat16")
