import torch
import torch.nn as nn
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit
from typing import Any, Dict, List, Tuple, Optional
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AdaptiveHinaAdamW(AdamW8bit):
    """
    自適應 HinaAdamW 優化器 - 基於動態貢獻度評估的自適應優化器

    1. 動態參數重要性評估
    2. 自適應參數關係發現
    3. 基於貢獻度的學習率調整
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
        # 動態自適應學習率功能
        use_dynamic_adaptation: bool = True,
        adaptation_strength: float = 1.0,
        relationship_discovery_interval: int = 100,  # 每 N 步重新發現參數關係
        importance_decay: float = 0.95,  # 重要性分數的衰減係數
        compatibility_threshold: float = 0.3,  # 參數相容性閾值
        # Dynamic weight decay configuration
        dynamic_weight_decay: bool = True,
        wd_transition_steps: int = 1000,
        wd_decay_factor: float = 0.7,
        wd_min_ratio: float = 0.1,
        **kwargs
    ):
        """
        初始化自適應 HinaAdamW 優化器

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

        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, optim_bits=optim_bits, args=args,
            min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
            block_wise=block_wise, is_paged=is_paged, **kwargs
        )

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

        # 存儲初始參數（用於 SPD）
        if self.use_spd:
            self._store_initial_parameters()

        logger.info(f"AdaptiveHinaAdamWOptimizer 初始化完成，動態自適應: {use_dynamic_adaptation}")

    def _initialize_adaptive_metadata(self):
        """初始化自適應版本的元數據結構"""
        self.param_groups_metadata = {}

        for group_idx, group in enumerate(self.param_groups):
            self.param_groups_metadata[group_idx] = {
                'param_count': len(group['params']),
                'param_list': list(group['params']),  # 參數列表
                'grad_history': {},      # 梯度歷史記錄
                'adaptation_history': {} # 自適應歷史記錄
            }

            # 為每個參數初始化追蹤信息
            for param in group['params']:
                param_id = id(param)
                self.importance_scores[param_id] = 1.0  # 初始重要性分數

                # 在組內記錄參數相關信息
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
        """
        計算參數的實際貢獻度分數

        基於多個維度評估參數的重要性：
        1. 梯度一致性和大小 - 評估參數在訓練過程中的活躍程度
        2. 參數變化率 - 衡量參數相對於初始值的變化幅度
        3. 參數的內在特性 - 包括方差、稀疏性等統計特性
        """
        param_id = id(param)

        # 1. 梯度相關的貢獻度分析
        grad_contribution = 0.0
        if param.grad is not None:
            current_grad_norm = torch.norm(param.grad).item()
            grad_contribution = current_grad_norm

            # 如果有梯度歷史，計算梯度的一致性
            grad_history = group_metadata['grad_history'].get(param_id, [])
            if len(grad_history) > 1:
                recent_norms = grad_history[-5:]  # 分析最近5步的梯度
                mean_grad = torch.mean(torch.tensor(recent_norms))
                std_grad = torch.std(torch.tensor(recent_norms))
                grad_consistency = 1.0 - (std_grad / (mean_grad + 1e-8)).item()
                grad_contribution *= max(0.1, grad_consistency)

            # 更新梯度歷史記錄
            grad_history.append(current_grad_norm)
            if len(grad_history) > 10:  # 只保持最近10步的歷史
                grad_history.pop(0)
            group_metadata['grad_history'][param_id] = grad_history

        # 2. 參數變化相關的貢獻度分析
        change_contribution = 0.0
        adaptation_info = group_metadata['adaptation_history'].get(param_id, {})

        if adaptation_info.get('initial_norm') is None:
            # 首次記錄，設定初始範數
            adaptation_info['initial_norm'] = torch.norm(param.data).item()
        else:
            # 計算相對於初始值的變化率
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

        # 綜合貢獻度分數（加權組合）
        # TODO: 加權組合的貢獻度是否需要調整？需要有更多的測試。
        total_contribution = (
            grad_contribution * 0.4 +      # 梯度貢獻 40%
            change_contribution * 0.3 +    # 變化貢獻 30%
            intrinsic_contribution * 0.3   # 內在貢獻 30%
        )

        return max(0.01, total_contribution)  # 確保最小值避免除零

    def _discover_parameter_relationships(self, group_metadata):
        """
        自動發現參數之間的潛在關聯，模擬 LoRA 配對機制

        這個方法會：
        1. 分析參數間的矩陣相容性（可否進行矩陣運算）
        2. 計算參數的語意相似性（基於分佈特徵）
        3. 建立動態的參數關係映射
        """
        param_list = group_metadata['param_list']
        new_relationships = {}

        for i, param1 in enumerate(param_list):
            if param1.dim() != 2:  # 只處理 2D 參數（矩陣）
                continue

            for j, param2 in enumerate(param_list[i+1:], i+1):
                if param2.dim() != 2:
                    continue

                # 計算兩個參數的相容性
                compatibility = self._compute_parameter_compatibility(param1, param2)

                if compatibility > self.compatibility_threshold:
                    param1_id = id(param1)
                    param2_id = id(param2)

                    # 計算聯合重要性分數
                    joint_importance = (
                        self.importance_scores.get(param1_id, 1.0) +
                        self.importance_scores.get(param2_id, 1.0)
                    ) / 2

                    # 建立關係記錄
                    new_relationships[param1_id] = {
                        'partner': param2,
                        'partner_id': param2_id,
                        'compatibility': compatibility,
                        'joint_importance': joint_importance,
                        'interaction_type': self._determine_interaction_type(param1, param2)
                    }

                    logger.debug(f"發現參數關係: {param1.shape} <-> {param2.shape}, "
                               f"相容性: {compatibility:.3f}")

        return new_relationships

    def _compute_parameter_compatibility(self, param1, param2):
        """
        計算兩個參數的相容性（是否適合配對）

        相容性基於兩個維度：
        1. 形狀相容性 - 是否可以進行矩陣運算
        2. 語意相似性 - 基於參數分佈的相關性
        """
        if param1.dim() != 2 or param2.dim() != 2:
            return 0.0

        shape1, shape2 = param1.shape, param2.shape

        # 檢查矩陣乘法的各種可能性
        multiplication_checks = [
            shape1[1] == shape2[0],  # param1 @ param2
            shape1[0] == shape2[1],  # param2 @ param1
            shape1[1] == shape2[1],  # param1 @ param2^T
            shape1[0] == shape2[0]   # param1^T @ param2
        ]

        if not any(multiplication_checks):
            return 0.0

        # 計算語意相似性（基於參數分佈相關性）
        try:
            flat1 = param1.data.flatten()
            flat2 = param2.data.flatten()

            # 如果參數大小不同，截取較小的大小進行比較
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

            # 形狀相容性分數
            shape_compatibility = sum(multiplication_checks) / len(multiplication_checks)

            # 綜合相容性分數
            total_compatibility = (shape_compatibility * 0.7 + correlation * 0.3)

            return total_compatibility

        except Exception as e:
            logger.warning(f"計算參數相容性時發生錯誤: {e}")
            return 0.0

    def _determine_interaction_type(self, param1, param2):
        """確定兩個參數的最佳交互類型"""
        shape1, shape2 = param1.shape, param2.shape

        if shape1[1] == shape2[0]:
            return 'matmul_12'  # param1 @ param2
        elif shape1[0] == shape2[1]:
            return 'matmul_21'  # param2 @ param1
        elif shape1[1] == shape2[1]:
            return 'matmul_12t' # param1 @ param2^T
        elif shape1[0] == shape2[0]:
            return 'matmul_1t2' # param1^T @ param2
        else:
            return 'norm_based' # 基於範數的交互

    def _compute_adaptive_lr_scale(self, param, group_metadata, state):
        """
        基於動態關係和重要性的學習率調整

        這是自適應版本的核心創新：完全不依賴預定義的參數類型，
        而是基於動態發現的關係和重要性進行學習率調整。
        """
        if not self.use_dynamic_adaptation:
            return 1.0

        param_id = id(param)
        base_scale = 1.0

        # 1. 基於重要性的調整
        importance = self.importance_scores.get(param_id, 1.0)
        importance_factor = min(3.0, max(0.1, importance * self.adaptation_strength))

        # 2. 基於參數關係的調整
        if param_id in self.parameter_relationships:
            rel_info = self.parameter_relationships[param_id]
            partner = rel_info['partner']
            interaction_type = rel_info['interaction_type']

            try:
                # 根據交互類型計算交互矩陣
                interaction_matrix = self._compute_interaction_matrix(
                    param, partner, interaction_type
                )

                if interaction_matrix is not None:
                    interaction_norm = torch.norm(interaction_matrix).item()
                    compatibility_bonus = rel_info['compatibility']

                    # 動態調整公式（受 ALoRA 論文啟發）
                    interaction_scale = 1.0 / (1.0 + interaction_norm * 0.1)
                    compatibility_scale = 1.0 + compatibility_bonus * 0.2

                    paired_scale = interaction_scale * compatibility_scale
                    base_scale *= paired_scale

                    logger.debug(f"參數 {param.shape} 配對調整: "
                               f"交互={interaction_scale:.3f}, "
                               f"相容性={compatibility_scale:.3f}")

            except Exception as e:
                logger.warning(f"計算配對效應時發生錯誤: {e}")

        # 3. 應用重要性加權
        final_scale = base_scale * importance_factor

        # 4. 穩定性約束（避免極端值）
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
            else:  # norm_based
                # 使用參數範數的組合作為交互強度
                norm1 = torch.norm(param1)
                norm2 = torch.norm(param2)
                return torch.tensor(norm1 * norm2)

        except Exception as e:
            logger.warning(f"交互矩陣計算失敗 ({interaction_type}): {e}")
            return None

    def _update_importance_scores(self, group_metadata):
        """更新所有參數的重要性分數"""
        for param in group_metadata['param_list']:
            param_id = id(param)

            # 計算新的貢獻度
            current_contribution = self._compute_parameter_contribution_score(
                param, self.state.get(param, {}), group_metadata
            )

            # 使用指數移動平均更新重要性分數
            old_importance = self.importance_scores.get(param_id, 1.0)
            new_importance = (
                self.importance_decay * old_importance +
                (1 - self.importance_decay) * current_contribution
            )

            self.importance_scores[param_id] = new_importance

    def _apply_spd_regularization(self, param, group, state):
        """應用 Selective Projection Decay 正則化"""
        if param not in self.initial_params:
            return 0

        initial_param = self.initial_params[param]
        # 確保 initial_param 與 param.data 在同一個裝置上
        if initial_param.device != param.data.device:
            initial_param = initial_param.to(param.data.device)
            self.initial_params[param] = initial_param

        param_diff = param.data - initial_param

        # 計算偏差比率
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

        # 計算梯度的正交投影
        try:
            U, S, V = torch.svd(param.data)

            # 將梯度投影到參數的零空間
            grad_flat = grad.view(-1)
            U_flat = U.view(U.size(0), -1)

            if U_flat.size(1) == grad_flat.size(0):
                proj = torch.mm(U_flat.T, grad_flat.unsqueeze(1))
                orthogonal_grad = grad - torch.mm(U_flat, proj).view(grad.shape)
                return orthogonal_grad
        except Exception as e:
            logger.warning(f"正交梯度投影失敗: {e}")

        return grad

    def _apply_agr_regularization(self, grad):
        """應用 Adaptive Gradient Regularization"""
        grad_norm = torch.norm(grad)

        if grad_norm > 1.0:
            # 自適應梯度裁剪
            clip_factor = 1.0 / grad_norm
            return grad * clip_factor

        return grad

    def _apply_cautious_update(self, update, grad):
        """應用謹慎更新策略"""
        # 計算更新與梯度的對齊程度
        update_flat = update.view(-1)
        grad_flat = grad.view(-1)

        if torch.norm(update_flat) > 0 and torch.norm(grad_flat) > 0:
            alignment = torch.dot(update_flat, grad_flat) / (
                torch.norm(update_flat) * torch.norm(grad_flat)
            )

            # 如果對齊程度較差，減小更新步長
            if alignment < 0.1:
                return update * 0.5

        return update

    def _apply_tam_damping(self, momentum, grad, state):
        """應用 Torque-Aware Momentum 阻尼"""
        if 'momentum_alignment' not in state:
            state['momentum_alignment'] = 0.0

        # 計算梯度與動量的對齊程度
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
            logger.warning(f"TAM: 計算對齊度時發生錯誤: {e}")
            alignment = 0.0

        # 平滑對齊估計
        state['momentum_alignment'] = (
            self.tam_beta * state['momentum_alignment'] +
            (1 - self.tam_beta) * alignment
        )

        # 計算阻尼因子
        damping_factor = (1 + state['momentum_alignment']) / 2
        return damping_factor

    def update_device(self, device):
        """當模型被移動到新裝置時，更新優化器內部狀態"""
        if hasattr(self, 'initial_params'):
            for param, initial_param in self.initial_params.items():
                if initial_param.device != device:
                    self.initial_params[param] = initial_param.to(device)

        # 更新所有狀態中的張量
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    state[key] = value.to(device)

    def step(self, closure=None):
        """執行優化步驟 - 自適應版本的核心邏輯"""
        loss = None
        if closure is not None:
            loss = closure()

        # 全局步數計數（用於定期更新關係）
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        self.global_step += 1

        for group_idx, group in enumerate(self.param_groups):
            group_metadata = self.param_groups_metadata[group_idx]

            # 定期更新參數關係和重要性分數
            if (self.global_step - self.last_relationship_update >=
                self.relationship_discovery_interval):

                logger.debug(f"第 {self.global_step} 步：更新參數關係和重要性分數")

                # 更新重要性分數
                self._update_importance_scores(group_metadata)

                # 重新發現參數關係
                if self.use_dynamic_adaptation:
                    new_relationships = self._discover_parameter_relationships(group_metadata)
                    self.parameter_relationships.update(new_relationships)

                self.last_relationship_update = self.global_step

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaptiveHinaAdamWOptimizer 不支援稀疏梯度')

                state = self.state[param]

                # 狀態初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)
                    if self.use_adopt_stability:
                        state['exp_avg_sq_prev'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)
                else:
                    # 檢查狀態張量形狀是否匹配
                    if state['exp_avg'].shape != param.data.shape:
                        logger.warning(f"狀態張量形狀不匹配，重新初始化參數 {param.data.shape}")
                        state['exp_avg'] = torch.zeros_like(param.data)
                        state['exp_avg_sq'] = torch.zeros_like(param.data)
                        if self.use_adopt_stability:
                            state['exp_avg_sq_prev'] = torch.zeros_like(param.data)

                state['step'] += 1

                beta1, beta2 = group['betas']
                step_size = group['lr']

                # AGR 正則化
                if self.use_agr:
                    grad = self._apply_agr_regularization(grad)

                # 正交梯度投影
                if self.use_orthogonal_grad:
                    grad = self._apply_orthogonal_gradient(grad, param)

                # 偏差校正的學習率
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 更新一階和二階動量估計
                if self.use_adopt_stability:
                    if 'exp_avg_sq_prev' in state:
                        state['exp_avg_sq_prev'] = state['exp_avg_sq'].clone()

                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 計算更新
                if self.use_adopt_stability and 'exp_avg_sq_prev' in state:
                    denom = (torch.sqrt(torch.maximum(state['exp_avg_sq'], state['exp_avg_sq_prev'])) /
                            math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                update = (state['exp_avg'] / bias_correction1) / denom

                # TAM 阻尼
                if self.use_tam:
                    damping_factor = self._apply_tam_damping(state['exp_avg'], grad, state)
                    update = update * damping_factor

                # 謹慎更新
                if self.use_cautious:
                    update = self._apply_cautious_update(update, grad)

                # 核心功能：動態自適應學習率調整
                current_step_size = step_size
                if self.use_dynamic_adaptation:
                    lr_scale = self._compute_adaptive_lr_scale(param, group_metadata, state)
                    current_step_size *= lr_scale

                    if lr_scale != 1.0:
                        logger.debug(f"參數 {param.shape} 學習率調整: {lr_scale:.4f}")

                # 應用更新
                param.data.add_(update, alpha=-current_step_size)

                # 權重衰減
                current_weight_decay = group['weight_decay']

                # 動態權重衰減
                if self.dynamic_weight_decay:
                    if state['step'] > self.wd_transition_steps:
                        progress = (state['step'] - self.wd_transition_steps) / self.wd_transition_steps
                        decay_multiplier = max(
                            self.wd_min_ratio,
                            self.wd_decay_factor ** min(progress, 2.0)
                        )
                        current_weight_decay *= decay_multiplier

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
            'optimizer_type': 'AdaptiveHinaAdamWOptimizer',
            'version': '自適應版本 - 無參數類型依賴',
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
                'dynamic_weight_decay': self.dynamic_weight_decay
            },
            'adaptation_config': {
                'adaptation_strength': self.adaptation_strength,
                'relationship_discovery_interval': self.relationship_discovery_interval,
                'importance_decay': self.importance_decay,
                'compatibility_threshold': self.compatibility_threshold
            }
        }

        # 添加動態統計信息
        if hasattr(self, 'global_step'):
            info['training_stats'] = {
                'global_step': self.global_step,
                'total_relationships': len(self.parameter_relationships),
                'avg_importance_score': sum(self.importance_scores.values()) / max(1, len(self.importance_scores))
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
            # 找到參數形狀
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

        # 統計重要性分數
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

        # 分類高/低重要性參數
        mean_importance = analysis['importance_statistics']['mean']
        for score in scores:
            if score > mean_importance * 1.5:
                analysis['high_importance_params'] += 1
            elif score < mean_importance * 0.5:
                analysis['low_importance_params'] += 1

        return analysis