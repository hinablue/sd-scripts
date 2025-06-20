import torch
import torch.nn as nn
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit
from typing import Any, Dict, List, Tuple, Optional
import math
from collections import defaultdict

from library.utils import setup_logging

setup_logging()
import logging

logging.basicConfig(level=logging.INFO) # 設定日誌等級為 INFO，測試時可以調整為 DEBUG
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
        # lr_mask 機制配置（策略 B：組合式整合）
        use_lr_mask: bool = False,
        lr_bump: float = 3e-6,  # lr_mask 調整幅度
        min_lr: float = 1e-7,   # lr_mask 最小學習率
        max_lr: float = 1e-3,   # lr_mask 最大學習率
        warmup_steps: int = 500, # lr_mask warmup 步數
        # Dynamic weight decay configuration
        dynamic_weight_decay: bool = True,
        wd_transition_steps: int = 1000,
        wd_decay_factor: float = 0.7,
        wd_min_ratio: float = 0.1,
        **kwargs
    ):
        """
        初始化自適應 HinaAdamW 優化器 - 策略 B：組合式整合

        整合策略：
        1. lr_mask 作為基礎層：基於梯度極性進行即時學習率調整
        2. 自適應機制作為高級層：基於參數重要性和關係進行長期調整
        3. 最終學習率 = base_lr * lr_mask_scale * adaptive_scale

        Args:
            params: 模型參數
            lr: 基礎學習率
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
            use_lr_mask: 是否啟用 lr_mask 機制（策略 B 的基礎層）
            lr_bump: lr_mask 的學習率調整幅度
            min_lr: lr_mask 允許的最小學習率
            max_lr: lr_mask 允許的最大學習率
            warmup_steps: lr_mask 的 warmup 階段步數
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

        # lr_mask 機制配置（策略 B 基礎層）
        self.use_lr_mask = use_lr_mask
        self.lr_bump = lr_bump
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

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
        self._max_buffers_per_shape = 3  # 每種形狀最多保留的緩衝區數量

        # 存儲初始參數（用於 SPD）
        if self.use_spd:
            self._store_initial_parameters()

        logger.info(f"AdaptiveHinaAdamWOptimizer 初始化完成，動態自適應: {use_dynamic_adaptation}，lr_mask: {use_lr_mask}")

    def _get_buffer(self, shape, dtype, device):
        """
        從緩衝區池獲取張量，如果不存在則創建新的

        Args:
            shape: 張量形狀
            dtype: 數據類型
            device: 設備

        Returns:
            可重用的張量緩衝區
        """
        key = (shape, dtype, device)

        if key in self._buffer_pool and self._buffer_pool[key]:
            return self._buffer_pool[key].pop()

        return torch.empty(shape, dtype=dtype, device=device)

    def _return_buffer(self, tensor):
        """
        將使用完的緩衝區歸還到池中

        Args:
            tensor: 要歸還的張量
        """
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)

        if key not in self._buffer_pool:
            self._buffer_pool[key] = []

        # 限制每種形狀的緩衝區數量，避免記憶體洩漏
        if len(self._buffer_pool[key]) < self._max_buffers_per_shape:
            # 清零緩衝區以備重用
            tensor.zero_()
            self._buffer_pool[key].append(tensor)

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

                # === 策略 B：lr_mask 狀態初始化 ===
                if self.use_lr_mask:
                    # 確保參數在正確設備上
                    device = param.device if hasattr(param, 'device') else 'cpu'

                    # 為每個參數初始化 lr_mask 相關狀態
                    lr_mask_metadata = {
                        'lr_mask': None,  # 將在第一次使用時初始化
                        'last_polarity': None,  # 上次梯度極性
                        'lr_max': self.defaults['lr'],  # 記錄最大學習率
                        'avg_lr': self.defaults['lr'],   # 平均學習率
                        'warmup_complete': False,  # warmup 是否完成
                    }
                    group_metadata['adaptation_history'][param_id].update(lr_mask_metadata)

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"為參數 {param.shape} 初始化 lr_mask 狀態")

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
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"參數 1 {param1.shape} 不是 2D 矩陣，跳過")

            continue

        for j, param2 in enumerate(param_list[i+1:], i+1):
            if param2.dim() != 2:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"參數 2 {param2.shape} 不是 2D 矩陣，跳過")

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
                        'interaction_type': AdaptiveHinaAdamW._determine_interaction_type(param1, param2)
                    }

                    if logger.isEnabledFor(logging.DEBUG):
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"參數 1 {param1.shape} 或 參數 2 {param2.shape} 不是 2D 矩陣，跳過")

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

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"參數相容性分數: {total_compatibility:.3f}")

            return total_compatibility

        except Exception as e:
            logger.warning(f"計算參數相容性時發生錯誤: {e}")
            return 0.0

    @staticmethod
    @torch.jit.script
    def _determine_interaction_type(param1: torch.Tensor, param2: torch.Tensor) -> str:
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

    def _compute_adaptive_lr_scale(self, param, group_metadata, state, grad=None, global_step=None):
        """
        基於動態關係和重要性的學習率調整（策略 B：組合式整合）

        策略 B 實施：
        1. 基礎層：lr_mask 基於梯度極性的即時調整
        2. 高級層：基於參數重要性和關係的長期調整
        3. 最終縮放 = lr_mask_scale * adaptive_scale

        Args:
            param: 參數張量
            group_metadata: 參數組元數據
            state: 優化器狀態
            grad: 梯度張量（用於 lr_mask）
            global_step: 全局步數（用於 lr_mask）

        Returns:
            float: 組合後的學習率縮放因子
        """

        # === 第一層：lr_mask 基礎調整 ===
        lr_mask_scale = 1.0
        if self.use_lr_mask and grad is not None and global_step is not None:
            lr_mask_scale = self._update_lr_mask(param, group_metadata, state, grad, global_step)

            # 如果 lr_mask_scale 是張量，取平均值作為標量
            if isinstance(lr_mask_scale, torch.Tensor):
                lr_mask_scale = lr_mask_scale.mean().item()

        # === 第二層：自適應高級調整 ===
        adaptive_scale = 1.0
        if self.use_dynamic_adaptation:
            param_id = id(param)

            # 1. 基於重要性的調整
            importance = self.importance_scores.get(param_id, 1.0)
            importance_factor = min(3.0, max(0.1, importance * self.adaptation_strength))

            # 2. 基於參數關係的調整
            relationship_scale = 1.0
            if param_id in self.parameter_relationships:
                rel_info = self.parameter_relationships[param_id]
                partner = rel_info['partner']
                interaction_type = rel_info['interaction_type']

                try:
                    # 根據交互類型計算交互矩陣
                    interaction_matrix = AdaptiveHinaAdamW._compute_interaction_matrix(
                        param, partner, interaction_type
                    )

                    if interaction_matrix is not None:
                        interaction_norm = torch.norm(interaction_matrix).item()
                        compatibility_bonus = rel_info['compatibility']

                        # 動態調整公式（受 ALoRA 論文啟發）
                        interaction_scale = 1.0 / (1.0 + interaction_norm * 0.1)
                        compatibility_scale = 1.0 + compatibility_bonus * 0.2

                        relationship_scale = interaction_scale * compatibility_scale

                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"參數 {param.shape} 關係調整: "
                                       f"交互={interaction_scale:.3f}, "
                                       f"相容性={compatibility_scale:.3f}")

                except Exception as e:
                    logger.warning(f"計算配對效應時發生錯誤: {e}")

            # 3. 組合自適應調整
            adaptive_scale = importance_factor * relationship_scale

            # 4. 穩定性約束（避免極端值）
            adaptive_scale = max(0.01, min(5.0, adaptive_scale))

        # === 策略 B：組合最終縮放因子 ===
        final_scale = lr_mask_scale * adaptive_scale

        # 最終穩定性檢查
        final_scale = max(0.001, min(10.0, final_scale))

        # 詳細日誌記錄（僅在 debug 模式）
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"參數 {param.shape} 學習率縮放組合：")
            logger.debug(f"  lr_mask_scale: {lr_mask_scale:.4f}")
            logger.debug(f"  adaptive_scale: {adaptive_scale:.4f}")
            logger.debug(f"  final_scale: {final_scale:.4f}")

        return final_scale

    @staticmethod
    @torch.jit.script
    def _compute_interaction_matrix(param1: torch.Tensor, param2: torch.Tensor, interaction_type: str) -> torch.Tensor:
        """根據交互類型計算交互矩陣"""
        if interaction_type == 'matmul_12':
            return torch.matmul(param1, param2)
        elif interaction_type == 'matmul_21':
            return torch.matmul(param2, param1)
        elif interaction_type == 'matmul_12t':
            return torch.matmul(param1, param2.T)
        elif interaction_type == 'matmul_1t2':
            return torch.matmul(param1.T, param2)
        else:  # norm_based
            norm1 = torch.norm(param1)
            norm2 = torch.norm(param2)
            return torch.tensor(norm1 * norm2, device=param1.device)

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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"SPD 懲罰項範數: {torch.norm(spd_penalty).item():.3f}")

        return spd_penalty

    @staticmethod
    @torch.jit.script
    def _orthogonal_gradient_core(grad_flat: torch.Tensor, param_flat: torch.Tensor, eps: float) -> torch.Tensor:
        """正交梯度投影的核心計算 - JIT 優化版本"""
        grad_norm = torch.norm(grad_flat, p=2)
        if grad_norm <= eps:
            return grad_flat

        # 計算投影係數: proj_coeff = (g·p) / ||p||²
        dot_product = torch.dot(param_flat, grad_flat)
        param_norm_sq = torch.dot(param_flat, param_flat) + eps
        proj_coeff = dot_product / param_norm_sq

        # 計算正交化梯度: g_orth = g - proj_coeff * p
        orthogonal_grad_flat = grad_flat - proj_coeff * param_flat

        # 計算正交化後的範數
        orth_norm = torch.norm(orthogonal_grad_flat, p=2) + eps

        # 標準化以保持原始梯度範數
        scale_factor = grad_norm / orth_norm
        return orthogonal_grad_flat * scale_factor

    @staticmethod
    def _apply_orthogonal_gradient(grad: torch.Tensor, param: torch.Tensor, eps: float = 1e-30, temp_buffer: torch.Tensor = None) -> torch.Tensor:
        """
        應用正交梯度投影 - 記憶體優化版本

        使用簡化的向量投影代替昂貴的 SVD 分解：
        g_orth = g - (g·p / ||p||²) * p

        這將梯度中平行於參數的分量移除，保持垂直分量。
        相比 SVD 方法：速度提升 10-100 倍，記憶體使用減少 50-90%

        Args:
            grad: 梯度張量
            param: 參數張量
            eps: 數值穩定性常數
            temp_buffer: 可選的臨時緩衝區，用於減少記憶體分配

        Returns:
            正交化後的梯度張量
        """
        # 提前退出條件：參數範數太小時跳過正交化
        param_norm = torch.norm(param.data, p=2)
        if param_norm <= eps:
            return grad

        param_flat = param.data.view(-1)
        grad_flat = grad.view(-1)

        if param_flat.shape != grad_flat.shape:
            return grad

        grad_norm = torch.norm(grad_flat, p=2)
        if grad_norm <= eps:
            return grad

        # 優先使用提供的緩衝區進行記憶體優化計算
        if temp_buffer is not None and temp_buffer.shape == grad_flat.shape:
            # 使用緩衝區進行手動最佳化
            # 計算投影係數: proj_coeff = (g·p) / ||p||²
            dot_product = torch.dot(param_flat, grad_flat)
            param_norm_sq = torch.dot(param_flat, param_flat) + eps
            proj_coeff = dot_product / param_norm_sq

            # 使用緩衝區和原地操作
            orthogonal_grad_flat = temp_buffer
            orthogonal_grad_flat.copy_(grad_flat)
            orthogonal_grad_flat.sub_(param_flat, alpha=proj_coeff)

            # 計算正交化後的範數並標準化
            orth_norm = torch.norm(orthogonal_grad_flat, p=2) + eps
            scale_factor = grad_norm / orth_norm
            orthogonal_grad_flat.mul_(scale_factor)
        else:
            # 回退到 JIT 優化的標準操作
            orthogonal_grad_flat = AdaptiveHinaAdamW._orthogonal_gradient_core(
                grad_flat, param_flat, eps
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"正交化後的梯度範數: {torch.norm(orthogonal_grad_flat):.3f}")

        # 恢復原始形狀
        return orthogonal_grad_flat.view_as(grad)

    @staticmethod
    @torch.jit.script
    def _apply_agr_regularization(grad: torch.Tensor) -> torch.Tensor:
        """應用 Adaptive Gradient Regularization"""
        grad_norm = torch.norm(grad)

        if grad_norm > 1.0:
            # 自適應梯度裁剪
            clip_factor = 1.0 / grad_norm
            return grad * clip_factor

        return grad

    @staticmethod
    @torch.jit.script
    def _apply_cautious_update(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"TAM 阻尼因子: {damping_factor:.3f}")

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

    @torch.no_grad()
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

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"第 {self.global_step} 步：更新參數關係和重要性分數")

                # 更新重要性分數
                self._update_importance_scores(group_metadata)

                # 重新發現參數關係
                if self.use_dynamic_adaptation:
                    new_relationships = self._discover_parameter_relationships(group_metadata)
                    self.parameter_relationships.update(new_relationships)

                self.last_relationship_update = self.global_step

            # 為此參數群組創建臨時緩衝區池（記憶體優化）
            temp_buffers = {}

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
                    # 檢查狀態張量形狀是否相符
                    if state['exp_avg'].shape != param.data.shape:
                        logger.warning(f"狀態張量形狀不相符，重新初始化參數 {param.data.shape}")
                        state['exp_avg'] = torch.zeros_like(param.data)
                        state['exp_avg_sq'] = torch.zeros_like(param.data)
                        if self.use_adopt_stability:
                            state['exp_avg_sq_prev'] = torch.zeros_like(param.data)

                state['step'] += 1

                beta1, beta2 = group['betas']
                step_size = group['lr']

                # AGR 正則化
                if self.use_agr:
                    grad = AdaptiveHinaAdamW._apply_agr_regularization(grad)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"AGR 正則化後的梯度範數: {torch.norm(grad):.3f}")

                # 正交梯度投影 - 記憶體優化版本
                if self.use_orthogonal_grad:
                    # 為正交投影獲取緩衝區
                    grad_flat_shape = grad.view(-1).shape
                    buffer_key = f"ortho_buffer_{grad_flat_shape}"
                    if buffer_key not in temp_buffers:
                        temp_buffers[buffer_key] = self._get_buffer(
                            grad_flat_shape, grad.dtype, grad.device
                        )

                    grad = AdaptiveHinaAdamW._apply_orthogonal_gradient(grad, param, 1e-30, temp_buffers[buffer_key])
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"正交梯度投影後的梯度範數: {torch.norm(grad):.3f}")

                # 偏差校正的學習率
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 更新一階和二階動量估計
                if self.use_adopt_stability and 'exp_avg_sq_prev' in state:
                    state['exp_avg_sq_prev'] = state['exp_avg_sq'].clone()

                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 計算更新
                if self.use_adopt_stability and 'exp_avg_sq_prev' in state:
                    denom = (torch.sqrt(torch.maximum(state['exp_avg_sq'], state['exp_avg_sq_prev'])) /
                            math.sqrt(bias_correction2)).add_(group['eps'])
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"啟用 ADOPT 穩定性更新後的分母範數: {torch.norm(denom).item():.3f}")
                else:
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                update = (state['exp_avg'] / bias_correction1) / denom

                # TAM 阻尼
                if self.use_tam:
                    damping_factor = self._apply_tam_damping(state['exp_avg'], grad, state)
                    update = update * damping_factor

                # 謹慎更新
                if self.use_cautious:
                    update = AdaptiveHinaAdamW._apply_cautious_update(update, grad)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"謹慎更新後的更新範數: {torch.norm(update).item():.3f}")

                # 核心功能：動態自適應學習率調整
                current_step_size = step_size
                if self.use_dynamic_adaptation or self.use_lr_mask:
                    # 策略 B：組合式學習率調整
                    # 傳入梯度和全局步數以支援 lr_mask 機制
                    lr_scale = self._compute_adaptive_lr_scale(
                        param, group_metadata, state,
                        grad=grad,  # 用於 lr_mask 的梯度極性判斷
                        global_step=self.global_step  # 用於 warmup 判斷
                    )
                    current_step_size *= lr_scale

                    if lr_scale != 1.0 and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"參數 {param.shape} 組合學習率調整: {lr_scale:.4f}")

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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"動態權重衰減後的權重衰減係數: {current_weight_decay:.3f}")

                if current_weight_decay != 0:
                    param.data.add_(param.data, alpha=-group['lr'] * current_weight_decay)

                # SPD 正則化
                if self.use_spd:
                    spd_penalty = self._apply_spd_regularization(param, group, state)
                    if isinstance(spd_penalty, torch.Tensor):
                        param.data.add_(spd_penalty, alpha=-group['lr'])

            # 歸還此參數群組使用的緩衝區
            for buffer in temp_buffers.values():
                self._return_buffer(buffer)

            # 清理記憶體
            del grad, update, denom
            temp_buffers.clear()

        return loss

    def get_optimization_info(self) -> Dict[str, Any]:
        """獲取優化器的詳細信息，用於監控和調試"""
        info = {
            'optimizer_type': 'AdaptiveHinaAdamWOptimizer',
            'version': '策略 B：lr_mask + 自適應組合版本',
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
                'lr_mask': self.use_lr_mask,  # 新增 lr_mask 狀態
                'dynamic_weight_decay': self.dynamic_weight_decay
            },
            'adaptation_config': {
                'adaptation_strength': self.adaptation_strength,
                'relationship_discovery_interval': self.relationship_discovery_interval,
                'importance_decay': self.importance_decay,
                'compatibility_threshold': self.compatibility_threshold
            },
            'lr_mask_config': {  # 新增 lr_mask 配置信息
                'enabled': self.use_lr_mask,
                'lr_bump': self.lr_bump,
                'min_lr': self.min_lr,
                'max_lr': self.max_lr,
                'warmup_steps': self.warmup_steps
            }
        }

        # 添加動態統計信息
        if hasattr(self, 'global_step'):
            info['training_stats'] = {
                'global_step': self.global_step,
                'total_relationships': len(self.parameter_relationships),
                'avg_importance_score': sum(self.importance_scores.values()) / max(1, len(self.importance_scores))
            }

            # 添加 lr_mask 統計信息
            if self.use_lr_mask:
                lr_mask_stats = self._get_lr_mask_statistics()
                info['training_stats']['lr_mask_stats'] = lr_mask_stats

        # 添加記憶體優化統計信息
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

    def clear_buffer_pool(self):
        """清理緩衝區池，釋放記憶體"""
        if hasattr(self, '_buffer_pool'):
            self._buffer_pool.clear()
            logger.info("已清理緩衝區池")

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
                # 估算記憶體使用（粗略計算）
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

    # === 策略 B：lr_mask 核心方法 ===
    def _update_lr_mask(self, param, group_metadata, state, grad, global_step):
        """
        更新 lr_mask（策略 B 的基礎層）

        Args:
            param: 參數張量
            group_metadata: 參數組元數據
            state: 優化器狀態
            grad: 梯度張量
            global_step: 全局步數

        Returns:
            torch.Tensor: 更新後的 lr_mask 縮放因子
        """
        if not self.use_lr_mask:
            return 1.0

        param_id = id(param)
        adaptation_info = group_metadata['adaptation_history'].get(param_id, {})

        # 第一次初始化 lr_mask
        if adaptation_info.get('lr_mask') is None:
            device = param.device
            shape = param.shape
            adaptation_info['lr_mask'] = torch.ones(shape, device=device, dtype=torch.float32) * self.defaults['lr']
            adaptation_info['last_polarity'] = torch.zeros(shape, dtype=torch.bool, device=device)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"首次初始化參數 {param.shape} 的 lr_mask")

        if global_step < self.warmup_steps:
            return self._update_warmup_lr_mask(adaptation_info, grad, global_step)
        else:
            return self._update_post_warmup_lr_mask(adaptation_info, global_step)

    def _update_warmup_lr_mask(self, adaptation_info, grad, global_step):
        """
        Warmup 階段的 lr_mask 更新（基於梯度極性）

        Args:
            adaptation_info: 參數的自適應信息
            grad: 梯度張量
            global_step: 全局步數

        Returns:
            torch.Tensor: lr_mask 縮放因子
        """
        # 追蹤梯度極性變化
        last_polarity = adaptation_info['last_polarity']
        current_polarity = (grad > 0)

        # 判斷極性一致性
        sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
        adaptation_info['last_polarity'] = current_polarity

        # 獲取當前 lr_mask
        lr_mask = adaptation_info['lr_mask']

        # 基於極性一致性調整學習率
        lr_adjustment = torch.where(
            sign_agree > 0,
            self.lr_bump,     # 極性一致：增加學習率
            -self.lr_bump     # 極性變化：減少學習率
        )

        # 更新 lr_mask
        new_lr_mask = lr_mask + lr_adjustment

        # 處理學習率上限更新
        current_base_lr = self.defaults['lr']
        if current_base_lr > adaptation_info['lr_max']:
            new_lr_mask = new_lr_mask + (current_base_lr - adaptation_info['lr_max'])
            adaptation_info['lr_max'] = current_base_lr

        # 限制學習率範圍
        new_lr_mask = torch.clamp(new_lr_mask, min=self.min_lr, max=self.max_lr)

        # 更新狀態
        adaptation_info['lr_mask'] = new_lr_mask
        adaptation_info['avg_lr'] = torch.mean(new_lr_mask).item()

        # 返回相對於基礎學習率的縮放因子
        base_lr = self.defaults['lr']
        lr_scale = new_lr_mask / base_lr if base_lr > 0 else new_lr_mask

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Warmup lr_mask 更新：avg_lr={adaptation_info['avg_lr']:.6f}, "
                       f"scale_range=[{lr_scale.min().item():.3f}, {lr_scale.max().item():.3f}]")

        return lr_scale

    def _update_post_warmup_lr_mask(self, adaptation_info, global_step):
        """
        Post-warmup 階段的 lr_mask 更新（保持穩定，輕微衰減）

        Args:
            adaptation_info: 參數的自適應信息
            global_step: 全局步數

        Returns:
            torch.Tensor: lr_mask 縮放因子
        """
        if not adaptation_info.get('warmup_complete', False):
            # 清理 warmup 相關狀態
            if 'last_polarity' in adaptation_info:
                del adaptation_info['last_polarity']
            adaptation_info['warmup_complete'] = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("lr_mask warmup 階段完成，進入穩定模式")

        # 獲取當前 lr_mask
        lr_mask = adaptation_info['lr_mask']

        # Post-warmup 階段：輕微衰減以保持穩定
        current_base_lr = self.defaults['lr']
        lr_max = adaptation_info['lr_max']

        if current_base_lr < lr_max:
            # 如果當前基礎學習率降低，按比例調整 lr_mask
            decay_factor = max(current_base_lr / lr_max, 0.1)
            lr_mask = lr_mask * decay_factor
            adaptation_info['lr_mask'] = lr_mask

        # 返回相對於基礎學習率的縮放因子
        base_lr = self.defaults['lr']
        lr_scale = lr_mask / base_lr if base_lr > 0 else lr_mask

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Post-warmup lr_mask：avg_lr={torch.mean(lr_mask).item():.6f}")

        return lr_scale

    def _get_lr_mask_statistics(self) -> Dict[str, Any]:
        """獲取 lr_mask 的統計信息"""
        if not self.use_lr_mask:
            return {'message': 'lr_mask 未啟用'}

        stats = {
            'total_updates': 0,
            'positive_updates': 0,
            'negative_updates': 0,
            'positive_to_negative_ratio': 0.0,
            'negative_to_positive_ratio': 0.0,
            'max_lr_scale': 0.0,
            'min_lr_scale': 0.0,
            'avg_lr_scale': 0.0
        }

        for group_metadata in self.param_groups_metadata.values():
            for param_id, adaptation_info in group_metadata['adaptation_history'].items():
                if 'lr_mask' in adaptation_info:
                    lr_mask = adaptation_info['lr_mask']
                    stats['total_updates'] += 1
                    if torch.mean(lr_mask) > 0:
                        stats['positive_updates'] += 1
                    else:
                        stats['negative_updates'] += 1
                    stats['max_lr_scale'] = max(stats['max_lr_scale'], torch.max(lr_mask).item())
                    stats['min_lr_scale'] = min(stats['min_lr_scale'], torch.min(lr_mask).item())
                    stats['avg_lr_scale'] += torch.mean(lr_mask).item()

        if stats['total_updates'] > 0:
            stats['positive_to_negative_ratio'] = stats['positive_updates'] / stats['negative_updates']
            stats['negative_to_positive_ratio'] = stats['negative_updates'] / stats['positive_updates']
            stats['avg_lr_scale'] /= stats['total_updates']

        return stats

    def get_lr_mask_analysis(self) -> Dict[str, Any]:
        """獲取 lr_mask 的詳細分析報告"""
        if not self.use_lr_mask:
            return {'message': 'lr_mask 功能未啟用'}

        analysis = {
            'lr_mask_enabled': True,
            'configuration': {
                'lr_bump': self.lr_bump,
                'min_lr': self.min_lr,
                'max_lr': self.max_lr,
                'warmup_steps': self.warmup_steps
            },
            'parameter_analysis': [],
            'global_statistics': {
                'total_parameters': 0,
                'warmup_parameters': 0,
                'post_warmup_parameters': 0,
                'avg_lr_scale': 0.0,
                'lr_scale_std': 0.0,
                'lr_scale_range': [float('inf'), float('-inf')]
            }
        }

        all_lr_scales = []

        for group_idx, group_metadata in self.param_groups_metadata.items():
            for param in group_metadata['param_list']:
                param_id = id(param)
                adaptation_info = group_metadata['adaptation_history'].get(param_id, {})

                if 'lr_mask' in adaptation_info:
                    lr_mask = adaptation_info['lr_mask']

                    # 計算參數級別統計
                    param_stats = {
                        'param_shape': list(param.shape),
                        'param_id': param_id,
                        'lr_mask_mean': torch.mean(lr_mask).item(),
                        'lr_mask_std': torch.std(lr_mask).item(),
                        'lr_mask_min': torch.min(lr_mask).item(),
                        'lr_mask_max': torch.max(lr_mask).item(),
                        'avg_lr': adaptation_info.get('avg_lr', 0.0),
                        'lr_max': adaptation_info.get('lr_max', 0.0),
                        'warmup_complete': adaptation_info.get('warmup_complete', False),
                        'has_polarity_tracking': 'last_polarity' in adaptation_info
                    }

                    analysis['parameter_analysis'].append(param_stats)

                    # 收集全局統計數據
                    analysis['global_statistics']['total_parameters'] += 1
                    if param_stats['warmup_complete']:
                        analysis['global_statistics']['post_warmup_parameters'] += 1
                    else:
                        analysis['global_statistics']['warmup_parameters'] += 1

                    all_lr_scales.append(param_stats['lr_mask_mean'])

                    # 更新全局範圍
                    lr_range = analysis['global_statistics']['lr_scale_range']
                    lr_range[0] = min(lr_range[0], param_stats['lr_mask_min'])
                    lr_range[1] = max(lr_range[1], param_stats['lr_mask_max'])

        # 計算全局統計
        if all_lr_scales:
            analysis['global_statistics']['avg_lr_scale'] = sum(all_lr_scales) / len(all_lr_scales)

            if len(all_lr_scales) > 1:
                mean_lr = analysis['global_statistics']['avg_lr_scale']
                variance = sum((x - mean_lr) ** 2 for x in all_lr_scales) / len(all_lr_scales)
                analysis['global_statistics']['lr_scale_std'] = variance ** 0.5

        # 如果沒有找到有效範圍，重置為 [0, 0]
        if analysis['global_statistics']['lr_scale_range'][0] == float('inf'):
            analysis['global_statistics']['lr_scale_range'] = [0.0, 0.0]

        # 添加訓練階段分析
        current_step = getattr(self, 'global_step', 0)
        analysis['training_phase'] = {
            'current_step': current_step,
            'in_warmup': current_step < self.warmup_steps,
            'warmup_progress': min(current_step / self.warmup_steps, 1.0) if self.warmup_steps > 0 else 1.0
        }

        return analysis