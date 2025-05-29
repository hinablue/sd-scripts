"""
重構版 Automagic_CameAMP_Improved 優化器
專門針對 LoRA 訓練優化，具有邊緣和背景過擬合控制功能

主要改進：
1. 模組化設計 - 將功能分解為獨立的類別和方法
2. 清晰的介面 - 使用策略模式和工廠模式
3. 更好的文件說明 - 詳細的 docstring 和類型提示
4. 錯誤處理 - 增加輸入驗證和異常處理
5. 可測試性 - 分離純函數和狀態管理
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import math
import warnings


class OptimizationPhase(Enum):
    """優化階段枚舉"""
    WARMUP_EARLY = "warmup_early"      # 早期預熱階段
    WARMUP_LATE = "warmup_late"        # 後期預熱階段
    STABLE = "stable"                   # 穩定階段
    MATURE = "mature"                   # 成熟階段


@dataclass
class OptimizerConfig:
    """優化器配置類別 - 使用資料類別提高可讀性"""

    # 基本學習率參數
    lr: float = 1e-6
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    lr_bump: float = 3e-6

    # 數值穩定性參數
    eps: Tuple[float, float, float] = (1e-30, 1e-16, 1e-8)
    clip_threshold: float = 1.0

    # 動量和衰減參數
    betas: Tuple[float, float, float] = (0.8, 0.99, 0.999)
    beta1_decay: float = 0.9995
    weight_decay: float = 5e-4

    # 訓練階段參數
    warmup_steps: int = 500
    eta: float = 2.0

    # 功能開關
    came_enabled: bool = True
    full_finetune: bool = False
    verbose: bool = False

    # 邊緣和背景控制參數
    edge_suppression: bool = True
    edge_threshold: float = 0.6
    edge_penalty: float = 0.1
    background_regularization: bool = True

    # 空間感知參數
    spatial_awareness: bool = True
    frequency_penalty: float = 0.05
    detail_preservation: float = 0.8

    # LoRA 特定參數
    lora_rank_penalty: bool = True
    rank_penalty_strength: float = 0.01
    low_rank_emphasis: float = 1.2

    # TAM (Torque-Aware Momentum) 優化參數
    tam_correlation_method: str = "auto"  # "auto", "cosine_similarity", "manual", "vectorized"
    tam_large_tensor_threshold: int = 1000  # 大張量閾值，決定使用哪種計算方法
    tam_use_inplace_ops: bool = True  # 是否使用原地操作

    def __post_init__(self):
        """配置驗證"""
        self._validate_config()

    def _validate_config(self):
        """驗證配置參數的有效性"""
        if self.lr <= 0:
            raise ValueError("學習率必須大於 0")
        if self.min_lr >= self.max_lr:
            raise ValueError("最小學習率必須小於最大學習率")
        if self.warmup_steps <= 0:
            raise ValueError("預熱步數必須大於 0")
        if not (0 < self.edge_threshold < 1):
            raise ValueError("邊緣閾值必須在 0 和 1 之間")


class TensorUtils:
    """張量工具類別 - 提供靜態方法用於張量操作"""

    @staticmethod
    def compute_rms(tensor: torch.Tensor) -> torch.Tensor:
        """
        計算張量的均方根值

        Args:
            tensor: 輸入張量

        Returns:
            均方根值
        """
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    @staticmethod
    def compute_ratio(new_p: torch.Tensor, p: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """
        計算選擇性投影衰減的比率

        Args:
            new_p: 新參數
            p: 當前參數
            pre: 前一步參數

        Returns:
            衰減比率
        """
        curr_norm = torch.norm(new_p - pre)
        prev_norm = torch.norm(p - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return F.hardtanh(ratio, 0.0, 1.0)

    @staticmethod
    def safe_svd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        安全的 SVD 分解，處理異常情況

        Args:
            tensor: 輸入張量

        Returns:
            U, S, Vh 分解結果
        """
        try:
            return torch.linalg.svd(tensor, full_matrices=False)
        except RuntimeError as e:
            warnings.warn(f"SVD 分解失敗，使用零張量替代: {e}")
            m, n = tensor.shape
            device = tensor.device
            rank = min(m, n)
            U = torch.zeros(m, rank, device=device)
            S = torch.zeros(rank, device=device)
            Vh = torch.zeros(rank, n, device=device)
            return U, S, Vh


class RegularizationStrategy(ABC):
    """正則化策略抽象基類"""

    @abstractmethod
    def apply(self, grad: torch.Tensor, **kwargs) -> torch.Tensor:
        """應用正則化"""
        pass


class EdgeSuppressionRegularizer(RegularizationStrategy):
    """邊緣抑制正則化器"""

    def __init__(self, threshold: float = 0.6, penalty: float = 0.1):
        self.threshold = threshold
        self.penalty = penalty

    def apply(self, grad: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        計算邊緣懲罰項，使用拉普拉斯算子檢測邊緣

        Args:
            grad: 梯度張量

        Returns:
            邊緣懲罰項
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        laplacian = self._compute_laplacian(grad)
        edge_strength = torch.abs(laplacian)
        edge_mask = (edge_strength > self.threshold).float()

        return edge_mask * edge_strength * self.penalty

    def _compute_laplacian(self, grad: torch.Tensor) -> torch.Tensor:
        """計算拉普拉斯算子"""
        laplacian = torch.zeros_like(grad)

        if len(grad.shape) == 2:
            h, w = grad.shape
            if h > 2:
                # x 方向二階導數
                laplacian[1:-1, :] += grad[2:, :] - 2 * grad[1:-1, :] + grad[:-2, :]
            if w > 2:
                # y 方向二階導數
                laplacian[:, 1:-1] += grad[:, 2:] - 2 * grad[:, 1:-1] + grad[:, :-2]
        else:
            # 高維張量處理
            *batch_dims, h, w = grad.shape
            if h > 2:
                laplacian[..., 1:-1, :] += (grad[..., 2:, :] - 2 * grad[..., 1:-1, :] + grad[..., :-2, :])
            if w > 2:
                laplacian[..., :, 1:-1] += (grad[..., :, 2:] - 2 * grad[..., :, 1:-1] + grad[..., :, :-2])

        return laplacian


class FrequencyRegularizer(RegularizationStrategy):
    """頻率正則化器"""

    def __init__(self, penalty: float = 0.05, high_freq_ratio: float = 0.3):
        self.penalty = penalty
        self.high_freq_ratio = high_freq_ratio

    def apply(self, grad: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        計算頻率懲罰項，抑制高頻噪聲

        Args:
            grad: 梯度張量

        Returns:
            頻率懲罰項
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        if len(grad.shape) == 2:
            return self._compute_2d_frequency_penalty(grad)

        return torch.zeros_like(grad)

    def _compute_2d_frequency_penalty(self, grad: torch.Tensor) -> torch.Tensor:
        """計算 2D 頻率懲罰"""
        try:
            # FFT 變換
            grad_fft = torch.fft.fft2(grad)
            freq_magnitude = torch.abs(grad_fft)

            # 創建高頻遮罩
            h, w = grad.shape
            center_h, center_w = h // 2, w // 2
            y, x = torch.meshgrid(
                torch.arange(h, device=grad.device),
                torch.arange(w, device=grad.device),
                indexing='ij'
            )

            distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            high_freq_mask = (distance > min(h, w) * self.high_freq_ratio).float()

            # 應用懲罰
            penalty = freq_magnitude * high_freq_mask * self.penalty

            # 逆變換回空間域
            penalty_spatial = torch.real(torch.fft.ifft2(penalty))
            return penalty_spatial

        except Exception as e:
            warnings.warn(f"頻率懲罰計算失敗: {e}")
            return torch.zeros_like(grad)


class LoRARegularizer(RegularizationStrategy):
    """LoRA 低秩正則化器"""

    def __init__(self, rank_strength: float = 0.01, rank_threshold: int = 10):
        self.rank_strength = rank_strength
        self.rank_threshold = rank_threshold

    def apply(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        LoRA 低秩正則化，鼓勵學習低秩結構

        Args:
            param: 參數張量

        Returns:
            正則化梯度
        """
        if len(param.shape) != 2:
            return torch.zeros_like(param)

        U, S, Vh = TensorUtils.safe_svd(param)

        # 對大奇異值施加懲罰
        if len(S) > self.rank_threshold:
            penalty_indices = S.argsort(descending=True)[self.rank_threshold:]
            rank_penalty = torch.sum(S[penalty_indices])
        else:
            rank_penalty = 0.0

        # 重建懲罰梯度
        if rank_penalty > 0:
            penalty_grad = U @ torch.diag(S * self.rank_strength) @ Vh
            return penalty_grad

        return torch.zeros_like(param)

class MomentumStrategy(ABC):
    """動量策略抽象基類"""

    @abstractmethod
    def update(
        self,
        state: Dict[str, Any],
        scaled_grad: torch.Tensor,
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """更新動量"""
        pass


class TorqueAwareMomentum(MomentumStrategy):
    """扭矩感知動量 - 優化版本"""

    def __init__(self, config: Optional[OptimizerConfig] = None):
        # 預計算常數，避免重複計算
        self.decay_rate = 0.9
        self.beta1 = 0.9
        self.eps_corr = 1e-8  # 用於相關性計算的 eps
        self.eps_norm = 1e-10  # 用於範數計算的 eps

        # 從配置獲取優化參數
        if config:
            self.correlation_method = config.tam_correlation_method
            self.large_tensor_threshold = config.tam_large_tensor_threshold
            self.use_inplace_ops = config.tam_use_inplace_ops
        else:
            self.correlation_method = "auto"
            self.large_tensor_threshold = 1000
            self.use_inplace_ops = True

    def update(
        self,
        state: Dict[str, Any],
        scaled_grad: torch.Tensor,
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """更新扭矩感知動量 - 高效實現"""
        eps1 = group["eps"][0]

        if 's' in state:
            s, exp_avg = state['s'], state['exp_avg']

            # 優化 1: 根據配置選擇最佳的相關性計算方法
            corr = self._compute_correlation_adaptive(exp_avg, scaled_grad, group)

            # 優化 2: 使用原地操作減少記憶體分配
            if self.use_inplace_ops:
                s.mul_(self.decay_rate).add_(corr, alpha=1.0 - self.decay_rate)
            else:
                s = s * self.decay_rate + corr * (1.0 - self.decay_rate)
                state['s'] = s

            # 優化 3: 融合多個操作，減少中間張量
            if self.use_inplace_ops:
                # 使用原地操作版本
                torch.addcmul(
                    scaled_grad * eps1,  # eps1 * scaled_grad
                    s, scaled_grad,      # s * scaled_grad
                    value=0.5,           # 0.5 係數
                    out=s               # 原地操作，重用 s 作為輸出緩衝區
                )
                s.add_(scaled_grad, alpha=0.5)  # 加上 0.5 * scaled_grad
            else:
                # 非原地操作版本（可能更穩定）
                d = ((1.0 + s) / 2.0).add_(eps1).mul_(scaled_grad)
                state['s'] = d.clone()  # 儲存中間結果

            # LoRA 特定調整（如果需要）
            if group.get('lora_rank_penalty', True) and len(scaled_grad.shape) == 2:
                low_rank_factor = group.get('low_rank_emphasis', 1.2)
                if self.use_inplace_ops:
                    s.mul_(low_rank_factor)
                else:
                    s = s * low_rank_factor
                    state['s'] = s

            # 優化 4: 原地更新動量
            if self.use_inplace_ops:
                exp_avg.mul_(self.beta1).add_(s)
            else:
                exp_avg = exp_avg * self.beta1 + s
                state['exp_avg'] = exp_avg

            return exp_avg

        return scaled_grad

    def _compute_correlation_adaptive(
        self,
        exp_avg: torch.Tensor,
        scaled_grad: torch.Tensor,
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """
        自適應選擇最佳的相關性計算方法
        """
        method = self.correlation_method

        # 自動選擇最佳方法
        if method == "auto":
            if exp_avg.numel() > self.large_tensor_threshold:
                if len(exp_avg.shape) >= 2:
                    method = "vectorized"
                else:
                    method = "cosine_similarity"
            else:
                method = "manual"

        # 根據選擇的方法計算相關性
        if method == "cosine_similarity":
            return self._compute_efficient_correlation(exp_avg, scaled_grad)
        elif method == "vectorized":
            return self._compute_vectorized_correlation(exp_avg, scaled_grad)
        elif method == "manual":
            return self._compute_manual_correlation(exp_avg, scaled_grad)
        else:
            # 預設回退到高效方法
            return self._compute_efficient_correlation(exp_avg, scaled_grad)

    def _compute_manual_correlation(
        self,
        exp_avg: torch.Tensor,
        scaled_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        手動計算相關性（適用於小張量的最快方法）
        """
        # 計算點積
        dot_product = torch.sum(exp_avg * scaled_grad)

        # 計算範數
        exp_avg_norm = torch.norm(exp_avg) + self.eps_norm
        scaled_grad_norm = torch.norm(scaled_grad) + self.eps_norm

        # 計算餘弦相似度
        cosine_sim = dot_product / (exp_avg_norm * scaled_grad_norm)

        # 廣播到原始形狀
        return cosine_sim.expand_as(exp_avg)

    def _compute_efficient_correlation(
        self,
        exp_avg: torch.Tensor,
        scaled_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        高效計算相關性係數

        使用餘弦相似度的直接計算方式，避免昂貴的正規化操作
        corr = (exp_avg · scaled_grad) / (||exp_avg|| * ||scaled_grad||)
        """
        # 方法 1: 使用 torch.cosine_similarity (適用於較大張量)
        if exp_avg.numel() > 1000:
            # 展平張量以使用 cosine_similarity
            exp_avg_flat = exp_avg.view(-1)
            scaled_grad_flat = scaled_grad.view(-1)

            # 計算餘弦相似度
            cosine_sim = F.cosine_similarity(
                exp_avg_flat.unsqueeze(0),
                scaled_grad_flat.unsqueeze(0),
                eps=self.eps_corr
            )

            # 廣播到原始形狀
            return cosine_sim.expand_as(exp_avg)

        # 方法 2: 手動計算（適用於較小張量）
        else:
            # 計算點積
            dot_product = torch.sum(exp_avg * scaled_grad)

            # 計算範數（使用快速平方根近似）
            exp_avg_norm = torch.norm(exp_avg) + self.eps_norm
            scaled_grad_norm = torch.norm(scaled_grad) + self.eps_norm

            # 計算餘弦相似度
            cosine_sim = dot_product / (exp_avg_norm * scaled_grad_norm)

            # 廣播到原始形狀
            return cosine_sim.expand_as(exp_avg)

    def _compute_vectorized_correlation(
        self,
        exp_avg: torch.Tensor,
        scaled_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        向量化相關性計算（備選方案）

        當張量具有特定形狀時使用，進一步優化性能
        """
        if len(exp_avg.shape) >= 2:
            # 對於矩陣形狀，按行計算相關性
            # 重塑為 (batch_size, features)
            original_shape = exp_avg.shape
            exp_avg_2d = exp_avg.view(-1, original_shape[-1])
            scaled_grad_2d = scaled_grad.view(-1, original_shape[-1])

            # 按行計算餘弦相似度
            dot_products = torch.sum(exp_avg_2d * scaled_grad_2d, dim=1, keepdim=True)
            exp_avg_norms = torch.norm(exp_avg_2d, dim=1, keepdim=True) + self.eps_norm
            scaled_grad_norms = torch.norm(scaled_grad_2d, dim=1, keepdim=True) + self.eps_norm

            # 計算相關性
            corr_2d = dot_products / (exp_avg_norms * scaled_grad_norms)

            # 廣播到原始形狀
            corr = corr_2d.expand_as(exp_avg_2d).view(original_shape)
            return corr
        else:
            # 退回到標準方法
            return self._compute_efficient_correlation(exp_avg, scaled_grad)


class StandardMomentum(MomentumStrategy):
    """標準動量"""

    def update(
        self,
        state: Dict[str, Any],
        scaled_grad: torch.Tensor,
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """更新標準動量"""
        beta1, beta2, beta3 = group["betas"]
        step = state["step"]

        # 動態調整 beta1
        beta1_t = max(beta1 * group['beta1_decay'] ** step, 0.4)

        exp_avg = state['exp_avg']
        exp_avg.mul_(beta1_t).add_(scaled_grad, alpha=1 - beta1_t)

        return exp_avg


class AdaptiveLearningRateManager:
    """自適應學習率管理器"""

    def __init__(self, config: OptimizerConfig):
        self.config = config

    def update_lr_mask(
        self,
        state: Dict[str, Any],
        grad: torch.Tensor,
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """更新學習率遮罩"""
        if 'last_polarity' not in state:
            return state.get('lr_mask', torch.ones_like(grad) * self.config.lr)

        # 檢查梯度符號一致性
        last_polarity = state['last_polarity']
        current_polarity = (grad > 0)
        sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
        state['last_polarity'] = current_polarity

        lr_mask = state['lr_mask']

        # 空間感知的學習率調整
        lr_bump = self._compute_spatial_lr_bump(state, group)

        # 更新學習率
        new_lr = torch.where(
            sign_agree > 0,
            lr_mask + lr_bump,
            lr_mask - lr_bump
        )

        # 處理全域學習率變化
        if group["lr"] > state.get("lr_max", self.config.lr):
            new_lr = new_lr + (group["lr"] - state.get("lr_max", self.config.lr))
            state["lr_max"] = group["lr"]

        # 限制學習率範圍
        new_lr = torch.clamp(new_lr, min=self.config.min_lr, max=self.config.max_lr)
        state['lr_mask'] = new_lr
        state['avg_lr'] = torch.mean(new_lr).item()

        return new_lr

    def _compute_spatial_lr_bump(
        self,
        state: Dict[str, Any],
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """計算空間感知的學習率增量"""
        if not group.get('spatial_awareness', True):
            return self.config.lr_bump

        spatial_var = state.get('spatial_variance', torch.ones_like(state['lr_mask']))
        detail_factor = group.get('detail_preservation', 0.8)
        spatial_factor = (spatial_var * detail_factor).clamp(0.5, 1.5)

        return self.config.lr_bump * spatial_factor


class OptimizerState:
    """優化器狀態管理器"""

    def __init__(self, config: OptimizerConfig):
        self.config = config

    def initialize_param_state(
        self,
        param: torch.Tensor,
        group: Dict[str, Any]
    ) -> Dict[str, Any]:
        """初始化參數狀態"""
        device = param.device
        shape = param.shape

        state = {
            "lr_max": self.config.lr,
            "step": 0,
            "lr_mask": torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr,
            "avg_lr": float(self.config.lr),
            "last_polarity": torch.zeros(shape, dtype=torch.bool, device=device),
            "exp_avg": torch.zeros_like(param),
            "s": torch.zeros_like(param),
            "exp_avg_res": torch.zeros_like(param),
        }

        # CAME 相關狀態
        if group.get('came_enabled', True):
            state["exp_avg_sq"] = torch.zeros_like(param)

        # 邊緣抑制狀態
        if group.get('edge_suppression', True):
            state["edge_history"] = torch.zeros_like(param)
            state["edge_momentum"] = torch.zeros_like(param)

        # 空間感知狀態
        if group.get('spatial_awareness', True):
            state["spatial_variance"] = torch.ones_like(param)
            state["detail_tracker"] = torch.zeros_like(param)

        # LoRA 特定狀態
        if not group.get('full_finetune', True) and len(param.shape) == 2:
            row_norm = param.norm(dim=1, keepdim=True)
            eta = group.get('eta', 2.0)
            state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (eta**2))

            if group.get('lora_rank_penalty', True):
                state["rank_tracker"] = torch.zeros(min(param.shape), device=device)

        return state

    def cleanup_warmup_state(self, state: Dict[str, Any]):
        """清理預熱階段後不需要的狀態"""
        cleanup_keys = ['s', 'last_polarity']
        for key in cleanup_keys:
            if key in state:
                del state[key]

        if 'pre' in state and state.get("pre") is not None:
            del state['pre']

    def get_optimization_phase(self, step: int, warmup_steps: int) -> OptimizationPhase:
        """獲取當前優化階段"""
        if step < warmup_steps // 2:
            return OptimizationPhase.WARMUP_EARLY
        elif step < warmup_steps:
            return OptimizationPhase.WARMUP_LATE
        elif step < warmup_steps * 2:
            return OptimizationPhase.STABLE
        else:
            return OptimizationPhase.MATURE


class Automagic_CameAMP_Improved(torch.optim.Optimizer):
    """
    重構版 Automagic_CameAMP 優化器

    專門針對 LoRA 訓練優化，具有以下特性：
    - 模組化設計，易於擴展和測試
    - 多種正則化策略
    - 自適應學習率管理
    - 階段性優化策略
    - 完整的錯誤處理
    """

    def __init__(self, params, config: Optional[OptimizerConfig] = None, **kwargs):
        """
        初始化優化器

        Args:
            params: 參數列表
            config: 優化器配置
            **kwargs: 額外配置參數
        """
        if config is None:
            config = OptimizerConfig(**kwargs)

        self.config = config

        # 創建組件
        self.state_manager = OptimizerState(config)
        self.lr_manager = AdaptiveLearningRateManager(config)

        # 創建正則化器
        self.edge_regularizer = EdgeSuppressionRegularizer(
            config.edge_threshold, config.edge_penalty
        )
        self.freq_regularizer = FrequencyRegularizer(config.frequency_penalty)
        self.lora_regularizer = LoRARegularizer(config.rank_penalty_strength)

        # 創建動量策略
        self.torque_momentum = TorqueAwareMomentum(config)
        self.standard_momentum = StandardMomentum()

        # PyTorch 優化器初始化
        defaults = self._create_defaults()
        super().__init__(params, defaults)

    def _create_defaults(self) -> Dict[str, Any]:
        """創建預設參數字典"""
        return dict(
            lr=self.config.lr,
            eps=self.config.eps,
            clip_threshold=self.config.clip_threshold,
            betas=self.config.betas,
            eta=self.config.eta,
            beta1_decay=self.config.beta1_decay,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            came_enabled=self.config.came_enabled,
            full_finetune=self.config.full_finetune,
            edge_suppression=self.config.edge_suppression,
            background_regularization=self.config.background_regularization,
            spatial_awareness=self.config.spatial_awareness,
            lora_rank_penalty=self.config.lora_rank_penalty,
            rank_penalty_strength=self.config.rank_penalty_strength,
            low_rank_emphasis=self.config.low_rank_emphasis,
            tam_correlation_method=self.config.tam_correlation_method,
            tam_large_tensor_threshold=self.config.tam_large_tensor_threshold,
            tam_use_inplace_ops=self.config.tam_use_inplace_ops,
        )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        執行優化步驟

        Args:
            closure: 可選的閉包函數

        Returns:
            損失值（如果提供了閉包）
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            # 計算群組梯度統計
            group_grad_stats = self._compute_group_grad_stats(group)

            for param in group["params"]:
                if param.grad is None or not param.requires_grad:
                    continue

                self._update_single_param(param, group, group_grad_stats)

        if self.config.verbose:
            self._print_lr_stats()

        return loss

    def _compute_group_grad_stats(self, group: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """計算群組梯度統計"""
        grads = []
        for param in group["params"]:
            if param.grad is not None and param.requires_grad:
                grads.append(param.grad.view(-1))

        if not grads:
            return {}

        all_grads = torch.cat(grads)
        return {
            "sum_abs_grads": torch.sum(torch.abs(all_grads)),
            "mean_grads": all_grads.mean(),
            "std_grads": all_grads.std(unbiased=False)
        }

    def _update_single_param(
        self,
        param: torch.Tensor,
        group: Dict[str, Any],
        grad_stats: Dict[str, torch.Tensor]
    ):
        """更新單一參數"""
        grad = param.grad
        state = self.state[param]

        # 初始化狀態
        if len(state) == 0:
            state.update(self.state_manager.initialize_param_state(param, group))

        state["step"] += 1
        step = state["step"]
        warmup_steps = group["warmup_steps"]

        # 獲取優化階段
        phase = self.state_manager.get_optimization_phase(step, warmup_steps)

        # 清理預熱後的狀態
        if step == warmup_steps:
            self.state_manager.cleanup_warmup_state(state)

        # 應用正則化
        processed_grad = self._apply_regularizations(grad, param, group, grad_stats)

        # CAME 核心處理
        scaled_grad = self._apply_came_scaling(processed_grad, state, group)

        # 動量更新
        momentum_grad = self._update_momentum(scaled_grad, state, group, phase)

        # CAME 信心引導優化
        update_tensor = self._apply_came_confidence(momentum_grad, state, group)

        # 學習率調整
        lr_mask = self._update_learning_rate(processed_grad, state, group, phase)

        # 梯度方向控制
        controlled_update = self._apply_gradient_control(update_tensor, processed_grad, state, group, phase)

        # 正交梯度
        grams_tensor = self._apply_orthograd_regularizer(param, controlled_update, phase)

        # 應用學習率遮罩
        final_update = grams_tensor * lr_mask

        # 選擇性投影衰減
        self._apply_selective_projection_decay(param, final_update, state, group, phase)

        # 更新空間感知狀態
        self._update_spatial_awareness(processed_grad, state, group)

    def _apply_regularizations(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        group: Dict[str, Any],
        grad_stats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """應用各種正則化"""
        processed_grad = grad.clone()

        # AGR 自適應梯度正則化
        if "sum_abs_grads" in grad_stats:
            abs_grad = torch.abs(processed_grad)
            alpha = abs_grad / (grad_stats["sum_abs_grads"] + 1e-10)

            # 邊緣感知正則化
            if group.get('edge_suppression', True):
                edge_penalty = self.edge_regularizer.apply(processed_grad)
                edge_factor = 1.0 + edge_penalty
                alpha = alpha * edge_factor

            processed_grad = processed_grad * (1 - alpha)

        # 頻率感知調整
        if group.get('spatial_awareness', True) and len(processed_grad.shape) >= 2:
            freq_penalty = self.freq_regularizer.apply(processed_grad)
            processed_grad = processed_grad - freq_penalty

        # LoRA 低秩正則化
        if group.get('lora_rank_penalty', True) and len(param.shape) == 2:
            rank_penalty = self.lora_regularizer.apply(param)
            processed_grad = processed_grad + rank_penalty

        return processed_grad

    def _apply_came_scaling(
        self,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """應用 CAME 縮放"""
        if not group.get('came_enabled', True):
            return grad

        eps1 = group["eps"][0]
        beta2 = 0.999

        update_p = grad.pow(2) + eps1
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2).add_(update_p, alpha=1 - beta2)

        scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
        rms_ratio = (TensorUtils.compute_rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0)
        scaled_grad.div_(rms_ratio)

        return scaled_grad

    def _update_momentum(
        self,
        scaled_grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        phase: OptimizationPhase
    ) -> torch.Tensor:
        """更新動量"""
        if phase == OptimizationPhase.WARMUP_EARLY:
            return self.torque_momentum.update(state, scaled_grad, group)
        else:
            return self.standard_momentum.update(state, scaled_grad, group)

    def _apply_orthograd_regularizer(self, param: torch.Tensor, grad: torch.Tensor, phase: OptimizationPhase) -> torch.Tensor:
        """應用正交梯度正則化"""
        """
        OrthoGrad 正交梯度正則化器

        實現自 "Grokking at the Edge of Numerical Stability"
        使用與當前權重方向正交的梯度分量來更新權重，
        有助於防止過擬合並改善泛化能力。

        參考：https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev

        計算正交梯度分量

        Args:
            param: 參數張量（當前權重）
            grad: 梯度張量
            phase: 優化階段

        Returns:
            正交化後的梯度
        """

        # 如果參數範數太小，直接返回原梯度
        if phase != OptimizationPhase.WARMUP_EARLY or param.norm(2) <= 1e-30:
            return grad

        # 保存原始形狀
        original_shape = grad.shape

        # 展平為向量進行計算
        w = param.view(-1)
        g = grad.view(-1)

        # 計算梯度範數
        g_norm = g.norm(2)

        # 計算投影：proj = (w·g / w·w) * w
        proj_coeff = torch.dot(w, g) / (torch.dot(w, w) + 1e-30)

        # 計算正交分量：g_orth = g - proj
        g_orth = g - proj_coeff * w

        # 保持梯度範數：縮放正交分量使其具有原始梯度的範數
        g_orth_norm = g_orth.norm(2)
        if g_orth_norm > 1e-30:
            g_orth_scaled = g_orth * (g_norm / (g_orth_norm + 1e-30))
        else:
            # 如果正交分量為零，返回原梯度
            g_orth_scaled = g

        # 恢復原始形狀
        return g_orth_scaled.view(original_shape)

    def _apply_came_confidence(
        self,
        momentum_grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any]
    ) -> torch.Tensor:
        """應用 CAME 信心引導優化"""
        eps2 = group["eps"][1]
        beta3 = 0.9999

        exp_avg_res = state["exp_avg_res"]
        res = (momentum_grad - state["exp_avg"]).pow(2) + eps2
        exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)

        update_tensor = state["exp_avg"].clone().mul_(exp_avg_res.rsqrt())
        return update_tensor

    def _update_learning_rate(
        self,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        phase: OptimizationPhase
    ) -> torch.Tensor:
        """更新學習率遮罩"""
        if phase in [OptimizationPhase.WARMUP_EARLY, OptimizationPhase.WARMUP_LATE]:
            return self.lr_manager.update_lr_mask(state, grad, group)
        else:
            # 穩定階段使用固定學習率調整
            lr_mask = state.get('lr_mask', torch.ones_like(grad) * group["lr"])
            if group["lr"] != state.get("lr_max", group["lr"]):
                lr_mask = lr_mask * (group["lr"] / state.get("lr_max", group["lr"]))
                state["lr_max"] = group["lr"]
            return lr_mask

    def _apply_gradient_control(
        self,
        update_tensor: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        phase: OptimizationPhase
    ) -> torch.Tensor:
        """應用梯度方向控制"""
        if phase == OptimizationPhase.WARMUP_EARLY:
            # 早期階段：使用絕對值和符號
            controlled_update = update_tensor.abs_().mul_(grad.sign())

            # 邊緣抑制
            if group.get('edge_suppression', True):
                edge_history = state.get('edge_history', torch.zeros_like(controlled_update))
                current_edge = self.edge_regularizer.apply(controlled_update)
                edge_history.mul_(0.9).add_(current_edge, alpha=0.1)

                edge_suppression_factor = 1.0 - edge_history
                controlled_update = controlled_update * edge_suppression_factor.clamp(0.1, 1.0)
                state['edge_history'] = edge_history

            return controlled_update
        else:
            # 後期階段：謹慎優化
            mask = (update_tensor * grad > 0).to(grad.dtype)
            mask = mask / mask.mean().clamp_(min=1e-3)

            # 背景正則化
            if group.get('background_regularization', True):
                grad_variance = torch.var(grad) if grad.numel() > 1 else torch.tensor(0.0)
                if grad_variance < 1e-6:
                    mask = mask * 0.5  # 減少背景區域更新

            return update_tensor * mask

    def _apply_selective_projection_decay(
        self,
        param: torch.Tensor,
        update_tensor: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        phase: OptimizationPhase
    ):
        """應用選擇性投影衰減"""
        if phase not in [OptimizationPhase.WARMUP_EARLY, OptimizationPhase.WARMUP_LATE]:
            param.add_(update_tensor, alpha=-1)
            return

        pre = state.get("pre", torch.zeros_like(param))
        grad = param.grad
        condition = -torch.sum(grad * (param - pre))

        if condition < 0.0:
            new_param = param - update_tensor
            ratio = TensorUtils.compute_ratio(new_param, param, pre)

            # LoRA 感知的權重衰減
            weight_decay = group["weight_decay"]
            if group.get('lora_rank_penalty', True) and len(param.shape) == 2:
                U, S, Vh = TensorUtils.safe_svd(new_param - pre)
                if len(S) > 0:
                    rank_weights = torch.exp(-torch.arange(len(S), device=S.device) * 0.1)
                    weight_decay = weight_decay * (1.0 + rank_weights.mean().item())

            new_param = new_param - weight_decay * ratio * (new_param - pre)
            param.copy_(new_param)
        else:
            param.add_(update_tensor, alpha=-1)

        state["pre"] = param.clone()

    def _update_spatial_awareness(
        self,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any]
    ):
        """更新空間感知狀態"""
        if not group.get('spatial_awareness', True) or len(grad.shape) < 2:
            return

        try:
            if grad.shape[-1] > 1:
                current_variance = torch.var(grad, dim=-1, keepdim=True)
            else:
                current_variance = torch.ones_like(grad)

            spatial_var = state.get('spatial_variance', torch.ones_like(current_variance))
            spatial_var.mul_(0.9).add_(current_variance, alpha=0.1)
            state['spatial_variance'] = spatial_var
        except Exception as e:
            warnings.warn(f"空間感知狀態更新失敗: {e}")

    def _print_lr_stats(self):
        """列印學習率統計"""
        avg_lrs = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    state = self.state.get(param, {})
                    if 'lr_mask' in state:
                        avg_lrs.append(torch.mean(state['lr_mask']).item())
                    else:
                        avg_lrs.append(group["lr"])

        if avg_lrs:
            print(f"平均學習率: {avg_lrs}")

    def state_dict(self) -> Dict[str, Any]:
        """獲取優化器狀態字典"""
        state = super().state_dict()
        state['optimizer_version'] = 'refactored_v1'
        state['config'] = self.config.__dict__
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """載入優化器狀態字典"""
        if 'optimizer_version' not in state_dict:
            warnings.warn('[警告] 載入的狀態字典可能來自舊版本，某些功能可能無法正常工作！')

        super().load_state_dict(state_dict)


# 工廠函數和便利介面
def create_lora_optimizer(params, lr: float = 1e-6, **kwargs) -> Automagic_CameAMP_Improved:
    """
    創建 LoRA 優化器的便利函數

    Args:
        params: 參數列表
        lr: 學習率
        **kwargs: 其他配置參數

    Returns:
        配置好的優化器實例
    """
    config = OptimizerConfig(
        lr=lr,
        full_finetune=False,
        lora_rank_penalty=True,
        edge_suppression=True,
        spatial_awareness=True,
        **kwargs
    )
    return Automagic_CameAMP_Improved(params, config)


def create_full_finetune_optimizer(params, lr: float = 1e-6, **kwargs) -> Automagic_CameAMP_Improved:
    """
    創建全量微調優化器的便利函數

    Args:
        params: 參數列表
        lr: 學習率
        **kwargs: 其他配置參數

    Returns:
        配置好的優化器實例
    """
    config = OptimizerConfig(
        lr=lr,
        full_finetune=True,
        lora_rank_penalty=False,
        edge_suppression=True,
        spatial_awareness=True,
        **kwargs
    )
    return Automagic_CameAMP_Improved(params, config)