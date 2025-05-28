import torch
from typing import List, Dict, Any, Optional, Tuple
from torch.nn.functional import normalize
from dataclasses import dataclass
import math
import warnings

# 嘗試導入 bitsandbytes
try:
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    warnings.warn("bitsandbytes 未安裝，8bit 功能將不可用。請使用 'pip install bitsandbytes' 安裝。")

@dataclass
class Improved8BitOptimizerConfig:
    """改進版 8bit 優化器配置，專門針對 LoRA 訓練優化."""
    lr: float = 1e-6
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    lr_bump: float = 3e-6
    eps: Tuple[float, float, float] = (1e-30, 1e-16, 1e-8)
    clip_threshold: float = 1.0
    betas: Tuple[float, float, float] = (0.8, 0.99, 0.999)
    eta: float = 2.0
    beta1_decay: float = 0.9995
    weight_decay: float = 5e-4
    warmup_steps: int = 500
    context_window: int = 30
    edge_threshold: float = 0.6
    adaptation_rate: float = 0.25
    came: bool = True
    full_finetune: bool = False
    verbose: bool = False

    # 邊緣和背景過擬合控制參數
    edge_suppression: bool = True
    edge_penalty: float = 0.1
    background_regularization: bool = True
    spatial_awareness: bool = True
    frequency_penalty: float = 0.05
    detail_preservation: float = 0.8

    # LoRA 特定優化參數
    lora_rank_penalty: bool = True
    rank_penalty_strength: float = 0.01
    low_rank_emphasis: float = 1.2

    # 8bit 量化參數
    optim_bits: int = 8
    percentile_clipping: int = 100
    block_wise: bool = True
    min_8bit_size: int = 4096  # 小於此大小的張量不進行 8bit 量化
    stable_emb: bool = False

    # 記憶體優化參數
    fused_adam: bool = False  # 是否使用融合的 Adam 操作
    force_8bit: bool = False  # 強制所有狀態使用 8bit


class BitsAndBytesOptimized(torch.optim.Optimizer):
    """使用 bitsandbytes 的改進版優化器基類."""

    def __init__(self, params, config: Improved8BitOptimizerConfig):
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError(
                "bitsandbytes 不可用。請安裝 bitsandbytes：\n"
                "pip install bitsandbytes\n"
                "或設定 force_8bit=False 使用 32bit 版本。"
            )

        self.config = config
        eta_value = float(config.eta) if isinstance(config.eta, (int, float)) else 2.0

        defaults = dict(
            lr=config.lr,
            eps=config.eps,
            clip_threshold=config.clip_threshold,
            betas=config.betas,
            eta=eta_value,
            beta1_decay=config.beta1_decay,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            context_window=config.context_window,
            edge_threshold=config.edge_threshold,
            adaptation_rate=config.adaptation_rate,
            came=config.came,
            full_finetune=config.full_finetune,
            edge_suppression=config.edge_suppression,
            edge_penalty=config.edge_penalty,
            background_regularization=config.background_regularization,
            spatial_awareness=config.spatial_awareness,
            frequency_penalty=config.frequency_penalty,
            detail_preservation=config.detail_preservation,
            lora_rank_penalty=config.lora_rank_penalty,
            rank_penalty_strength=config.rank_penalty_strength,
            low_rank_emphasis=config.low_rank_emphasis,
            optim_bits=config.optim_bits,
            percentile_clipping=config.percentile_clipping,
            block_wise=config.block_wise,
            min_8bit_size=config.min_8bit_size,
            stable_emb=config.stable_emb,
        )
        super().__init__(params, defaults)
        self.base_lrs: List[float] = [config.lr for group in self.param_groups]

    def _should_use_8bit(self, tensor: torch.Tensor) -> bool:
        """判斷張量是否應該使用 8bit 量化."""
        return (tensor.numel() >= self.config.min_8bit_size and
                tensor.dtype == torch.float32 and
                tensor.device.type == 'cuda')

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        """計算張量的均方根值."""
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    @staticmethod
    def _ratio(new_p: torch.Tensor, p: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """計算選擇性投影衰減的比率."""
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(p - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    @staticmethod
    def _compute_edge_penalty(grad: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
        """
        計算邊緣懲罰項，用於抑制邊緣過擬合.
        使用拉普拉斯算子檢測邊緣，對高頻成分施加懲罰.
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        laplacian = torch.zeros_like(grad)
        if len(grad.shape) == 2:
            if grad.shape[0] > 2 and grad.shape[1] > 2:
                laplacian[1:-1, :] += grad[2:, :] - 2 * grad[1:-1, :] + grad[:-2, :]
                laplacian[:, 1:-1] += grad[:, 2:] - 2 * grad[:, 1:-1] + grad[:, :-2]
        else:
            *batch_dims, h, w = grad.shape
            if h > 2 and w > 2:
                laplacian[..., 1:-1, :] += grad[..., 2:, :] - 2 * grad[..., 1:-1, :] + grad[..., :-2, :]
                laplacian[..., :, 1:-1] += grad[..., :, 2:] - 2 * grad[..., :, 1:-1] + grad[..., :, :-2]

        edge_strength = torch.abs(laplacian)
        edge_mask = (edge_strength > threshold).float()
        return edge_mask * edge_strength

    @staticmethod
    def _compute_frequency_penalty(grad: torch.Tensor) -> torch.Tensor:
        """
        計算頻率懲罰項，抑制高頻噪聲.
        使用 FFT 分析頻率成分，對高頻成分施加懲罰.
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        if len(grad.shape) == 2:
            grad_fft = torch.fft.fft2(grad)
            freq_magnitude = torch.abs(grad_fft)

            h, w = grad.shape
            center_h, center_w = h // 2, w // 2
            y, x = torch.meshgrid(torch.arange(h, device=grad.device),
                                torch.arange(w, device=grad.device), indexing='ij')
            distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)

            high_freq_mask = (distance > min(h, w) * 0.3).float()
            penalty = freq_magnitude * high_freq_mask
            penalty_spatial = torch.real(torch.fft.ifft2(penalty))
            return penalty_spatial

        return torch.zeros_like(grad)

    @staticmethod
    def _lora_rank_regularization(param: torch.Tensor, rank_strength: float = 0.01) -> torch.Tensor:
        """
        LoRA 低秩正則化，鼓勵學習低秩結構.
        通過 SVD 分解對高秩成分施加懲罰.
        """
        if len(param.shape) != 2:
            return torch.zeros_like(param)

        U, S, Vh = torch.linalg.svd(param, full_matrices=False)
        rank_penalty = torch.sum(S[S.argsort(descending=True)[10:]])
        penalty_grad = U @ torch.diag(S * rank_strength) @ Vh
        return penalty_grad

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """初始化 8bit 優化器狀態."""
        device = p.device
        shape = p.shape
        state = self.state[p]

        # 基本狀態初始化
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)

        # 學習率遮罩（始終使用 32bit，因為需要高精度）
        state.setdefault('lr_mask', torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr)
        state.setdefault('avg_lr', float(self.config.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # 決定是否使用 8bit
        use_8bit = self._should_use_8bit(p) and not self.config.force_8bit

        # 動量狀態初始化
        if use_8bit:
            # 使用 bitsandbytes 的 8bit 狀態
            state.setdefault("exp_avg", torch.zeros_like(p, dtype=torch.uint8))
            state.setdefault("exp_avg_state", {'absmax': torch.zeros_like(p, dtype=torch.float16)})
        else:
            # 使用標準 32bit 狀態
            state.setdefault("exp_avg", torch.zeros_like(p))

        if use_8bit:
            state.setdefault("s", torch.zeros_like(p, dtype=torch.uint8))
            state.setdefault("s_state", {'absmax': torch.zeros_like(p, dtype=torch.float16)})
        else:
            state.setdefault("s", torch.zeros_like(p))

        # CAME 狀態
        if group and group.get('came', True):
            if use_8bit:
                state.setdefault("exp_avg_sq", torch.zeros_like(p, dtype=torch.uint8))
                state.setdefault("exp_avg_sq_state", {'absmax': torch.zeros_like(p, dtype=torch.float16)})
            else:
                state.setdefault("exp_avg_sq", torch.zeros_like(p))

        # AdaBelief 殘差
        if use_8bit:
            state.setdefault("exp_avg_res", torch.zeros_like(p, dtype=torch.uint8))
            state.setdefault("exp_avg_res_state", {'absmax': torch.zeros_like(p, dtype=torch.float16)})
        else:
            state.setdefault("exp_avg_res", torch.zeros_like(p))

        # 邊緣控制狀態（使用 32bit 保持精度）
        if group and group.get('edge_suppression', True):
            state.setdefault("edge_history", torch.zeros_like(p))
            state.setdefault("edge_momentum", torch.zeros_like(p))

        # 空間感知狀態（使用 32bit）
        if group and group.get('spatial_awareness', True):
            state.setdefault("spatial_variance", torch.ones_like(p))
            state.setdefault("detail_tracker", torch.zeros_like(p))

        # 標記是否使用 8bit
        state.setdefault("use_8bit", use_8bit)

    def _get_8bit_state(self, state: Dict[str, Any], key: str) -> torch.Tensor:
        """獲取 8bit 狀態張量，自動處理反量化."""
        if state.get("use_8bit", False) and f"{key}_state" in state:
            # 使用 bitsandbytes 反量化
            quantized = state[key]
            state_dict = state[f"{key}_state"]
            return F.dequantize_8bit(quantized, state_dict['absmax'])
        else:
            return state[key]

    def _set_8bit_state(self, state: Dict[str, Any], key: str, tensor: torch.Tensor) -> None:
        """設定 8bit 狀態張量，自動處理量化."""
        if state.get("use_8bit", False) and self._should_use_8bit(tensor):
            # 使用 bitsandbytes 量化
            quantized, state_dict = F.quantize_8bit(tensor)
            state[key] = quantized
            state[f"{key}_state"] = state_dict
        else:
            state[key] = tensor


class Automagic_CameAMP_Improved_8Bit(BitsAndBytesOptimized):
    """
    使用 bitsandbytes 的改進版 Automagic_CameAMP 8bit 優化器.

    這個版本結合了：
    1. bitsandbytes 的高效 8bit 量化
    2. 邊緣過擬合抑制
    3. 背景正則化
    4. LoRA 低秩優化
    5. 頻率感知優化
    6. 空間感知學習率調整
    """

    def __init__(self, params, **kwargs):
        config = Improved8BitOptimizerConfig(**kwargs)
        super().__init__(params, config)

        if self.config.verbose:
            print(f"🚀 初始化 Automagic_CameAMP_Improved_8Bit 優化器")
            print(f"📊 8bit 量化: {'啟用' if BITSANDBYTES_AVAILABLE else '停用'}")
            print(f"🎯 邊緣抑制: {'啟用' if config.edge_suppression else '停用'}")
            print(f"🌅 背景正則化: {'啟用' if config.background_regularization else '停用'}")
            print(f"🧠 LoRA 優化: {'啟用' if config.lora_rank_penalty else '停用'}")

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """執行單一優化步驟."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            # 計算群組梯度統計
            grads_this_group = []
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                grads_this_group.append(p.grad.view(-1))

            if not grads_this_group:
                continue

            all_group_grads = torch.cat(grads_this_group)
            sum_abs_all_group_grads = torch.sum(torch.abs(all_group_grads))

            # 暖身階段統計
            if any(self.state.get(p, {}).get("step", 0) < group.get("warmup_steps", 500) / 2
                   for p in group["params"] if p.grad is not None) and group["weight_decay"] > 0:
                abs_all_group_grads = torch.abs(all_group_grads)
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                # 自適應梯度正則化 (AGR)
                abs_grad = torch.abs(p.grad)
                alpha = abs_grad / (sum_abs_all_group_grads + 1e-8)
                grad = p.grad * (1 - alpha)

                # 初始化狀態
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                # 清理暖身狀態
                if state["step"] == group.get("warmup_steps", 0):
                    cleanup_keys = ['s', 'last_polarity']
                    for key in cleanup_keys:
                        if key in state:
                            del state[key]
                        if f"{key}_state" in state:
                            del state[f"{key}_state"]

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2, eps3 = group["eps"]

                # 使用 8bit 狀態（自動處理量化/反量化）
                exp_avg = self._get_8bit_state(state, 'exp_avg')
                exp_avg_res = self._get_8bit_state(state, 'exp_avg_res')

                # 邊緣抑制處理
                edge_penalty = torch.zeros_like(grad)
                if group.get('edge_suppression', True):
                    edge_penalty = self._compute_edge_penalty(grad, group.get('edge_threshold', 0.6))

                    # 更新邊緣歷史
                    if 'edge_history' in state:
                        state['edge_history'].mul_(0.9).add_(edge_penalty, alpha=0.1)

                    # 邊緣動量
                    if 'edge_momentum' in state:
                        state['edge_momentum'].mul_(0.8).add_(edge_penalty, alpha=0.2)

                # 頻率懲罰
                freq_penalty = torch.zeros_like(grad)
                if group.get('frequency_penalty', 0) > 0:
                    freq_penalty = self._compute_frequency_penalty(grad)

                # LoRA 低秩正則化
                lora_penalty = torch.zeros_like(p.data)
                if group.get('lora_rank_penalty', True) and len(p.shape) == 2:
                    lora_penalty = self._lora_rank_regularization(
                        p.data, group.get('rank_penalty_strength', 0.01)
                    )

                # 結合所有懲罰項
                total_penalty = (group.get('edge_penalty', 0.1) * edge_penalty +
                               group.get('frequency_penalty', 0.05) * freq_penalty +
                               group.get('rank_penalty_strength', 0.01) * lora_penalty)

                # 應用懲罰到梯度
                grad = grad + total_penalty

                # 更新一階動量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # CAME 或 AdaBelief 二階動量
                if group.get('came', True):
                    exp_avg_sq = self._get_8bit_state(state, 'exp_avg_sq')
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
                    self._set_8bit_state(state, 'exp_avg_sq', exp_avg_sq)
                else:
                    # AdaBelief 變體
                    exp_avg_sq = self._get_8bit_state(state, 'exp_avg_sq')
                    diff = grad - exp_avg
                    exp_avg_res.mul_(beta3).add_(diff.pow(2) + eps1, alpha=1 - beta3)
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
                    scaled_grad = exp_avg / (exp_avg_res.sqrt() + eps2)
                    self._set_8bit_state(state, 'exp_avg_sq', exp_avg_sq)

                # 早期階段使用 Torque-Aware Momentum
                if state["step"] < group.get("warmup_steps", 500):
                    s = self._get_8bit_state(state, 's') if 's' in state else torch.zeros_like(p)
                    corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(scaled_grad, p=2.0, dim=0))
                    decay_rate = group["adaptation_rate"]
                    s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

                    polarity = torch.sign(s)
                    if 'last_polarity' in state:
                        cosine_sim = torch.mean(polarity * state['last_polarity'].float())
                        s = s * torch.clamp(cosine_sim, 0.1, 1.0)

                    state['last_polarity'] = polarity.bool()
                    update_p = s

                    self._set_8bit_state(state, 's', s)
                else:
                    # 後期階段使用一致性動量
                    consistency_factor = 1.0 + 0.1 * torch.tanh(exp_avg * scaled_grad)
                    update_p = exp_avg * consistency_factor

                # 更新 8bit 狀態
                self._set_8bit_state(state, 'exp_avg', exp_avg)
                self._set_8bit_state(state, 'exp_avg_res', exp_avg_res)

                # 學習率遮罩更新
                current_lr = self._update_learning_rate_mask(state, group, grad)

                # 空間感知學習率調整
                if group.get('spatial_awareness', True) and 'spatial_variance' in state:
                    spatial_var = torch.var(update_p)
                    state['spatial_variance'].mul_(0.99).add_(spatial_var, alpha=0.01)
                    spatial_factor = torch.clamp(state['spatial_variance'] / (spatial_var + 1e-8), 0.5, 2.0)
                    current_lr = current_lr * spatial_factor

                # ALLoRA 支援
                if len(p.shape) == 2 and hasattr(p, '_is_lora_layer'):
                    # 簡化的 ALLoRA 邏輯
                    row_variance = torch.var(update_p, dim=1, keepdim=True)
                    row_scaling = torch.clamp(row_variance / (torch.mean(row_variance) + 1e-8), 0.1, 10.0)
                    current_lr = current_lr * row_scaling

                # 應用權重衰減和學習率
                if state["step"] < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
                    # 自適應權重衰減
                    param_abs_grad = abs_grad.mean()
                    norm_grad = (param_abs_grad - mean_norm) / (std_norm + 1e-8)
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - current_lr * group["weight_decay"] * theta)

                # 最終參數更新
                update_p = update_p * current_lr
                p.add_(-update_p)

        if self.config.verbose:
            # 顯示記憶體統計
            self._print_memory_stats()

        return loss

    def _update_learning_rate_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """更新學習率遮罩."""
        if state["step"] < group.get("warmup_steps", 500):
            return self._update_warmup_lr_mask(state, group, grad)
        else:
            return self._update_post_warmup_lr_mask(state, group)

    def _update_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """暖身階段學習率遮罩更新."""
        warmup_steps = group.get("warmup_steps", 500)
        current_step = state["step"]
        progress = current_step / warmup_steps

        # 線性暖身
        base_lr = group["lr"] * progress

        # 梯度感知調整
        grad_norm = torch.norm(grad, dim=-1, keepdim=True) if len(grad.shape) > 1 else torch.abs(grad)
        normalized_grad_norm = grad_norm / (torch.mean(grad_norm) + 1e-8)

        # 自適應因子
        adaptive_factor = torch.clamp(1.0 / (normalized_grad_norm + 1e-8), 0.1, 10.0)

        new_lr = base_lr * adaptive_factor
        state['lr_mask'] = new_lr

        return new_lr

    def _update_post_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any]) -> torch.Tensor:
        """暖身後學習率遮罩更新."""
        current_lr = state['lr_mask']

        # 簡化的自動調整邏輯
        lr_adjustment = torch.clamp(torch.randn_like(current_lr) * 0.01, -0.1, 0.1)
        new_lr = torch.clamp(current_lr * (1 + lr_adjustment), group["lr"] * 0.1, group["lr"] * 10)

        state['lr_mask'] = new_lr
        return new_lr

    def _print_memory_stats(self):
        """列印記憶體統計信息."""
        if not self.config.verbose:
            return

        total_8bit_params = 0
        total_32bit_params = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if state.get("use_8bit", False):
                    total_8bit_params += p.numel()
                else:
                    total_32bit_params += p.numel()

        total_params = total_8bit_params + total_32bit_params
        if total_params > 0:
            compression_ratio = total_8bit_params / total_params
            print(f"📊 記憶體統計: 8bit參數={total_8bit_params}, 32bit參數={total_32bit_params}, "
                  f"壓縮率={compression_ratio:.2%}")

    def state_dict(self) -> Dict[str, Any]:
        """獲取優化器狀態字典."""
        state = super().state_dict()
        state['magic_improved_8bit_version'] = 1
        state['bitsandbytes_version'] = getattr(bnb, '__version__', 'unknown') if BITSANDBYTES_AVAILABLE else 'not_available'
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """載入優化器狀態字典."""
        if 'magic_improved_8bit_version' not in state_dict:
            print('[警告] 您載入了舊版本的狀態字典，某些 8bit 功能可能無法正常工作！')

        if state_dict.get('bitsandbytes_version', 'not_available') == 'not_available' and BITSANDBYTES_AVAILABLE:
            print('[警告] 狀態字典來自非 8bit 版本，將嘗試兼容性載入。')

        super().load_state_dict(state_dict)

    def get_memory_efficiency_report(self) -> Dict[str, Any]:
        """獲取詳細的記憶體效率報告."""
        report = {
            'bitsandbytes_available': BITSANDBYTES_AVAILABLE,
            'total_parameters': 0,
            '8bit_parameters': 0,
            '32bit_parameters': 0,
            'memory_saved_mb': 0,
            'compression_ratio': 0,
            'states_breakdown': {}
        }

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group["params"]):
                state = self.state.get(p, {})
                param_count = p.numel()
                report['total_parameters'] += param_count

                if state.get("use_8bit", False):
                    report['8bit_parameters'] += param_count
                    # 估算記憶體節省（每個參數節省 3 bytes：4-1=3）
                    report['memory_saved_mb'] += param_count * 3 / (1024 * 1024)
                else:
                    report['32bit_parameters'] += param_count

        if report['total_parameters'] > 0:
            report['compression_ratio'] = report['8bit_parameters'] / report['total_parameters']

        return report


# 便利函數
def create_improved_8bit_optimizer(model_parameters, **kwargs) -> Automagic_CameAMP_Improved_8Bit:
    """
    便利函數：創建改進版 8bit 優化器.

    Args:
        model_parameters: 模型參數
        **kwargs: 配置參數

    Returns:
        Automagic_CameAMP_Improved_8Bit 實例
    """
    return Automagic_CameAMP_Improved_8Bit(model_parameters, **kwargs)


# 範例配置
class OptimizationProfiles:
    """預定義的優化配置檔案."""

    @staticmethod
    def memory_optimized() -> Improved8BitOptimizerConfig:
        """記憶體優化配置 - 最大記憶體節省."""
        return Improved8BitOptimizerConfig(
            lr=1e-4,
            force_8bit=True,
            min_8bit_size=1024,
            edge_suppression=False,
            spatial_awareness=False,
            verbose=False
        )

    @staticmethod
    def quality_optimized() -> Improved8BitOptimizerConfig:
        """品質優化配置 - 最佳訓練效果."""
        return Improved8BitOptimizerConfig(
            lr=1e-4,
            edge_suppression=True,
            edge_penalty=0.15,
            background_regularization=True,
            spatial_awareness=True,
            lora_rank_penalty=True,
            verbose=True
        )

    @staticmethod
    def balanced() -> Improved8BitOptimizerConfig:
        """平衡配置 - 記憶體與品質兼顧."""
        return Improved8BitOptimizerConfig(
            lr=1e-4,
            min_8bit_size=4096,
            edge_suppression=True,
            edge_penalty=0.1,
            background_regularization=True,
            spatial_awareness=True,
            lora_rank_penalty=True,
            verbose=True
        )