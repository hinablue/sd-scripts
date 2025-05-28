import torch
from typing import List, Dict, Any, Optional, Tuple, Deque
from torch.nn.functional import normalize
from dataclasses import dataclass
from collections import deque
import math
import random

@dataclass
class ImprovedOptimizerConfig:
    """改進版優化器配置，專門針對 LoRA 訓練優化."""
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
    edge_threshold: float = 0.6
    came: bool = True
    full_finetune: bool = False
    verbose: bool = False
    # 新增：邊緣和背景過擬合控制參數
    edge_suppression: bool = True
    edge_penalty: float = 0.1
    background_regularization: bool = True
    spatial_awareness: bool = True
    frequency_penalty: float = 0.05
    detail_preservation: float = 0.8
    # 新增：LoRA 特定優化參數
    lora_rank_penalty: bool = True
    rank_penalty_strength: float = 0.01
    low_rank_emphasis: float = 1.2

class ImprovedBaseOptimizer(torch.optim.Optimizer):
    """改進版基礎優化器，包含邊緣和背景過擬合控制."""

    def __init__(self, params, config: ImprovedOptimizerConfig):
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
            edge_threshold=config.edge_threshold,
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
        )
        super().__init__(params, defaults)
        self.base_lrs: List[float] = [config.lr for group in self.param_groups]

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

        # 計算拉普拉斯算子（二階導數）
        if len(grad.shape) == 2:
            # 對於 2D 張量，計算 x 和 y 方向的二階導數
            laplacian = torch.zeros_like(grad)
            if grad.shape[0] > 2 and grad.shape[1] > 2:
                # x 方向二階導數
                laplacian[1:-1, :] += grad[2:, :] - 2 * grad[1:-1, :] + grad[:-2, :]
                # y 方向二階導數
                laplacian[:, 1:-1] += grad[:, 2:] - 2 * grad[:, 1:-1] + grad[:, :-2]
        else:
            # 對於高維張量，計算沿最後兩個維度的拉普拉斯算子
            *batch_dims, h, w = grad.shape
            laplacian = torch.zeros_like(grad)
            if h > 2 and w > 2:
                laplacian[..., 1:-1, :] += grad[..., 2:, :] - 2 * grad[..., 1:-1, :] + grad[..., :-2, :]
                laplacian[..., :, 1:-1] += grad[..., :, 2:] - 2 * grad[..., :, 1:-1] + grad[..., :, :-2]

        # 計算邊緣強度
        edge_strength = torch.abs(laplacian)
        # 對超過閾值的邊緣施加懲罰
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

        # 對 2D 張量執行 FFT
        if len(grad.shape) == 2:
            grad_fft = torch.fft.fft2(grad)
            freq_magnitude = torch.abs(grad_fft)

            # 創建高頻懲罰遮罩
            h, w = grad.shape
            center_h, center_w = h // 2, w // 2
            y, x = torch.meshgrid(torch.arange(h, device=grad.device),
                                torch.arange(w, device=grad.device), indexing='ij')
            distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)

            # 高頻區域（距離中心較遠）
            high_freq_mask = (distance > min(h, w) * 0.3).float()
            penalty = freq_magnitude * high_freq_mask

            # 逆 FFT 回到空間域
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

        # 計算 SVD
        U, S, Vh = torch.linalg.svd(param, full_matrices=False)

        # 對較大的奇異值施加懲罰（鼓勵低秩）
        rank_penalty = torch.sum(S[S.argsort(descending=True)[10:]])  # 懲罰前 10 個之外的奇異值

        # 重建懲罰梯度
        penalty_grad = U @ torch.diag(S * rank_strength) @ Vh
        return penalty_grad

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """初始化優化器狀態."""
        device = p.device
        shape = p.shape
        state = self.state[p]

        # 基本狀態初始化
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)

        # 學習率遮罩初始化
        state.setdefault('lr_mask', torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr)
        state.setdefault('avg_lr', float(self.config.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # 動量和變異數初始化
        state.setdefault("exp_avg", torch.zeros_like(p))
        state.setdefault("s", torch.zeros_like(p))

        if group and group.get('came', True):
            state.setdefault("exp_avg_sq", torch.zeros_like(p))

        state.setdefault("exp_avg_res", torch.zeros_like(p))

        # 新增：邊緣和背景過擬合控制狀態
        if group and group.get('edge_suppression', True):
            state.setdefault("edge_history", torch.zeros_like(p))
            state.setdefault("edge_momentum", torch.zeros_like(p))

        if group and group.get('spatial_awareness', True):
            state.setdefault("spatial_variance", torch.ones_like(p))
            state.setdefault("detail_tracker", torch.zeros_like(p))

        # ALLoRA 和 LoRA 特定初始化
        if group and not group.get('full_finetune', True):
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group.get('eta', 2.0)**2))

                # LoRA 低秩結構追蹤
                if group.get('lora_rank_penalty', True):
                    state["rank_tracker"] = torch.zeros(min(p.shape), device=device)


class Automagic_CameAMP_Improved(ImprovedBaseOptimizer):
    """改進版 Automagic_CameAMP 優化器，專門優化 LoRA 訓練並減少邊緣、背景過擬合."""

    def __init__(self, params, **kwargs):
        config = ImprovedOptimizerConfig(**kwargs)
        super().__init__(params, config)

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

            if any(self.state.get(p, {}).get("step", 0) < group.get("warmup_steps", 500) / 2
                   for p in group["params"] if p.grad is not None) and group["weight_decay"] > 0:
                abs_all_group_grads = torch.abs(all_group_grads)
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                state = self.state[p]

                # 狀態初始化
                if len(state) == 0:
                    self._init_state(p, group)

                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                # 清理 warmup 後不需要的狀態
                if state["step"] == group["warmup_steps"]:
                    for key in ['s', 'last_polarity']:
                        if key in state:
                            del state[key]
                    if 'pre' in state and state.get("pre") is not None:
                        del state['pre']

                # === 改進 1：增強的 AGR 自適應梯度正則化 ===
                abs_grad = torch.abs(grad)
                alpha = abs_grad / (sum_abs_all_group_grads + 1e-10)

                # 新增：邊緣感知的梯度正則化
                if group.get('edge_suppression', True):
                    edge_penalty = self._compute_edge_penalty(grad, group.get('edge_threshold', 0.6))
                    edge_factor = 1.0 + group.get('edge_penalty', 0.1) * edge_penalty
                    alpha = alpha * edge_factor

                grad = grad * (1 - alpha)

                # === 改進 2：頻率感知的梯度調整 ===
                if group.get('spatial_awareness', True) and len(grad.shape) >= 2:
                    freq_penalty = self._compute_frequency_penalty(grad)
                    freq_factor = group.get('frequency_penalty', 0.05)
                    grad = grad - freq_factor * freq_penalty

                # === 改進 3：LoRA 低秩正則化 ===
                if group.get('lora_rank_penalty', True) and len(p.shape) == 2:
                    rank_penalty = self._lora_rank_regularization(p, group.get('rank_penalty_strength', 0.01))
                    grad = grad + rank_penalty

                # 原始 CAME 核心處理
                beta1, beta2, beta3 = 0.9, 0.999, 0.9999
                eps1, eps2 = group["eps"][:2]

                # CAME 自適應記憶體高效優化
                if group.get('came', True):
                    update_p = grad.pow(2) + eps1
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update_p, alpha=1 - beta2)
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
                    scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))
                else:
                    scaled_grad = grad

                # === 改進 4：增強的動量處理 ===
                if state["step"] < group["warmup_steps"] / 2:
                    # Torque-Aware Momentum with LoRA adaptation
                    decay_rate = 0.9
                    if 's' in state:
                        s, exp_avg = state['s'], state['exp_avg']
                        corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(scaled_grad, p=2.0, dim=0))
                        s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

                        # LoRA 特定調整：強調低秩方向
                        if group.get('lora_rank_penalty', True) and len(p.shape) == 2:
                            low_rank_factor = group.get('low_rank_emphasis', 1.2)
                            s = s * low_rank_factor

                        d = ((1.0 + s) / 2.0).add_(eps1).mul_(scaled_grad)
                        exp_avg.mul_(beta1).add_(d)
                else:
                    beta1, beta2, beta3 = group["betas"]
                    beta1_t = max(beta1 * group['beta1_decay'] ** state["step"], 0.4)
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1_t).add_(scaled_grad, alpha=1 - beta1_t)

                # CAME 核心：信心引導的記憶體高效優化
                exp_avg_res = state["exp_avg_res"]
                res = (scaled_grad - exp_avg).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt())

                # === 改進 5：增強的 Automagic 學習率遮罩 ===
                if state["step"] < group["warmup_steps"]:
                    if 'last_polarity' in state:
                        last_polarity = state['last_polarity']
                        current_polarity = (grad > 0)
                        sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
                        state['last_polarity'] = current_polarity

                        lr_mask = state['lr_mask']

                        # 新增：空間感知的學習率調整
                        if group.get('spatial_awareness', True):
                            spatial_var = state.get('spatial_variance', torch.ones_like(lr_mask))
                            detail_factor = group.get('detail_preservation', 0.8)
                            spatial_factor = (spatial_var * detail_factor).clamp(0.5, 1.5)
                            lr_bump = self.config.lr_bump * spatial_factor
                        else:
                            lr_bump = self.config.lr_bump

                        new_lr = torch.where(
                            sign_agree > 0,
                            lr_mask + lr_bump,
                            lr_mask - lr_bump
                        )

                        if group["lr"] > state["lr_max"]:
                            new_lr = new_lr + (group["lr"] - state["lr_max"])
                            state["lr_max"] = group["lr"]

                        new_lr = torch.clamp(new_lr, min=self.config.min_lr, max=self.config.max_lr)
                        state['lr_mask'] = new_lr
                        state['avg_lr'] = torch.mean(new_lr).item()
                else:
                    new_lr = state['lr_mask']
                    if group["lr"] > state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    if group["lr"] < state["lr_max"]:
                        new_lr = new_lr * (group["lr"] / state["lr_max"])

                # === 改進 6：智能梯度方向控制 ===
                if state["step"] < group["warmup_steps"] / 2:
                    # Grams with edge awareness
                    update_p.abs_().mul_(grad.sign())

                    # 新增：邊緣抑制調整
                    if group.get('edge_suppression', True):
                        edge_history = state.get('edge_history', torch.zeros_like(update_p))
                        current_edge = self._compute_edge_penalty(update_p)
                        edge_history.mul_(0.9).add_(current_edge, alpha=0.1)
                        edge_suppression_factor = 1.0 - group.get('edge_penalty', 0.1) * edge_history
                        update_p = update_p * edge_suppression_factor.clamp(0.1, 1.0)
                        state['edge_history'] = edge_history
                else:
                    # Cautious optimization with spatial awareness
                    mask = (update_p * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))

                    # 新增：背景正則化
                    if group.get('background_regularization', True):
                        # 檢測背景區域（梯度變化較小的區域）
                        grad_variance = torch.var(grad) if grad.numel() > 1 else torch.tensor(0.0)
                        if grad_variance < 1e-6:  # 可能是背景區域
                            background_factor = 0.5  # 減少背景區域的更新強度
                            mask = mask * background_factor

                    update_p = update_p * mask

                # 應用學習率遮罩
                update_p = update_p * new_lr

                # === 改進 7：增強的選擇性投影衰減 ===
                do_spd = False
                if state["step"] < group.get("warmup_steps", 500):
                    pre = state.get("pre", torch.zeros_like(p))
                    condition = -torch.sum(grad * (p - pre))

                    if condition < 0.0:
                        do_spd = True
                        new_p = p - update_p
                        ratio = self._ratio(new_p, p, pre)

                        # 新增：LoRA 感知的權重衰減
                        if group.get('lora_rank_penalty', True) and len(p.shape) == 2:
                            # 對低秩成分減少衰減，對高秩成分增加衰減
                            U, S, Vh = torch.linalg.svd(new_p - pre, full_matrices=False)
                            rank_weights = torch.exp(-torch.arange(len(S), device=S.device) * 0.1)
                            weighted_decay = group["weight_decay"] * (1.0 + rank_weights.mean())
                        else:
                            weighted_decay = group["weight_decay"]

                        new_p = new_p - weighted_decay * ratio * (new_p - pre)
                        p.copy_(new_p)

                    state["pre"] = p.clone()

                # 最終參數更新
                if not do_spd:
                    p.add_(update_p, alpha=-1)

                # === 改進 8：更新空間感知狀態 ===
                if group.get('spatial_awareness', True):
                    # 更新空間變異數追蹤
                    if len(grad.shape) >= 2:
                        current_variance = torch.var(grad, dim=-1, keepdim=True) if grad.shape[-1] > 1 else torch.ones_like(grad)
                        spatial_var = state.get('spatial_variance', torch.ones_like(current_variance))
                        spatial_var.mul_(0.9).add_(current_variance, alpha=0.1)
                        state['spatial_variance'] = spatial_var

        if self.config.verbose:
            avg_lrs = [torch.mean(state['lr_mask']).item() if 'lr_mask' in state else group["lr"]
                      for group in self.param_groups for state in [self.state.get(p, {}) for p in group["params"] if p.grad is not None]]
            print(f"平均學習率: {avg_lrs}")

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """獲取優化器狀態字典."""
        state = super().state_dict()
        state['magic_version'] = 2  # 標記為改進版本
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """載入優化器狀態字典."""
        if 'magic_version' not in state_dict:
            print('[警告] 您載入了舊版本的狀態字典，某些新功能可能無法正常工作！')
        elif state_dict['magic_version'] != 2:
            print(f'[警告] 狀態字典版本不匹配：期望版本 2，實際版本 {state_dict["magic_version"]}')
        super().load_state_dict(state_dict)