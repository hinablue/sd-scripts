import torch
from typing import List, Dict, Any, Optional, Tuple
import bitsandbytes.functional as F
from torch.nn.functional import normalize
from dataclasses import dataclass
import math

@dataclass
class OptimizerConfig:
    """Configuration for the Automagic_CameAMP optimizer."""
    lr: float = 1e-6
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    lr_bump: float = 3e-6
    eps: Tuple[float, float, float] = (1e-30, 1e-16, 1e-8)
    clip_threshold: float = 1.0
    betas: Tuple[float, float, float] = (0.8, 0.99, 0.999)
    eta: float = 2.0
    beta1_decay: float = 0.9995
    weight_decay: float = 1.0
    warmup_steps: int = 500
    cautious: bool = True
    came: bool = True
    full_finetune: bool = False
    verbose: bool = False

class BaseOptimizer(torch.optim.Optimizer):
    """Base class for Automagic optimizers with common functionality."""

    def __init__(self, params, config: OptimizerConfig):
        self.config = config
        # Handle eta value: if not float use 2.0
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
            cautious=config.cautious,
            came=config.came,
            full_finetune=config.full_finetune,
        )
        super().__init__(params, defaults)
        self.base_lrs: List[float] = [config.lr for group in self.param_groups]

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        """Calculate root mean square of tensor."""
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        """Approximate square gradient for factored matrices."""
        r_factor = (exp_avg_sq_row / (exp_avg_sq_row.mean(dim=-1, keepdim=True) + 1e-12)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _ratio(new_p: torch.Tensor, p: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Calculate the ratio for selective projection decay."""
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(p - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    # Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
    @staticmethod
    def orthograd_(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if p.norm(2) <= 1e-30:
            return grad

        G_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)
        g_norm = g.norm(2)

        proj = torch.dot(w, g) / torch.dot(w, w).add(1e-30)
        g_orth = g.sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g_norm / g_orth.norm(2).add(1e-30))

        return g_orth_scaled.view(G_shape)

    @staticmethod
    def _update_torque_aware_momentum(state: Dict[str, Any], scaled_grad: torch.Tensor, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum using Torque-Aware Momentum during early training.

        Implementation from:
        https://arxiv.org/abs/2412.18790
        https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/tam.py

        Args:
            state: Optimizer state for the parameter
            scaled_grad: Scaled gradient tensor
            eps1: Epsilon parameter for numerical stability

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated momentum average
        """
        # Set fixed beta values for early training
        beta1, beta2, beta3 = 0.9, 0.999, 0.9999
        decay_rate = 0.9

        # Get state tensors
        s, exp_avg = state['s'], state['exp_avg']

        # Calculate correlation between normalized momentum and gradient
        corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(scaled_grad, p=2.0, dim=0))

        # Update correlation state
        s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

        # Calculate torque-aware update
        d = ((1.0 + s) / 2.0).add_(eps1).mul_(scaled_grad)

        # Update momentum
        exp_avg.mul_(beta1).add_(d)

        # Calculate momentum average
        exp_avg_bar = exp_avg

        return exp_avg_bar, exp_avg

    @staticmethod
    def _update_consistency_momentum(state: Dict[str, Any], group: Dict[str, Any], scaled_grad: torch.Tensor, beta1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum using consistency-based approach after early training.

        Implementation from:
        Towards Faster Training of Diffusion Models: An Inspiration of A Consistency Phenomenon
        https://arxiv.org/abs/2404.07946

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings
            scaled_grad: Scaled gradient tensor
            beta1: Beta1 parameter for momentum

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated momentum average
        """
        # Calculate time-dependent beta1
        beta1_t = max(beta1 * group['beta1_decay'] ** state["step"], 0.4)

        # Get momentum state
        exp_avg = state['exp_avg']

        # Update momentum with time-dependent beta1
        exp_avg.mul_(beta1_t).add_(scaled_grad, alpha=1 - beta1_t)

        # Calculate momentum average
        exp_avg_bar = exp_avg * beta1 + scaled_grad * (1 - beta1)

        return exp_avg_bar, exp_avg

    @staticmethod
    def _update_post_warmup_lr_mask(state: Dict[str, Any], group: Dict[str, Any]) -> torch.Tensor:
        """Update learning rate mask after warmup phase.

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings

        Returns:
            torch.Tensor: Updated learning rate mask
        """
        new_lr = state['lr_mask']

        # Update maximum learning rate if needed
        if group["lr"] > state["lr_max"]:
            state["lr_max"] = group["lr"]

        # Scale learning rate if current lr is less than maximum
        if group["lr"] < state["lr_max"]:
            new_lr = new_lr * max((group["lr"] / state["lr_max"]), 0.1)

        return new_lr

    def _get_group_lr(self, group: Dict[str, Any]) -> float:
        """Get the average learning rate for a parameter group."""
        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])
        return float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.config.lr

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """Initialize optimizer state for a parameter."""
        device = p.device
        shape = p.shape
        state = self.state[p]

        # Basic state initialization
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)

        # Learning rate mask initialization
        state.setdefault('lr_mask', torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr)
        state.setdefault('avg_lr', float(self.config.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # Momentum and variance initialization
        state.setdefault("exp_avg", torch.zeros_like(p))
        state.setdefault("s", torch.zeros_like(p))

        if group['came']:
            state.setdefault("exp_avg_sq", torch.zeros_like(p))

        state.setdefault("exp_avg_res", torch.zeros_like(p))

        # Full finetune initialization
        if group is not None and group['full_finetune'] is False:
            """
            ==== ALLoRA ====
            ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            https://arxiv.org/abs/2410.09692
            """
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))

    def _update_learning_rate_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """Update the learning rate mask based on gradient polarity and current state.

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings
            grad: Gradient tensor

        Returns:
            torch.Tensor: Updated learning rate mask
        """
        if state["step"] < group["warmup_steps"]:
            return self._update_warmup_lr_mask(state, group, grad)
        else:
            return self._update_post_warmup_lr_mask(state, group)

    def _update_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """Update learning rate mask during warmup phase.

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings
            grad: Gradient tensor

        Returns:
            torch.Tensor: Updated learning rate mask
        """
        # Update polarity tracking
        last_polarity = state['last_polarity']
        current_polarity = (grad > 0)
        sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
        state['last_polarity'] = current_polarity

        # Calculate new learning rate
        lr_mask = state['lr_mask']
        new_lr = torch.where(
            sign_agree > 0,
            lr_mask + self.config.lr_bump,
            lr_mask - self.config.lr_bump
        )

        # Handle learning rate maximum
        if group["lr"] > state["lr_max"]:
            new_lr = new_lr + (group["lr"] - state["lr_max"])
            state["lr_max"] = group["lr"]

        # Clamp learning rate to valid range
        new_lr = torch.clamp(new_lr, min=self.config.min_lr, max=self.config.max_lr)

        # Update state
        state['lr_mask'] = new_lr
        state['avg_lr'] = torch.mean(new_lr).item()

        return new_lr

    def _update_momentum(self, state: Dict[str, Any], group: Dict[str, Any], scaled_grad: torch.Tensor, beta1: float, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum based on current training phase.

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings
            scaled_grad: Scaled gradient tensor
            beta1: Beta1 parameter for momentum
            eps1: Epsilon parameter for numerical stability

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated momentum average
        """
        if state["step"] < group["warmup_steps"] / 2:
            exp_avg_bar, exp_avg = self._update_torque_aware_momentum(state, scaled_grad, eps1)
        else:
            exp_avg_bar, exp_avg = self._update_consistency_momentum(state, group, scaled_grad, beta1)

        return exp_avg_bar, exp_avg

class Automagic_CameAMP(BaseOptimizer):
    """Automagic_CameAMP optimizer implementation."""

    def __init__(self, params, **kwargs):
        config = OptimizerConfig(**kwargs)
        super().__init__(params, config)

        self._step = 0

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))
            if len(grads_this_group) == 0:
                continue

            all_group_grads = torch.cat(grads_this_group)
            abs_all_group_grads = torch.abs(all_group_grads)
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads)

            if self._step < self.config.warmup_steps / 2 and self.config.weight_decay > 0:
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                """
                === grad 初始化 ===
                ==== AGR自適應梯度正則 ====
                Adaptive Gradient Regularization: A Faster and Generalizable Optimization Technique for Deep Neural Networks
                https://arxiv.org/pdf/2407.16944
                """
                abs_grad = torch.abs(p.grad)
                alpha = abs_grad / sum_abs_all_group_grads
                grad = p.grad * (1 - alpha)

                # === state 初始化 ===
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                self._step = state["step"]
                if state["step"] == group.get("warmup_steps", 0):
                    if 's' in state:
                        del state['s']
                    if 'last_polarity' in state:
                        del state['last_polarity']

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2, eps3 = group["eps"]
                exp_avg , exp_avg_res = state['exp_avg'], state["exp_avg_res"]

                if group["came"]:
                    """
                    CAME: Confidence-guided Adaptive Memory Efficient Optimization
                    https://arxiv.org/pdf/2411.02853
                    https://github.com/yangluo7/CAME
                    """
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())

                    scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))
                else:
                    # Adabelief
                    clip = state['step'] ** 0.25
                    scaled_grad = grad.clamp(-clip, clip)

                """
                ==== Momentum Update ====
                """
                exp_avg_bar, exp_avg = self._update_momentum(state, group, scaled_grad, beta1, eps1)

                """
                ==== AdaBelief ====
                AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients

                https://arxiv.org/abs/2010.07468
                https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adabelief.py
                """
                res = (scaled_grad - exp_avg_bar).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt())

                """
                === Grams ===
                Grams: Gradient Descent with Adaptive Momentum Scaling
                https://arxiv.org/abs/2412.17107
                https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/grams.py
                """
                grams_update = update_p.abs() * grad.sign()
                alpha = 1.0 * group['beta1_decay'] ** state["step"]
                update_p = alpha * grams_update + (1 - alpha) * update_p

                if state["step"] < group["warmup_steps"] / 2:
                    """
                    === 正交梯度 ===
                    Grokking at the Edge of Numerical Stability

                    https://arxiv.org/abs/2501.04697
                    https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev
                    """
                    update_p = self.orthograd_(p, update_p)

                """
                ==== Automagic lrmask ====
                https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py
                """
                new_lr = self._update_learning_rate_mask(state, group, grad)

                if "row_scaling" in state:
                    new_lr = new_lr * state["row_scaling"]

                """
                Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias?
                https://arxiv.org/abs/2504.12883
                """
                if state["step"] < group["warmup_steps"] / 2 and group["weight_decay"] > 0:
                    """
                    Adaptive Weight Decay for Deep Neural Networks
                    https://arxiv.org/abs/1907.08931
                    """
                    param_abs_grad = abs_grad.mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - new_lr * group["weight_decay"] * theta)

                update_p = update_p.mul(new_lr)
                p.add_(-update_p)

        if self.config.verbose:
            print([group["lr"] for group in self.param_groups])
        return loss

    def state_dict(self) -> Dict[str, Any]:
        """Get the optimizer state dictionary."""
        state = super().state_dict()
        state['magic_version'] = 1
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the optimizer state dictionary."""
        if 'magic_version' not in state_dict or state_dict['magic_version'] != 1:
            print('[WARNING] You loaded an unexpected state dict, some dynamic mask parameters may not be properly synchronized!')
        super().load_state_dict(state_dict)

class Automagic_CameAMP8bit(BaseOptimizer):
    """8-bit version of Automagic_CameAMP optimizer."""

    def __init__(self, params, **kwargs):
        config = OptimizerConfig(**kwargs)
        super().__init__(params, config)

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """Initialize 8-bit optimizer state for a parameter.

        Args:
            p: Parameter tensor
            group: Parameter group containing optimization settings
        """
        device = p.device
        shape = p.shape
        state = self.state[p]

        # Basic state initialization
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)

        # Learning rate mask initialization
        lr_mask_init = torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr
        q_lr_mask, q_lr_mask_scale = F.quantize_blockwise(lr_mask_init, blocksize=4096)
        state.setdefault('lr_mask_q', q_lr_mask)
        state.setdefault('lr_mask_q_scale', q_lr_mask_scale)
        state.setdefault('avg_lr', float(self.config.lr))

        last_polarity_fp32 = torch.zeros(shape, dtype=torch.bool, device=device)
        q_last_polarity, q_last_polarity_scale = F.quantize_blockwise(last_polarity_fp32, blocksize=4096)
        state.setdefault('last_polarity_q', q_last_polarity)
        state.setdefault('last_polarity_q_scale', q_last_polarity_scale)

        # Momentum and variance initialization
        exp_avg_fp32 = torch.zeros_like(p)
        q_exp_avg, q_exp_avg_scale = F.quantize_blockwise(exp_avg_fp32, blocksize=4096)
        state.setdefault("exp_avg_q", q_exp_avg)
        state.setdefault("exp_avg_q_scale", q_exp_avg_scale)

        if group["came"]:
            exp_avg_sq_fp32 = torch.zeros_like(p)
            q_exp_avg_sq, q_exp_avg_sq_scale = F.quantize_blockwise(exp_avg_sq_fp32, blocksize=4096)
            state.setdefault("exp_avg_sq_q", q_exp_avg_sq)
            state.setdefault("exp_avg_sq_q_scale", q_exp_avg_sq_scale)

        exp_avg_res_fp32 = torch.zeros_like(p)
        q_exp_avg_res, q_exp_avg_res_scale = F.quantize_blockwise(exp_avg_res_fp32, blocksize=4096)
        state.setdefault("exp_avg_res_q", q_exp_avg_res)
        state.setdefault("exp_avg_res_q_scale", q_exp_avg_res_scale)

        s_fp32 = torch.zeros_like(p)
        q_s, q_s_scale = F.quantize_blockwise(s_fp32, blocksize=4096)
        state.setdefault("s_q", q_s)
        state.setdefault("s_q_scale", q_s_scale)

        # Full finetune initialization
        if group is not None and group['full_finetune'] is False:
            """
            ==== ALLoRA ====
            ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            https://arxiv.org/abs/2410.09692
            """
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                row_scaling = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))
                q_row_scaling, q_row_scaling_scale = F.quantize_blockwise(row_scaling, blocksize=4096)
                state.setdefault("row_scaling_q", q_row_scaling)
                state.setdefault("row_scaling_q_scale", q_row_scaling_scale)

    @staticmethod
    def _update_torque_aware_momentum(state: Dict[str, Any], scaled_grad: torch.Tensor, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum using Torque-Aware Momentum during early training.

        Implementation from:
        https://arxiv.org/abs/2412.18790
        https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/tam.py

        Args:
            state: Optimizer state for the parameter
            scaled_grad: Scaled gradient tensor
            eps1: Epsilon parameter for numerical stability

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated momentum average
        """
        # Set fixed beta values for early training
        beta1, beta2, beta3 = 0.9, 0.999, 0.9999
        decay_rate = 0.9

        # Get state tensors
        s = F.dequantize_blockwise(state['s_q'], state['s_q_scale'], blocksize=4096)
        exp_avg =F.dequantize_blockwise(state['exp_avg_q'], state['exp_avg_q_scale'], blocksize=4096)

        # Calculate correlation between normalized momentum and gradient
        corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(scaled_grad, p=2.0, dim=0))

        # Update correlation state
        s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

        # Calculate torque-aware update
        d = ((1.0 + s) / 2.0).add_(eps1).mul_(scaled_grad)

        # Update momentum
        exp_avg.mul_(beta1).add_(d)

        # Calculate momentum average
        exp_avg_bar = exp_avg

        return exp_avg_bar, exp_avg

    @staticmethod
    def _update_torque_aware_momentum(state: Dict[str, Any], scaled_grad: torch.Tensor, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum using consistency-based approach after early training.

        Implementation from:
        Towards Faster Training of Diffusion Models: An Inspiration of A Consistency Phenomenon
        https://arxiv.org/abs/2404.07946

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings
            scaled_grad: Scaled gradient tensor
            beta1: Beta1 parameter for momentum

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated momentum average
        """
        # Calculate time-dependent beta1
        beta1_t = max(beta1 * group['beta1_decay'] ** state["step"], 0.4)

        # Get momentum state
        exp_avg = F.dequantize_blockwise(state['exp_avg_q'], state['exp_avg_q_scale'], blocksize=4096)

        # Update momentum with time-dependent beta1
        exp_avg.mul_(beta1_t).add_(scaled_grad, alpha=1 - beta1_t)

        # Calculate momentum average
        exp_avg_bar = exp_avg * beta1 + scaled_grad * (1 - beta1)

        return exp_avg_bar, exp_avg

    @staticmethod
    def _update_post_warmup_lr_mask(state: Dict[str, Any], group: Dict[str, Any]) -> torch.Tensor:
        """Update learning rate mask after warmup phase.

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings

        Returns:
            torch.Tensor: Updated learning rate mask
        """
        new_lr = F.dequantize_blockwise(state['lr_mask_q'], state['lr_mask_q_scale'], blocksize=4096)

        # Update maximum learning rate if needed
        if group["lr"] > state["lr_max"]:
            state["lr_max"] = group["lr"]

        # Scale learning rate if current lr is less than maximum
        if group["lr"] < state["lr_max"]:
            new_lr = new_lr * max((group["lr"] / state["lr_max"]), 0.1)

        return new_lr

    def _update_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """Update learning rate mask during warmup phase.

        Args:
            state: Optimizer state for the parameter
            group: Parameter group containing optimization settings
            grad: Gradient tensor

        Returns:
            torch.Tensor: Updated learning rate mask
        """
        # Update polarity tracking
        last_polarity = F.dequantize_blockwise(state['last_polarity_q'], state['last_polarity_q_scale'], blocksize=4096)
        current_polarity = (grad > 0)
        sign_agree = torch.where(last_polarity == current_polarity, 1, -1)
        q_last_polarity, q_last_polarity_scale = F.quantize_blockwise(current_polarity, blocksize=4096)
        state['last_polarity_q'] = q_last_polarity
        state['last_polarity_q_scale'] = q_last_polarity_scale

        # Calculate new learning rate
        lr_mask = F.dequantize_blockwise(state['lr_mask_q'], state['lr_mask_q_scale'], blocksize=4096)
        new_lr = torch.where(
            sign_agree > 0,
            lr_mask + self.config.lr_bump,
            lr_mask - self.config.lr_bump
        )

        # Clamp learning rate to valid range
        new_lr = torch.clamp(new_lr, min=self.config.min_lr, max=self.config.max_lr)

        # Update quantized learning rate mask
        q_lr_mask, q_lr_mask_scale = F.quantize_blockwise(new_lr, blocksize=4096)
        state['lr_mask_q'] = q_lr_mask
        state['lr_mask_q_scale'] = q_lr_mask_scale
        state['avg_lr'] = torch.mean(new_lr).item()

        return new_lr

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with 8-bit quantization."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                """
                === grad 初始化 ===
                ==== AGR自適應梯度正則 ====
                Adaptive Gradient Regularization: A Faster and Generalizable Optimization Technique for Deep Neural Networks
                https://arxiv.org/pdf/2407.16944
                """
                abs_grad = torch.abs(p.grad)
                alpha = abs_grad / sum_abs_all_group_grads
                grad = p.grad * (1 - alpha)

                # === state 初始化 ===
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                self._step = state["step"]
                if state["step"] == group.get("warmup_steps", 0):
                    if 's' in state:
                        del state['s']
                        del state['s_q']
                        del state['s_q_scale']
                    if 'last_polarity' in state:
                        del state['last_polarity']
                        del state['last_polarity_q']
                        del state['last_polarity_q_scale']

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2, eps3 = group["eps"]
                exp_avg = F.dequantize_blockwise(state['exp_avg_q'], state['exp_avg_q_scale'], blocksize=4096)
                exp_avg_res = F.dequantize_blockwise(state['exp_avg_res_q'], state['exp_avg_res_q_scale'], blocksize=4096)

                if group["came"]:
                    """
                    CAME: Confidence-guided Adaptive Memory Efficient Optimization
                    https://arxiv.org/pdf/2411.02853
                    https://github.com/yangluo7/CAME
                    """
                    exp_avg_sq = F.dequantize_blockwise(state['exp_avg_sq_q'], state['exp_avg_sq_q_scale'], blocksize=4096)
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())

                    scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))
                else:
                    # Adabelief
                    clip = state['step'] ** 0.25
                    scaled_grad = grad.clamp(-clip, clip)

                """
                ==== Momentum Update ====
                """
                exp_avg_bar, exp_avg = self._update_momentum(state, group, scaled_grad, beta1, eps1)

                """
                ==== AdaBelief ====
                AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients

                https://arxiv.org/abs/2010.07468
                https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adabelief.py
                """
                res = (scaled_grad - exp_avg_bar).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt())

                """
                === Grams ===
                Grams: Gradient Descent with Adaptive Momentum Scaling
                https://arxiv.org/abs/2412.17107
                https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/grams.py
                """
                grams_update = update_p.abs() * grad.sign()
                alpha = 1.0 * group['beta1_decay'] ** state["step"]
                update_p = alpha * grams_update + (1 - alpha) * update_p

                if state["step"] < group["warmup_steps"] / 2:
                    """
                    === 正交梯度 ===
                    Grokking at the Edge of Numerical Stability

                    https://arxiv.org/abs/2501.04697
                    https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev
                    """
                    update_p = self.orthograd_(p, update_p)

                """
                ==== Automagic lrmask ====
                https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py
                """
                new_lr = self._update_learning_rate_mask(state, group, grad)

                if "row_scaling" in state:
                    row_scaling = F.dequantize_blockwise(state['row_scaling_q'], state['row_scaling_q_scale'], blocksize=4096)
                    new_lr = new_lr * row_scaling

                update_p = update_p.mul(new_lr)

                p.add_(-update_p)

        if self.config.verbose:
            print([group["lr"] for group in self.param_groups])
        return loss

    def state_dict(self) -> Dict[str, Any]:
        """Get the 8-bit optimizer state dictionary."""
        orig_sd = super().state_dict()
        new_state = {}
        for k, v in orig_sd['state'].items():
            # Don't store unquantized tensors
            save_state = {kk: vv for kk, vv in v.items()
                         if kk not in ('lr_mask_q', 'lr_mask_q_scale', 'last_polarity_q', 'last_polarity_q_scale', 'exp_avg_q', 'exp_avg_q_scale', 'exp_avg_sq_q', 'exp_avg_sq_q_scale', 'exp_avg_res_q', 'exp_avg_res_q_scale', 'row_scaling_q', 'row_scaling_q_scale', 's_q', 's_q_scale')}
            # Save quantized tensors
            if 'lr_mask_q' in v and 'lr_mask_q_scale' in v:
                save_state['lr_mask_q'] = v['lr_mask_q']
                save_state['lr_mask_q_scale'] = v['lr_mask_q_scale']
            if 'last_polarity_q' in v and 'last_polarity_q_scale' in v:
                save_state['last_polarity_q'] = v['last_polarity_q']
                save_state['last_polarity_q_scale'] = v['last_polarity_q_scale']
            if 'exp_avg_q' in v and 'exp_avg_q_scale' in v:
                save_state['exp_avg_q'] = v['exp_avg_q']
                save_state['exp_avg_q_scale'] = v['exp_avg_q_scale']
            if 'exp_avg_sq_q' in v and 'exp_avg_sq_q_scale' in v:
                save_state['exp_avg_sq_q'] = v['exp_avg_sq_q']
                save_state['exp_avg_sq_q_scale'] = v['exp_avg_sq_q_scale']
            if 'exp_avg_res_q' in v and 'exp_avg_res_q_scale' in v:
                save_state['exp_avg_res_q'] = v['exp_avg_res_q']
                save_state['exp_avg_res_q_scale'] = v['exp_avg_res_q_scale']
            if 'row_scaling_q' in v and 'row_scaling_q_scale' in v:
                save_state['row_scaling_q'] = v['row_scaling_q']
                save_state['row_scaling_q_scale'] = v['row_scaling_q_scale']
            if 's_q' in v and 's_q_scale' in v:
                save_state['s_q'] = v['s_q']
                save_state['s_q_scale'] = v['s_q_scale']

            new_state[k] = save_state
        orig_sd['state'] = new_state
        orig_sd['magic8_version'] = 1
        return orig_sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the 8-bit optimizer state dictionary."""
        if 'magic8_version' not in state_dict or state_dict['magic8_version'] != 1:
            print('[WARNING] You loaded an unexpected state dict, some 8-bit parameters may not be synchronized!')
        basic_sd = {'state': {}, 'param_groups': state_dict['param_groups']}
        for k, v in state_dict['state'].items():
            basic_sd['state'][k] = {kk: vv for kk, vv in v.items()
                     if kk not in ('lr_mask_q', 'lr_mask_q_scale', 'last_polarity_q', 'last_polarity_q_scale', 'exp_avg_q', 'exp_avg_q_scale', 'exp_avg_sq_q', 'exp_avg_sq_q_scale', 'exp_avg_res_q', 'exp_avg_res_q_scale', 'row_scaling_q', 'row_scaling_q_scale', 's_q', 's_q_scale')}
        super().load_state_dict(basic_sd)
        # Restore quantized tensors
        param_map = [p for g in self.param_groups for p in g['params']]
        for idx, p in enumerate(param_map):
            idx_str = str(idx)
            if idx_str not in state_dict['state']:
                continue
            src = state_dict['state'][idx_str]
            st = self.state[p]
            if 'lr_mask_q' in src and 'lr_mask_q_scale' in src:
                st['lr_mask_q'] = src['lr_mask_q']
                st['lr_mask_q_scale'] = src['lr_mask_q_scale']
            if 'last_polarity_q' in src and 'last_polarity_q_scale' in src:
                st['last_polarity_q'] = src['last_polarity_q']
                st['last_polarity_q_scale'] = src['last_polarity_q_scale']
            if 'exp_avg_q' in src and 'exp_avg_q_scale' in src:
                st['exp_avg_q'] = src['exp_avg_q']
                st['exp_avg_q_scale'] = src['exp_avg_q_scale']
            if 'exp_avg_sq_q' in src and 'exp_avg_sq_q_scale' in src:
                st['exp_avg_sq_q'] = src['exp_avg_sq_q']
                st['exp_avg_sq_q_scale'] = src['exp_avg_sq_q_scale']
            if 'exp_avg_res_q' in src and 'exp_avg_res_q_scale' in src:
                st['exp_avg_res_q'] = src['exp_avg_res_q']
                st['exp_avg_res_q_scale'] = src['exp_avg_res_q_scale']
            if 'row_scaling_q' in src and 'row_scaling_q_scale' in src:
                st['row_scaling_q'] = src['row_scaling_q']
                st['row_scaling_q_scale'] = src['row_scaling_q_scale']
            if 's_q' in src and 's_q_scale' in src:
                st['s_q'] = src['s_q']
                st['s_q_scale'] = src['s_q_scale']