import torch
from typing import List, Dict, Any, Optional, Tuple, Deque
from torch.nn.functional import normalize
from dataclasses import dataclass
from collections import deque
import math
import random

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
    weight_decay: float = 5e-4
    warmup_steps: int = 500
    context_window: int = 30,
    edge_threshold: float = 0.6,
    adaptation_rate: float = 0.25,
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
            context_window=config.context_window,
            edge_threshold=config.edge_threshold,
            adaptation_rate=config.adaptation_rate,
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
        beta1, beta2, beta3 = group["betas"]
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
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

            if self._step < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
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
                agr = abs_grad / sum_abs_all_group_grads
                grad = p.grad * (1 - agr)

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

                if group.get("came", True):
                    """
                    CAME: Confidence-guided Adaptive Memory Efficient Optimization
                    https://arxiv.org/pdf/2411.02853
                    https://github.com/yangluo7/CAME
                    """
                    exp_avg_sq = state["exp_avg_sq"]
                    grad_p2 = grad.pow(2) + eps1
                    if state['step'] == 1:
                        exp_avg_sq.add_(grad_p2)
                        continue
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
                    scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))
                    exp_avg_sq.mul_(beta2).add_(grad_p2, alpha=1 - beta2)
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
                update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt() + eps2)

                """
                === Grams ===
                Grams: Gradient Descent with Adaptive Momentum Scaling
                https://arxiv.org/abs/2412.17107
                https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/grams.py
                """
                grams_update = update_p.abs() * grad.sign()
                alpha = 1.0 * group['beta1_decay'] ** state["step"]
                update_p = alpha * grams_update + (1 - alpha) * update_p

                if state["step"] < group.get("warmup_steps", 500) / 2:
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
                if state["step"] < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
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

        torch.cuda.synchronize()

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
    """8-bit version of Automagic_CameAMP optimizer using bitsandbytes."""

    def __init__(self, params, **kwargs):
        config = OptimizerConfig(**kwargs)
        super().__init__(params, config)
        self._step = 0

        # Check if bitsandbytes is available and supports CUDA
        try:
            import bitsandbytes.functional as F
            self.F = F
            # Test if quantization works
            test_tensor = torch.randn(10, 10, dtype=torch.float32)
            _, _ = F.quantize_blockwise(test_tensor, blocksize=256)
        except Exception as e:
            raise RuntimeError(f"bitsandbytes 8-bit quantization not available: {e}")

    def _quantize_tensor(self, tensor: torch.Tensor, blocksize: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor using bitsandbytes blockwise quantization.

        Args:
            tensor: Input tensor to quantize
            blocksize: Block size for quantization (default: 4096)

        Returns:
            Tuple of (quantized_tensor, scale_tensor)
        """
        return self.F.quantize_blockwise(tensor, blocksize=blocksize)

    def _dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor, blocksize: int = 4096) -> torch.Tensor:
        """Dequantize a tensor using bitsandbytes blockwise dequantization.

        Args:
            quantized: Quantized tensor
            scale: Scale tensor from quantization
            blocksize: Block size for dequantization (default: 4096)

        Returns:
            Dequantized tensor
        """
        return self.F.dequantize_blockwise(quantized, scale, blocksize=blocksize)

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

        # Learning rate mask initialization with quantization
        lr_mask_init = torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr
        q_lr_mask, q_lr_mask_scale = self._quantize_tensor(lr_mask_init)
        state.setdefault('lr_mask_q', q_lr_mask)
        state.setdefault('lr_mask_q_scale', q_lr_mask_scale)
        state.setdefault('avg_lr', float(self.config.lr))

        # Boolean polarity tracking (no quantization needed for boolean)
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # Momentum initialization with quantization
        exp_avg_fp32 = torch.zeros_like(p, dtype=torch.float32)
        q_exp_avg, q_exp_avg_scale = self._quantize_tensor(exp_avg_fp32)
        state.setdefault("exp_avg_q", q_exp_avg)
        state.setdefault("exp_avg_q_scale", q_exp_avg_scale)

        # Variance initialization with quantization (for CAME)
        if group and group.get('came', True):
            exp_avg_sq_fp32 = torch.zeros_like(p, dtype=torch.float32)
            q_exp_avg_sq, q_exp_avg_sq_scale = self._quantize_tensor(exp_avg_sq_fp32)
            state.setdefault("exp_avg_sq_q", q_exp_avg_sq)
            state.setdefault("exp_avg_sq_q_scale", q_exp_avg_sq_scale)

        # AdaBelief residual initialization with quantization
        exp_avg_res_fp32 = torch.zeros_like(p, dtype=torch.float32)
        q_exp_avg_res, q_exp_avg_res_scale = self._quantize_tensor(exp_avg_res_fp32)
        state.setdefault("exp_avg_res_q", q_exp_avg_res)
        state.setdefault("exp_avg_res_q_scale", q_exp_avg_res_scale)

        # Torque-aware momentum state initialization with quantization
        s_fp32 = torch.zeros_like(p, dtype=torch.float32)
        q_s, q_s_scale = self._quantize_tensor(s_fp32)
        state.setdefault("s_q", q_s)
        state.setdefault("s_q_scale", q_s_scale)

        # Full finetune initialization (ALLoRA)
        if group is not None and group.get('full_finetune', False) is False:
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                row_scaling = 1.0 / torch.sqrt(row_norm + 1.0 / (group.get('eta', 2.0)**2))
                q_row_scaling, q_row_scaling_scale = self._quantize_tensor(row_scaling)
                state.setdefault("row_scaling_q", q_row_scaling)
                state.setdefault("row_scaling_q_scale", q_row_scaling_scale)

    def _update_torque_aware_momentum(self, state: Dict[str, Any], scaled_grad: torch.Tensor, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum using Torque-Aware Momentum during early training."""
        # Set fixed beta values for early training
        beta1, beta2, beta3 = 0.9, 0.999, 0.9999
        decay_rate = 0.9

        # Dequantize state tensors
        s = self._dequantize_tensor(state['s_q'], state['s_q_scale'])
        exp_avg = self._dequantize_tensor(state['exp_avg_q'], state['exp_avg_q_scale'])

        # Calculate correlation between normalized momentum and gradient
        corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(scaled_grad, p=2.0, dim=0))

        # Update correlation state
        s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

        # Calculate torque-aware update
        d = ((1.0 + s) / 2.0).add_(eps1).mul_(scaled_grad)

        # Update momentum
        exp_avg.mul_(beta1).add_(d)

        # Re-quantize updated states
        state['s_q'], state['s_q_scale'] = self._quantize_tensor(s)
        state['exp_avg_q'], state['exp_avg_q_scale'] = self._quantize_tensor(exp_avg)

        return exp_avg, exp_avg

    def _update_consistency_momentum(self, state: Dict[str, Any], group: Dict[str, Any], scaled_grad: torch.Tensor, beta1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum using consistency-based approach after early training."""
        # Calculate time-dependent beta1
        beta1_orig, beta2, beta3 = group["betas"]
        beta1_t = max(beta1_orig * group.get('beta1_decay', 0.9995) ** state["step"], 0.4)

        # Dequantize momentum state
        exp_avg = self._dequantize_tensor(state['exp_avg_q'], state['exp_avg_q_scale'])

        # Update momentum with time-dependent beta1
        exp_avg.mul_(beta1_t).add_(scaled_grad, alpha=1 - beta1_t)

        # Calculate momentum average
        exp_avg_bar = exp_avg * beta1_orig + scaled_grad * (1 - beta1_orig)

        # Re-quantize updated state
        state['exp_avg_q'], state['exp_avg_q_scale'] = self._quantize_tensor(exp_avg)

        return exp_avg_bar, exp_avg

    def _update_post_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any]) -> torch.Tensor:
        """Update learning rate mask after warmup phase."""
        new_lr = self._dequantize_tensor(state['lr_mask_q'], state['lr_mask_q_scale'])

        # Update maximum learning rate if needed
        if group["lr"] > state["lr_max"]:
            state["lr_max"] = group["lr"]

        # Scale learning rate if current lr is less than maximum
        if group["lr"] < state["lr_max"]:
            new_lr = new_lr * max((group["lr"] / state["lr_max"]), 0.1)

        # Re-quantize updated learning rate mask
        state['lr_mask_q'], state['lr_mask_q_scale'] = self._quantize_tensor(new_lr)

        return new_lr

    def _update_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """Update learning rate mask during warmup phase."""
        # Update polarity tracking (boolean tensor - no quantization needed)
        last_polarity = state['last_polarity']
        current_polarity = (grad > 0)
        sign_agree = torch.where(last_polarity == current_polarity, 1, -1)
        state['last_polarity'] = current_polarity

        # Dequantize learning rate mask
        lr_mask = self._dequantize_tensor(state['lr_mask_q'], state['lr_mask_q_scale'])

        # Calculate new learning rate
        new_lr = torch.where(
            sign_agree > 0,
            lr_mask + self.config.lr_bump,
            lr_mask - self.config.lr_bump
        )

        # Clamp learning rate to valid range
        new_lr = torch.clamp(new_lr, min=self.config.min_lr, max=self.config.max_lr)

        # Re-quantize updated learning rate mask
        state['lr_mask_q'], state['lr_mask_q_scale'] = self._quantize_tensor(new_lr)
        state['avg_lr'] = torch.mean(new_lr).item()

        return new_lr

    def _update_momentum(self, state: Dict[str, Any], group: Dict[str, Any], scaled_grad: torch.Tensor, beta1: float, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update momentum based on training phase."""
        if state["step"] < group.get("warmup_steps", 500) / 2:
            return self._update_torque_aware_momentum(state, scaled_grad, eps1)
        else:
            return self._update_consistency_momentum(state, group, scaled_grad, beta1)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with 8-bit quantization."""
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Collect gradients for adaptive gradient regularization
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))
            if len(grads_this_group) == 0:
                continue

            all_group_grads = torch.cat(grads_this_group)
            abs_all_group_grads = torch.abs(all_group_grads)
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads)

            if self._step < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                # Adaptive Gradient Regularization
                abs_grad = torch.abs(p.grad)
                alpha = abs_grad / sum_abs_all_group_grads
                grad = p.grad * (1 - alpha)

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                self._step = state["step"]

                # Clean up warmup-specific states after warmup
                if state["step"] == group.get("warmup_steps", 0):
                    if 's_q' in state:
                        del state['s_q']
                        del state['s_q_scale']
                    if 'last_polarity' in state:
                        del state['last_polarity']

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2, eps3 = group["eps"]

                # Dequantize main state tensors
                exp_avg = self._dequantize_tensor(state['exp_avg_q'], state['exp_avg_q_scale'])
                exp_avg_res = self._dequantize_tensor(state['exp_avg_res_q'], state['exp_avg_res_q_scale'])

                # CAME or AdaBelief gradient scaling
                if group.get("came", True):
                    # CAME: Confidence-guided Adaptive Memory Efficient Optimization
                    exp_avg_sq = self._dequantize_tensor(state['exp_avg_sq_q'], state['exp_avg_sq_q_scale'])
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
                    scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))

                    # Re-quantize exp_avg_sq
                    quantized, scale_tensor = self._quantize_tensor(exp_avg_sq)
                    state['exp_avg_sq_q'] = quantized
                    state['exp_avg_sq_q_scale'] = scale_tensor
                    state['exp_avg_sq'] = exp_avg_sq
                else:
                    # AdaBelief gradient clipping
                    clip = state['step'] ** 0.25
                    scaled_grad = grad.clamp(-clip, clip)

                # Update momentum
                exp_avg_bar, exp_avg = self._update_momentum(state, group, scaled_grad, beta1, eps1)

                # AdaBelief variance update
                res = (scaled_grad - exp_avg_bar).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt())

                # Grams: Gradient Descent with Adaptive Momentum Scaling
                grams_update = update_p.abs() * grad.sign()
                alpha = 1.0 * group.get('beta1_decay', 0.9995) ** state["step"]
                update_p = alpha * grams_update + (1 - alpha) * update_p

                # Orthogonal gradient during early warmup
                if state["step"] < group.get("warmup_steps", 500) / 2:
                    update_p = self.orthograd_(p, update_p)

                # Update learning rate mask
                new_lr = self._update_learning_rate_mask(state, group, grad)

                # Apply row scaling if available (ALLoRA)
                if "row_scaling_q" in state:
                    row_scaling = self._dequantize_tensor(state['row_scaling_q'], state['row_scaling_q_scale'])
                    new_lr = new_lr * row_scaling

                if state["step"] < group.get("warmup_steps", 500)/ 2 and group["weight_decay"] > 0:
                    param_abs_grad = abs_grad.mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - new_lr * group["weight_decay"] * theta)

                # Apply learning rate and update parameters
                update_p = update_p.mul(new_lr)
                p.add_(-update_p)

        if self.config.verbose:
            print([group["lr"] for group in self.param_groups])

        torch.cuda.synchronize()

        return loss

    def _update_learning_rate_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """Update learning rate mask based on training phase."""
        if state["step"] < group.get("warmup_steps", 500):
            return self._update_warmup_lr_mask(state, group, grad)
        else:
            return self._update_post_warmup_lr_mask(state, group)

    def state_dict(self) -> Dict[str, Any]:
        """Get the 8-bit optimizer state dictionary."""
        orig_sd = super().state_dict()
        new_state = {}

        # List of quantized state keys to preserve
        quantized_keys = {
            'lr_mask_q', 'lr_mask_q_scale',
            'exp_avg_q', 'exp_avg_q_scale',
            'exp_avg_sq_q', 'exp_avg_sq_q_scale',
            'exp_avg_res_q', 'exp_avg_res_q_scale',
            'row_scaling_q', 'row_scaling_q_scale',
            's_q', 's_q_scale'
        }

        for k, v in orig_sd['state'].items():
            # Save all non-quantized state and quantized tensors
            save_state = {}
            for kk, vv in v.items():
                if kk in quantized_keys or kk not in ['lr_mask', 'exp_avg', 'exp_avg_sq', 'exp_avg_res', 'row_scaling', 's']:
                    save_state[kk] = vv
            new_state[k] = save_state

        orig_sd['state'] = new_state
        orig_sd['magic8_version'] = 2  # Updated version number
        return orig_sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the 8-bit optimizer state dictionary."""
        if 'magic8_version' not in state_dict or state_dict['magic8_version'] < 2:
            print('[WARNING] You loaded an older state dict version, some 8-bit parameters may not be synchronized!')

        # Create basic state dict without quantized tensors for parent class
        basic_sd = {'state': {}, 'param_groups': state_dict['param_groups']}
        quantized_keys = {
            'lr_mask_q', 'lr_mask_q_scale',
            'exp_avg_q', 'exp_avg_q_scale',
            'exp_avg_sq_q', 'exp_avg_sq_q_scale',
            'exp_avg_res_q', 'exp_avg_res_q_scale',
            'row_scaling_q', 'row_scaling_q_scale',
            's_q', 's_q_scale'
        }

        for k, v in state_dict['state'].items():
            basic_sd['state'][k] = {kk: vv for kk, vv in v.items() if kk not in quantized_keys}

        super().load_state_dict(basic_sd)

        # Restore quantized tensors
        param_map = [p for g in self.param_groups for p in g['params']]
        for idx, p in enumerate(param_map):
            idx_str = str(idx)
            if idx_str not in state_dict['state']:
                continue
            src = state_dict['state'][idx_str]
            st = self.state[p]

            # Restore all quantized tensor pairs
            for base_key in ['lr_mask', 'exp_avg', 'exp_avg_sq', 'exp_avg_res', 'row_scaling', 's']:
                q_key = f'{base_key}_q'
                scale_key = f'{base_key}_q_scale'
                if q_key in src and scale_key in src:
                    st[q_key] = src[q_key]
                    st[scale_key] = src[scale_key]

class ContextualOptimizationModule:
    """
    基於 C-Optim 的上下文感知優化模組
    """

    def __init__(self,
                 context_window: int = 50,
                 edge_threshold: float = 0.9,
                 adaptation_rate: float = 0.1):
        self.context_window = context_window
        self.edge_threshold = edge_threshold
        self.adaptation_rate = adaptation_rate

        # 上下文歷史
        self.gradient_history: Deque[torch.Tensor] = deque(maxlen=context_window)
        self.loss_history: Deque[float] = deque(maxlen=context_window)
        self.lr_history: Deque[float] = deque(maxlen=context_window)

        # 新增：性能追蹤
        self.performance_history: Deque[float] = deque(maxlen=50)
        self.convergence_rate: float = 0.0
        self.stable_steps: int = 0

    def update_context(self, gradient: torch.Tensor, loss: float, lr: float):
        """更新上下文信息"""
        self.gradient_history.append(gradient.detach().clone())
        self.loss_history.append(loss)
        self.lr_history.append(lr)

        # 計算性能指標
        if len(self.loss_history) >= 2:
            loss_improvement = self.loss_history[-2] - self.loss_history[-1]
            self.performance_history.append(loss_improvement)

            # 更新收斂率
            if len(self.performance_history) >= 10:
                recent_improvements = list(self.performance_history)[-10:]
                self.convergence_rate = sum(recent_improvements) / len(recent_improvements)

                # 追蹤穩定步數
                if abs(loss_improvement) < 1e-6:
                    self.stable_steps += 1
                else:
                    self.stable_steps = 0

    def compute_gradient_consistency(self) -> float:
        """計算梯度一致性指標"""
        if len(self.gradient_history) < 2:
            return 1.0

        recent_grads = list(self.gradient_history)[-10:]  # 最近10步

        # 計算梯度方向的一致性
        consistencies = []
        for i in range(1, len(recent_grads)):
            prev_grad = recent_grads[i-1].flatten()
            curr_grad = recent_grads[i].flatten()

            # 餘弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(prev_grad.unsqueeze(0), curr_grad.unsqueeze(0))
            consistencies.append(cos_sim.item())

        return sum(consistencies) / len(consistencies) if consistencies else 1.0

    def compute_loss_stability(self) -> float:
        """計算損失穩定性"""
        if len(self.loss_history) < 10:
            return 1.0

        recent_losses = torch.tensor(list(self.loss_history)[-10:])
        loss_std = torch.std(recent_losses)
        loss_mean = torch.mean(recent_losses)

        # 計算變異係數（越小越穩定）
        cv = loss_std / (loss_mean + 1e-8)
        # 轉換為穩定性分數（0-1，越大越穩定）
        stability = torch.exp(-cv).item()
        return min(1.0, max(0.0, stability))

    def detect_edge_case(self) -> bool:
        """檢測是否為邊緣情況 - LoRA 優化版本"""
        if len(self.loss_history) < 10:
            return False

        recent_losses = list(self.loss_history)[-10:]

        # 檢測損失震盪（放寬標準）
        loss_variance = torch.var(torch.tensor(recent_losses))
        loss_mean = torch.mean(torch.tensor(recent_losses))
        cv = loss_variance.sqrt() / (loss_mean + 1e-8)  # 變異係數

        # 檢測梯度一致性（降低要求）
        grad_consistency = self.compute_gradient_consistency()

        # 檢測是否陷入停滯（放寬條件）
        loss_stagnation = self.stable_steps > 30  # 增加到 30 步

        # LoRA 友好的邊緣情況判斷：更寬容的閾值
        # 變異係數閾值從 0.3 提高到 0.5
        # 梯度一致性閾值從 0.4 降低到 0.2
        is_edge = cv > 0.5 or grad_consistency < 0.2 or loss_stagnation

        return is_edge

    def compute_contextual_lr_multiplier(self) -> float:
        """計算上下文感知的學習率乘數 - LoRA 優化版本"""
        if len(self.loss_history) < 5:
            return 1.2  # LoRA 初期需要較高學習率

        # 分析損失趨勢（使用更長的歷史）
        recent_losses = torch.tensor(list(self.loss_history)[-10:])

        # 計算多種趨勢指標
        short_trend = (recent_losses[-1] - recent_losses[-3]) / 2  # 短期趨勢
        long_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)  # 長期趨勢

        # 分析梯度一致性和損失穩定性
        grad_consistency = self.compute_gradient_consistency()
        loss_stability = self.compute_loss_stability()

        # 檢測邊緣情況（放寬條件）
        is_edge = self.detect_edge_case()

        # 計算收斂速度因子
        convergence_factor = 1.0
        if len(self.performance_history) >= 5:
            avg_improvement = sum(list(self.performance_history)[-5:]) / 5
            if avg_improvement > 0:
                convergence_factor = min(2.0, 1.0 + avg_improvement * 150)  # 增強正向提升
            elif avg_improvement < -1e-4:
                convergence_factor = max(0.8, 1.0 + avg_improvement * 30)   # 減少負向懲罰

        # LoRA 優化的乘數計算邏輯
        if is_edge:
            # 邊緣情況：對 LoRA 更寬容，保持合理的學習率
            if grad_consistency > 0.7:
                multiplier = 1.0 + 0.5 * grad_consistency  # 1.0-1.35（提高基準）
            else:
                multiplier = 0.8 + 0.6 * grad_consistency  # 0.8-1.4（提高上限）

        elif short_trend > 1e-5:  # 短期損失明顯上升
            # LoRA 訓練中輕微的損失上升是正常的
            multiplier = 0.9 + 0.3 * loss_stability  # 0.9-1.2（提高基準）

        elif long_trend < -1e-6:  # 長期有改善趨勢
            if grad_consistency > 0.8 and loss_stability > 0.7:
                # 良好的一致性和穩定性 → 積極學習（LoRA 適合）
                multiplier = 1.5 + 0.8 * grad_consistency  # 1.5-2.3
            elif grad_consistency > 0.6:
                # 中等一致性 → 適度提升
                multiplier = 1.3 + 0.4 * grad_consistency  # 1.3-1.7
            else:
                # 低一致性但有改善 → 保守但不過度
                multiplier = 1.1 + 0.3 * grad_consistency  # 1.1-1.4

        elif abs(long_trend) < 1e-7:  # 停滯狀態
            if self.stable_steps > 15:  # 放寬停滯檢測
                # LoRA 訓練可能需要更強的突破力度
                multiplier = 1.6 + 0.4 * random.uniform(0, 1)  # 1.6-2.0（帶隨機性）
            else:
                multiplier = 1.2  # 保持較高基準

        else:
            # 預設情況：LoRA 友好的基準值
            base_multiplier = 1.1 + 0.3 * grad_consistency  # 1.1-1.4
            stability_bonus = 0.2 * loss_stability  # 0-0.2（增加穩定性獎勵）
            multiplier = base_multiplier + stability_bonus

        # 應用收斂速度因子
        multiplier *= convergence_factor

        # LoRA 優化的動態邊界調整
        if is_edge:
            min_mult = 0.6  # 即使在邊緣情況也不過度降低
            max_mult = 3.5  # 提高上限
        else:
            min_mult = 0.8  # 提高正常情況的最小值
            max_mult = 4.0 if grad_consistency > 0.9 and loss_stability > 0.8 else 3.0

        return max(min_mult, min(max_mult, multiplier))

class Automagic_CameAMP_COptim(BaseOptimizer):
    """
    整合 C-Optim 的 Automagic_CameAMP 優化器
    """

    def __init__(self, params, **kwargs):
        config = OptimizerConfig(**kwargs)
        super().__init__(params, config)

        self._step = 0

        # C-Optim 模組
        self.c_optim = ContextualOptimizationModule(
            context_window=kwargs.get('context_window', 50),
            edge_threshold=kwargs.get('edge_threshold', 0.9),
            adaptation_rate=kwargs.get('adaptation_rate', 0.1)
        )

        # 多尺度動量
        self.momentum_scales = [1, 5, 20, 100]  # 不同時間尺度

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """初始化包含 C-Optim 的狀態"""
        super()._init_state(p, group)

        state = self.state[p]

        # C-Optim 特定狀態
        state.setdefault('c_optim_context', {})
        state.setdefault('edge_case_count', 0)
        state.setdefault('contextual_lr_multiplier', 1.0)

        # 多尺度動量
        for scale in self.momentum_scales:
            state.setdefault(f'momentum_scale_{scale}', torch.zeros_like(p))
            state.setdefault(f'momentum_count_{scale}', 0)

    def _update_multiscale_momentum(self, state: Dict[str, Any], grad: torch.Tensor, beta: float = 0.9):
        """更新多尺度動量"""
        for scale in self.momentum_scales:
            momentum_key = f'momentum_scale_{scale}'
            count_key = f'momentum_count_{scale}'

            # 每 scale 步更新一次該尺度的動量
            if state['step'] % scale == 0:
                state[momentum_key].mul_(beta).add_(grad, alpha=1-beta)
                state[count_key] += 1

    def _get_contextual_update(self, state: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """獲取上下文感知的更新"""
        # 獲取不同尺度的動量
        momentum_contributions = []
        total_weight = 0

        for scale in self.momentum_scales:
            momentum = state[f'momentum_scale_{scale}']
            count = state[f'momentum_count_{scale}']

            if count > 0:
                # 權重與尺度成反比（短期動量權重更高）
                weight = 1.0 / scale
                momentum_contributions.append(momentum * weight)
                total_weight += weight

        if momentum_contributions and total_weight > 0:
            # 加權平均不同尺度的動量
            combined_momentum = sum(momentum_contributions) / total_weight
        else:
            combined_momentum = grad

        return combined_momentum

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """執行一步優化，整合 C-Optim 功能"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 更新全域上下文（如果有損失值）
        if loss is not None:
            avg_grad_norm = 0
            param_count = 0
            total_grad_norm = 0

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        avg_grad_norm += grad_norm
                        total_grad_norm += grad_norm ** 2
                        param_count += 1

            if param_count > 0:
                avg_grad_norm /= param_count
                rms_grad_norm = (total_grad_norm / param_count) ** 0.5

                # 使用 RMS 梯度範數提供更穩定的上下文信息
                self.c_optim.update_context(
                    torch.tensor(rms_grad_norm),
                    loss.item(),
                    group.get("lr", self.config.lr)
                )

        for group in self.param_groups:
            # 計算全域上下文乘數
            global_lr_multiplier = self.c_optim.compute_contextual_lr_multiplier()
            is_edge_case = self.c_optim.detect_edge_case()

            # 收集梯度進行 AGR
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))
            if len(grads_this_group) == 0:
                continue

            all_group_grads = torch.cat(grads_this_group)
            abs_all_group_grads = torch.abs(all_group_grads)
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads)

            if self._step < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                # AGR 正則化
                abs_grad = torch.abs(p.grad)
                alpha = abs_grad / sum_abs_all_group_grads
                grad = p.grad * (1 - alpha)

                # 初始化狀態
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                self._step = state["step"]

                # 更新多尺度動量
                self._update_multiscale_momentum(state, grad)

                # 邊緣情況處理 - 改進版
                if is_edge_case:
                    state['edge_case_count'] += 1
                    # 根據邊緣情況的嚴重程度動態調整
                    grad_consistency = self.c_optim.compute_gradient_consistency()
                    loss_stability = self.c_optim.compute_loss_stability()

                    # 組合穩定性分數
                    stability_score = (grad_consistency + loss_stability) / 2

                    # 動態邊緣因子：穩定性越高，懲罰越小
                    edge_factor = 0.4 + 0.4 * stability_score  # 0.4-0.8
                else:
                    edge_factor = 1.0
                    # 重置邊緣情況計數
                    state['edge_case_count'] = max(0, state['edge_case_count'] - 1)

                # 清理暖身狀態
                if state["step"] == group.get("warmup_steps", 0):
                    if 's' in state:
                        del state['s']
                    if 'last_polarity' in state:
                        del state['last_polarity']

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2, eps3 = group["eps"]
                exp_avg, exp_avg_res = state['exp_avg'], state["exp_avg_res"]

                # CAME 或 AdaBelief 梯度縮放
                if group.get("came", True):
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
                    scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))
                else:
                    clip = state['step'] ** 0.25
                    scaled_grad = grad.clamp(-clip, clip)

                # 動量更新（整合多尺度）
                contextual_update = self._get_contextual_update(state, scaled_grad)
                exp_avg_bar, exp_avg = self._update_momentum(state, group, contextual_update, beta1, eps1)

                # AdaBelief 變異數更新
                res = (scaled_grad - exp_avg_bar).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt() + eps2)

                # Grams 更新
                grams_update = update_p.abs() * grad.sign()
                alpha = 1.0 * group.get('beta1_decay', 0.9995) ** state["step"]
                update_p = alpha * grams_update + (1 - alpha) * update_p

                # 正交梯度（早期暖身）
                if state["step"] < group.get("warmup_steps", 500) / 2:
                    update_p = self.orthograd_(p, update_p)

                # 學習率遮罩更新
                new_lr = self._update_learning_rate_mask(state, group, grad)

                # 應用上下文感知的學習率調整
                new_lr = new_lr * global_lr_multiplier * edge_factor
                state['contextual_lr_multiplier'] = global_lr_multiplier * edge_factor

                # 行縮放（ALLoRA）
                if "row_scaling" in state:
                    new_lr = new_lr * state["row_scaling"]

                # 自適應權重衰減
                if state["step"] < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
                    param_abs_grad = abs_grad.mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - new_lr * group["weight_decay"] * theta)

                # 應用更新
                update_p = update_p.mul(new_lr)
                p.add_(-update_p)

        if self.config.verbose:
            print(f"Lr: {[group['lr'] for group in self.param_groups]}")
            print(f"Contextual Lr Multiplier: {global_lr_multiplier:.4f}")
            print(f"Edge Case: {is_edge_case}")

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

class Automagic_CameAMP_COptim8bit(BaseOptimizer):
    """8-bit 版本的 Automagic_CameAMP_COptim 優化器，結合 C-Optim 和 bitsandbytes 量化技術"""

    def __init__(self, params, **kwargs):
        config = OptimizerConfig(**kwargs)
        super().__init__(params, config)
        self._step = 0

        # 檢查 bitsandbytes 可用性
        try:
            import bitsandbytes.functional as F
            self.F = F
            # 測試量化功能
            test_tensor = torch.randn(10, 10, dtype=torch.float32)
            _, _ = F.quantize_blockwise(test_tensor, blocksize=256)
        except Exception as e:
            raise RuntimeError(f"bitsandbytes 8-bit 量化不可用: {e}")

        # C-Optim 模組初始化
        self.c_optim = ContextualOptimizationModule(
            context_window=kwargs.get('context_window', 50),
            edge_threshold=kwargs.get('edge_threshold', 0.9),
            adaptation_rate=kwargs.get('adaptation_rate', 0.1)
        )

        # 多尺度動量配置
        self.momentum_scales = kwargs.get('momentum_scales', [1, 5, 20, 100])

        print(f"[INFO] Initialize Automagic_CameAMP_COptim8bit optimizer")
        print(f"[INFO] C-Optim config: context window={self.c_optim.context_window}, edge threshold={self.c_optim.edge_threshold}")
        print(f"[INFO] Multi-scale momentum: {self.momentum_scales}")

    def _quantize_tensor(self, tensor: torch.Tensor, blocksize: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
        """量化張量為 8-bit"""
        if tensor.numel() == 0 or tensor.dtype == torch.bool:
            return tensor, None

        try:
            # 確保張量在正確的設備上且為 float32
            if tensor.device.type == 'cuda' and tensor.dtype != torch.float32:
                tensor = tensor.float()

            quantized, scale = self.F.quantize_blockwise(tensor, blocksize=blocksize)
            return quantized, scale
        except Exception:
            # 量化失敗時返回原始張量
            return tensor, None

    def _dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor, blocksize: int = 4096) -> torch.Tensor:
        """反量化張量"""
        if scale is None:
            return quantized

        try:
            return self.F.dequantize_blockwise(quantized, scale, blocksize=blocksize)
        except Exception:
            return quantized

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """初始化參數狀態，包含量化支援"""
        state = self.state[p]

        # 基本狀態初始化
        state.setdefault('step', 0)
        state.setdefault('exp_avg', torch.zeros_like(p))
        state.setdefault('exp_avg_res', torch.zeros_like(p))

        # 量化狀態
        state.setdefault('exp_avg_quantized', None)
        state.setdefault('exp_avg_scale', None)
        state.setdefault('exp_avg_res_quantized', None)
        state.setdefault('exp_avg_res_scale', None)

        # CAME 相關狀態
        if group and group.get("came", True):
            state.setdefault('exp_avg_sq', torch.zeros_like(p))
            state.setdefault('exp_avg_sq_quantized', None)
            state.setdefault('exp_avg_sq_scale', None)

        # 學習率遮罩
        state.setdefault('lr_mask', torch.ones_like(p))
        state.setdefault('lr_mask_quantized', None)
        state.setdefault('lr_mask_scale', None)

        # 極性追蹤（布林值，不需量化）
        state.setdefault('polarity', torch.zeros(p.shape, dtype=torch.bool, device=p.device))
        state.setdefault('last_polarity', torch.zeros(p.shape, dtype=torch.bool, device=p.device))

        # ALLoRA 行縮放
        if p.dim() >= 2:
            state.setdefault('row_scaling', torch.ones(p.shape[0], device=p.device))

        # C-Optim 特定狀態
        state.setdefault('c_optim_context', {})
        state.setdefault('edge_case_count', 0)
        state.setdefault('contextual_lr_multiplier', 1.0)

        # 多尺度動量（量化支援）
        for scale in self.momentum_scales:
            momentum_key = f'momentum_scale_{scale}'
            state.setdefault(momentum_key, torch.zeros_like(p))
            state.setdefault(f'{momentum_key}_quantized', None)
            state.setdefault(f'{momentum_key}_scale', None)
            state.setdefault(f'momentum_count_{scale}', 0)

    def _update_multiscale_momentum(self, state: Dict[str, Any], grad: torch.Tensor, beta: float = 0.9):
        """更新多尺度動量（8-bit 版本）"""
        for scale in self.momentum_scales:
            momentum_key = f'momentum_scale_{scale}'
            count_key = f'momentum_count_{scale}'

            # 每 scale 步更新一次該尺度的動量
            if state['step'] % scale == 0:
                # 反量化動量
                if state[f'{momentum_key}_quantized'] is not None:
                    momentum = self._dequantize_tensor(
                        state[f'{momentum_key}_quantized'],
                        state[f'{momentum_key}_scale']
                    )
                else:
                    momentum = state[momentum_key]

                # 更新動量
                momentum.mul_(beta).add_(grad, alpha=1-beta)

                # 重新量化並儲存
                quantized, scale_tensor = self._quantize_tensor(momentum)
                state[f'{momentum_key}_quantized'] = quantized
                state[f'{momentum_key}_scale'] = scale_tensor
                state[momentum_key] = momentum  # 保留一份用於計算

                state[count_key] += 1

    def _get_contextual_update(self, state: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """獲取上下文感知的更新（8-bit 版本）"""
        momentum_contributions = []
        total_weight = 0

        for scale in self.momentum_scales:
            momentum_key = f'momentum_scale_{scale}'
            count = state[f'momentum_count_{scale}']

            if count > 0:
                # 反量化動量
                if state[f'{momentum_key}_quantized'] is not None:
                    momentum = self._dequantize_tensor(
                        state[f'{momentum_key}_quantized'],
                        state[f'{momentum_key}_scale']
                    )
                else:
                    momentum = state[momentum_key]

                # 權重與尺度成反比
                weight = 1.0 / scale
                momentum_contributions.append(momentum * weight)
                total_weight += weight

        if momentum_contributions and total_weight > 0:
            combined_momentum = sum(momentum_contributions) / total_weight
        else:
            combined_momentum = grad

        return combined_momentum

    def _update_torque_aware_momentum(self, state: Dict[str, Any], scaled_grad: torch.Tensor, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新扭矩感知動量（8-bit 版本）"""
        # 反量化 exp_avg
        if state['exp_avg_quantized'] is not None:
            exp_avg = self._dequantize_tensor(state['exp_avg_quantized'], state['exp_avg_scale'])
        else:
            exp_avg = state['exp_avg']

        # 計算扭矩
        torque = torch.cross(exp_avg.view(-1), scaled_grad.view(-1), dim=0)
        torque_magnitude = torch.norm(torque) + eps1

        # 扭矩感知更新
        torque_factor = 1.0 / (1.0 + torque_magnitude)
        exp_avg_bar = exp_avg * torque_factor

        # 重新量化並儲存
        quantized, scale_tensor = self._quantize_tensor(exp_avg)
        state['exp_avg_quantized'] = quantized
        state['exp_avg_scale'] = scale_tensor
        state['exp_avg'] = exp_avg

        return exp_avg_bar, exp_avg

    def _update_consistency_momentum(self, state: Dict[str, Any], group: Dict[str, Any], scaled_grad: torch.Tensor, beta1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新一致性動量（8-bit 版本）"""
        # 反量化 exp_avg
        if state['exp_avg_quantized'] is not None:
            exp_avg = self._dequantize_tensor(state['exp_avg_quantized'], state['exp_avg_scale'])
        else:
            exp_avg = state['exp_avg']

        # 計算梯度一致性
        if state['step'] > 1:
            grad_consistency = torch.cosine_similarity(
                exp_avg.view(-1),
                scaled_grad.view(-1),
                dim=0
            ).item()
        else:
            grad_consistency = 1.0

        # 自適應 beta1
        adaptive_beta1 = beta1 * (0.5 + 0.5 * abs(grad_consistency))

        # 更新動量
        exp_avg.mul_(adaptive_beta1).add_(scaled_grad, alpha=1 - adaptive_beta1)
        exp_avg_bar = exp_avg.clone()

        # 重新量化並儲存
        quantized, scale_tensor = self._quantize_tensor(exp_avg)
        state['exp_avg_quantized'] = quantized
        state['exp_avg_scale'] = scale_tensor
        state['exp_avg'] = exp_avg

        return exp_avg_bar, exp_avg

    def _update_post_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any]) -> torch.Tensor:
        """更新暖身後學習率遮罩（8-bit 版本）"""
        # 反量化 lr_mask
        if state['lr_mask_quantized'] is not None:
            lr_mask = self._dequantize_tensor(state['lr_mask_quantized'], state['lr_mask_scale'])
        else:
            lr_mask = state['lr_mask']

        # 極性變化檢測
        current_polarity = state['polarity']
        last_polarity = state['last_polarity']
        polarity_change = (current_polarity != last_polarity).float()

        # 更新學習率遮罩
        lr_mask = lr_mask * (1 - 0.1 * polarity_change) + 0.1 * polarity_change
        lr_mask = torch.clamp(lr_mask, min=group.get("min_lr", 1e-7) / group["lr"], max=1.0)

        # 更新極性歷史
        state['last_polarity'] = current_polarity.clone()

        # 重新量化並儲存
        quantized, scale_tensor = self._quantize_tensor(lr_mask)
        state['lr_mask_quantized'] = quantized
        state['lr_mask_scale'] = scale_tensor
        state['lr_mask'] = lr_mask

        return lr_mask

    def _update_warmup_lr_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """更新暖身期學習率遮罩（8-bit 版本）"""
        # 反量化 lr_mask
        if state['lr_mask_quantized'] is not None:
            lr_mask = self._dequantize_tensor(state['lr_mask_quantized'], state['lr_mask_scale'])
        else:
            lr_mask = state['lr_mask']

        # 梯度方向變化
        grad_sign = grad.sign()
        state['polarity'] = (grad_sign > 0)

        if 'last_grad_sign' in state:
            sign_change = (grad_sign != state['last_grad_sign']).float()
            lr_mask = lr_mask * (1 - 0.05 * sign_change)

        state['last_grad_sign'] = grad_sign
        lr_mask = torch.clamp(lr_mask, min=group.get("min_lr", 1e-7) / group["lr"], max=1.0)

        # 重新量化並儲存
        quantized, scale_tensor = self._quantize_tensor(lr_mask)
        state['lr_mask_quantized'] = quantized
        state['lr_mask_scale'] = scale_tensor
        state['lr_mask'] = lr_mask

        return lr_mask

    def _update_momentum(self, state: Dict[str, Any], group: Dict[str, Any], scaled_grad: torch.Tensor, beta1: float, eps1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新動量（8-bit 版本）"""
        if state["step"] < group.get("warmup_steps", 500) / 2:
            return self._update_torque_aware_momentum(state, scaled_grad, eps1)
        else:
            return self._update_consistency_momentum(state, group, scaled_grad, beta1)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """執行一步優化，整合 C-Optim 功能和 8-bit 量化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 更新全域上下文
        if loss is not None:
            avg_grad_norm = 0
            param_count = 0
            total_grad_norm = 0

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        avg_grad_norm += grad_norm
                        total_grad_norm += grad_norm ** 2
                        param_count += 1

            if param_count > 0:
                avg_grad_norm /= param_count
                rms_grad_norm = (total_grad_norm / param_count) ** 0.5

                # 使用 RMS 梯度範數提供更穩定的上下文信息
                self.c_optim.update_context(
                    torch.tensor(rms_grad_norm),
                    loss.item(),
                    group.get("lr", self.config.lr)
                )

        for group in self.param_groups:
            # 計算全域上下文乘數
            global_lr_multiplier = self.c_optim.compute_contextual_lr_multiplier()
            is_edge_case = self.c_optim.detect_edge_case()

            # AGR 正則化
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))
            if len(grads_this_group) == 0:
                continue

            all_group_grads = torch.cat(grads_this_group)
            abs_all_group_grads = torch.abs(all_group_grads)
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads)

            if self._step < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                # AGR 正則化
                abs_grad = torch.abs(p.grad)
                alpha = abs_grad / sum_abs_all_group_grads
                grad = p.grad * (1 - alpha)

                # 初始化狀態
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                self._step = state["step"]

                # 更新多尺度動量
                self._update_multiscale_momentum(state, grad)

                # 邊緣情況處理 - 改進版
                if is_edge_case:
                    state['edge_case_count'] += 1
                    # 根據邊緣情況的嚴重程度動態調整
                    grad_consistency = self.c_optim.compute_gradient_consistency()
                    loss_stability = self.c_optim.compute_loss_stability()

                    # 組合穩定性分數
                    stability_score = (grad_consistency + loss_stability) / 2

                    # 動態邊緣因子：穩定性越高，懲罰越小
                    edge_factor = 0.4 + 0.4 * stability_score  # 0.4-0.8
                else:
                    edge_factor = 1.0
                    # 重置邊緣情況計數
                    state['edge_case_count'] = max(0, state['edge_case_count'] - 1)

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2, eps3 = group["eps"]

                # 反量化狀態張量
                if state['exp_avg_quantized'] is not None:
                    exp_avg = self._dequantize_tensor(state['exp_avg_quantized'], state['exp_avg_scale'])
                else:
                    exp_avg = state['exp_avg']

                if state['exp_avg_res_quantized'] is not None:
                    exp_avg_res = self._dequantize_tensor(state['exp_avg_res_quantized'], state['exp_avg_res_scale'])
                else:
                    exp_avg_res = state['exp_avg_res']

                # CAME 或 AdaBelief 梯度縮放
                if group.get("came", True):
                    exp_avg_sq = self._dequantize_tensor(state['exp_avg_sq_q'], state['exp_avg_sq_q_scale'])
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2) + eps1, alpha=1 - beta2)
                    scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
                    scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))

                    # 重新量化 exp_avg_sq
                    quantized, scale_tensor = self._quantize_tensor(exp_avg_sq)
                    state['exp_avg_sq_quantized'] = quantized
                    state['exp_avg_sq_scale'] = scale_tensor
                    state['exp_avg_sq'] = exp_avg_sq
                else:
                    clip = state['step'] ** 0.25
                    scaled_grad = grad.clamp(-clip, clip)

                # 動量更新（整合多尺度和量化）
                contextual_update = self._get_contextual_update(state, scaled_grad)
                exp_avg_bar, exp_avg = self._update_momentum(state, group, contextual_update, beta1, eps1)

                # AdaBelief 變異數更新
                res = (scaled_grad - exp_avg_bar).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update_p = exp_avg.clone().mul_(exp_avg_res.rsqrt() + eps2)

                # 重新量化 exp_avg_res
                quantized, scale_tensor = self._quantize_tensor(exp_avg_res)
                state['exp_avg_res_quantized'] = quantized
                state['exp_avg_res_scale'] = scale_tensor
                state['exp_avg_res'] = exp_avg_res

                # Grams 更新
                grams_update = update_p.abs() * grad.sign()
                alpha = 1.0 * group.get('beta1_decay', 0.9995) ** state["step"]
                update_p = alpha * grams_update + (1 - alpha) * update_p

                # 正交梯度（早期暖身）
                if state["step"] < group.get("warmup_steps", 500) / 2:
                    update_p = self.orthograd_(p, update_p)

                # 應用上下文感知的學習率調整
                new_lr = new_lr * global_lr_multiplier * edge_factor
                state['contextual_lr_multiplier'] = global_lr_multiplier * edge_factor

                # 學習率遮罩更新
                new_lr = self._update_learning_rate_mask(state, group, grad)

                # 行縮放（ALLoRA）
                if "row_scaling" in state:
                    row_scaling = self._dequantize_tensor(state['row_scaling_q'], state['row_scaling_q_scale'])
                    new_lr = new_lr * row_scaling

                # 自適應權重衰減
                if state["step"] < group.get("warmup_steps", 500) / 2 and group["weight_decay"] > 0:
                    param_abs_grad = abs_grad.mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - new_lr * group["weight_decay"] * theta)

                # 應用更新
                update_p = update_p.mul(new_lr)
                p.add_(-update_p)

        if self.config.verbose:
            print(f"Lr: {[group['lr'] for group in self.param_groups]}")
            print(f"Contextual Lr Multiplier: {global_lr_multiplier:.4f}")
            print(f"Edge Case: {is_edge_case}")

        return loss

    def _update_learning_rate_mask(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
        """更新學習率遮罩（8-bit 版本）"""
        if state["step"] < group.get("warmup_steps", 500):
            return self._update_warmup_lr_mask(state, group, grad)
        else:
            return self._update_post_warmup_lr_mask(state, group)

    def state_dict(self) -> Dict[str, Any]:
        """獲取優化器狀態字典（包含量化狀態）"""
        state = super().state_dict()
        state['magic_version'] = 2  # 8-bit C-Optim 版本
        state['c_optim_state'] = {
            'context_window': self.c_optim.context_window,
            'edge_threshold': self.c_optim.edge_threshold,
            'adaptation_rate': self.c_optim.adaptation_rate,
            'gradient_history': self.c_optim.gradient_history,
            'loss_history': self.c_optim.loss_history,
            'lr_history': self.c_optim.lr_history
        }
        state['momentum_scales'] = self.momentum_scales
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """載入優化器狀態字典（包含量化狀態）"""
        if 'magic_version' not in state_dict or state_dict['magic_version'] != 2:
            print('[WARNING] Unexpected state dict, some dynamic mask parameters may not be properly synchronized!')

        # 載入 C-Optim 狀態
        if 'c_optim_state' in state_dict:
            c_state = state_dict['c_optim_state']
            self.c_optim.gradient_history = c_state.get('gradient_history', [])
            self.c_optim.loss_history = c_state.get('loss_history', [])
            self.c_optim.lr_history = c_state.get('lr_history', [])

        # 載入動量尺度
        if 'momentum_scales' in state_dict:
            self.momentum_scales = state_dict['momentum_scales']

        super().load_state_dict(state_dict)
