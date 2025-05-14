import torch
from typing import List, Dict, Any, Optional, Tuple
import bitsandbytes.functional as F
from torch.nn.functional import normalize
from dataclasses import dataclass

@dataclass
class OptimizerConfig:
    """Configuration for the Automagic_CameAMP optimizer."""
    lr: float = 1e-6
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    lr_bump: float = 3e-6
    eps: Tuple[float, float] = (1e-30, 1e-16)
    clip_threshold: float = 1.0
    betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999)
    weight_decay: float = 2.5
    warmup_steps: int = 500
    la_layers: int = 3
    alphas: Tuple[float, ...] = (0.6, 0.75, 0.85, 0.85)
    ks: Tuple[int, ...] = (5, 5, 3, 3)
    full_finetune: bool = False
    verbose: bool = False

class BaseOptimizer(torch.optim.Optimizer):
    """Base class for Automagic optimizers with common functionality."""

    def __init__(self, params, config: OptimizerConfig):
        self.config = config
        defaults = dict(
            lr=config.lr,
            eps=config.eps,
            clip_threshold=config.clip_threshold,
            betas=config.betas,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
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

    def _get_group_lr(self, group: Dict[str, Any]) -> float:
        """Get the average learning rate for a parameter group."""
        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])
        return float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.config.lr

    def _ratio(self, new_p: torch.Tensor, p: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Calculate the ratio for selective projection decay."""
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(p - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """Initialize optimizer state for a parameter."""
        device = p.device
        shape = p.shape
        state = self.state[p]

        # Basic state initialization
        state.setdefault("step", 0)
        state.setdefault("step2", 0)

        # Learning rate mask initialization
        state.setdefault('lr_mask', torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr)
        state.setdefault('avg_lr', float(self.config.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # Momentum and variance initialization
        state.setdefault("exp_avg", torch.zeros_like(p))
        state.setdefault("s", torch.zeros_like(p))
        state.setdefault("exp_avg_sq", torch.zeros_like(p))
        state.setdefault("exp_avg_res", torch.zeros_like(p))

        # Lookahead initialization
        for i in range(1, self.config.la_layers + 1):
            state.setdefault(f"slow{i}", p.data.clone())
        for i in range(2, self.config.la_layers + 1):
            state.setdefault(f"step{i}", 0)

        # Full finetune initialization
        if group is not None and group.get('full_finetune', False):
            state.setdefault("pre", p.clone())
        else:
            state.setdefault("pre", None)

class Automagic_CameAMP(BaseOptimizer):
    """Automagic_CameAMP optimizer implementation."""

    def __init__(self, params, **kwargs):
        config = OptimizerConfig(**kwargs)
        super().__init__(params, config)

    def _init_Lookahead_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Lookahead state for a parameter."""
        state = self.state[p]
        for i in range(1, self.config.la_layers + 1):
            state[f"slow{i}"] = p.data.clone()
        for i in range(2, self.config.la_layers + 1):
            state[f"step{i}"] = 0

    def _del_Lookahead_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """Delete Lookahead state for a parameter."""
        state = self.state[p]
        for i in range(1, self.la_layers + 1):
            if i > keep_layers:
                if f"slow{i}" in state:
                    del state[f"slow{i}"]
                if f"step{i}" in state:
                    del state[f"step{i}"]

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            grads_this_group = []
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                grads_this_group.append(p.grad.view(-1))
            all_group_grads = torch.cat(grads_this_group)
            sum_abs_all_group_grads = torch.sum(torch.abs(all_group_grads))

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                """
                ==== AGR自適應梯度正則 ====
                Adaptive Gradient Regularization: A Faster and Generalizable Optimization Technique for Deep Neural Networks
                https://arxiv.org/pdf/2407.16944
                """
                abs_grad = torch.abs(grad)
                alpha = abs_grad / sum_abs_all_group_grads
                grad = grad * (1 - alpha)

                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    self._init_state(p, group)

                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1
                if state["step"] == group["warmup_steps"]:
                    if 's' in state:
                        del state['s']
                    if 'last_polarity' in state:
                        del state['last_polarity']
                    if 'pre' in state and state["pre"] is not None:
                        del state['pre']

                if state["step"] == group["warmup_steps"]:
                    self._init_Lookahead_state(p, group)

                if state["step"] == group["warmup_steps"] * 2:
                    self._del_Lookahead_state(p, group)

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2 = group["eps"]

                """
                ==== Adafactor/RMS核心部分 (always non-factored) ====
                CAME: Confidence-guided Adaptive Memory Efficient Optimization
                https://arxiv.org/abs/2307.02047
                https://github.com/yangluo7/CAME
                """
                update = grad.pow(2) + eps1
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).add_(update, alpha=1 - beta2)
                scaled_grad = grad.clone().mul_(exp_avg_sq.rsqrt())
                scaled_grad.div_((self._rms(scaled_grad) / group["clip_threshold"]).clamp_(min=1.0))

                """
                ==== Torque-Aware Momentum ====
                https://arxiv.org/abs/2412.18790
                https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/tam.py
                """
                if state["step"] < group["warmup_steps"] / 2:
                    decay_rate = 0.9
                    s, exp_avg = state['s'], state['exp_avg']
                    corr = normalize(exp_avg, p=2.0, dim=0).mul_(normalize(scaled_grad, p=2.0, dim=0))
                    s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)
                    d = ((1.0 + s) / 2.0).add_(eps1).mul_(scaled_grad)
                    exp_avg.mul_(beta1).add_(d)
                else:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(scaled_grad, alpha=1 - beta1)

                # CAME core
                exp_avg_res = state["exp_avg_res"]
                res = (scaled_grad - exp_avg).pow(2) + eps2
                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)
                update = exp_avg.clone().mul_(exp_avg_res.rsqrt())

                """
                ==== Automagic lrmask ====
                https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py
                """
                if state["step"] < group["warmup_steps"]:
                    last_polarity = state['last_polarity']
                    current_polarity = (grad > 0)
                    sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
                    state['last_polarity'] = current_polarity
                    lr_mask = state['lr_mask']
                    new_lr = torch.where(
                        sign_agree > 0,
                        lr_mask + self.config.lr_bump,
                        lr_mask - self.config.lr_bump
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

                """
                === Grams ===
                Grams: Gradient Descent with Adaptive Momentum Scaling
                https://arxiv.org/abs/2412.17107
                https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/grams.py
                """
                if state["step"] < group["warmup_steps"] / 2:
                    update.abs_().mul_(grad.sign())

                update = update.mul(new_lr)

                """
                === SPD 選擇性投影decay ===
                Rethinking Weight Decay for Robust Fine-Tuning of Foundation Models
                https://arxiv.org/abs/2411.01713
                https://github.com/GT-RIPL/Selective-Projection-Decay/tree/main
                Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias?
                https://arxiv.org/abs/2504.12883
                """
                do_spd = False
                if state["step"] < group["warmup_steps"]:
                    pre = state["pre"] if state["pre"] is not None else torch.zeros_like(p)
                    condition = -torch.sum(grad * (p - pre))
                    if condition < 0.0:
                        do_spd = True
                        new_p = p - update
                        ratio = self._ratio(new_p, p, pre)
                        new_p = new_p - group["weight_decay"] * ratio * (new_p - pre)
                        p.copy_(new_p)
                if not do_spd:
                    p.add_(-update)

                """
                === Lookahead動態同步及釋放多餘層 ===
                Multilayer Lookahead: a Nested Version of Lookahead
                https://arxiv.org/abs/2110.14254
                """
                if state["step"] >= group["warmup_steps"]:
                    N = group["warmup_steps"]
                    lookahead_active_layers = self.config.la_layers
                    if lookahead_active_layers < self.config.la_layers:
                        self._del_Lookahead_state_till(p, lookahead_active_layers, group)
                    if lookahead_active_layers > 0:
                        if f"slow1" in state:
                            if state["step"] % self.config.ks[0] == 0:
                                state["slow1"].add_(p.data - state["slow1"], alpha=self.config.alphas[0])
                                p.data.copy_(state["slow1"])
                                for l in range(2, lookahead_active_layers + 1):
                                    slowk = f"slow{l}"
                                    stepk = f"step{l}"
                                    if slowk in state and stepk in state:
                                        state[stepk] += 1
                                        if state[stepk] % self.ks[l-1] == 0:
                                            state[slowk].add_(
                                                state[f"slow{l-1}"] - state[slowk],
                                                alpha=self.config.alphas[l-1],
                                            )
                                            state[f"slow{l-1}"].copy_(state[slowk])
                                        else:
                                            break

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
        """Initialize 8-bit optimizer state for a parameter."""
        device = p.device
        shape = p.shape
        state = self.state[p]
        state.setdefault("step", 0)

        # Initialize quantized learning rate mask
        lr_mask_init = torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr
        q_lr_mask, q_lr_mask_scale = F.quantize_blockwise(lr_mask_init, blocksize=2048)
        state.setdefault('lr_mask_q', q_lr_mask)
        state.setdefault('lr_mask_q_scale', q_lr_mask_scale)
        state.setdefault('avg_lr', float(self.config.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # Initialize quantized momentum
        exp_avg_fp32 = torch.zeros_like(p)
        q_exp_avg, q_exp_avg_scale = F.quantize_blockwise(exp_avg_fp32, blocksize=2048)
        state.setdefault("exp_avg_q", q_exp_avg)
        state.setdefault("exp_avg_q_scale", q_exp_avg_scale)

        # Initialize variance tracking
        if len(shape) >= 2:
            state["exp_avg_sq_row"] = torch.zeros(shape[:-1], device=device)
            state["exp_avg_sq_col"] = torch.zeros(shape[:-2]+shape[-1:], device=device)
            state["exp_avg_res_row"] = torch.zeros(shape[:-1], device=device)
            state["exp_avg_res_col"] = torch.zeros(shape[:-2]+shape[-1:], device=device)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["exp_avg_res"] = torch.zeros_like(p)

        # Full finetune initialization
        if group is not None and group.get('full_finetune', False):
            state.setdefault("pre", p.clone())
        else:
            state.setdefault("pre", None)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with 8-bit quantization."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                state = self.state[p]
                factored = len(p.shape) >= 2

                # Initialize state if needed
                if len(state) == 0:
                    self._init_state(p, group)

                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                beta1, beta2, beta3 = group["betas"]
                eps1, eps2 = group["eps"]

                # Adafactor/RMS core
                update = grad.pow(2) + eps1
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1 - beta2)
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1 - beta2)
                    update = grad.clone().mul_(exp_avg_sq.rsqrt())
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # Update quantized momentum
                exp_avg = F.dequantize_blockwise(state["exp_avg_q"], state["exp_avg_q_scale"], blocksize=2048)
                exp_avg.mul_(beta1).add_(update, alpha=1-beta1)
                q_exp_avg, q_exp_avg_scale = F.quantize_blockwise(exp_avg, blocksize=2048)
                state["exp_avg_q"] = q_exp_avg
                state["exp_avg_q_scale"] = q_exp_avg_scale

                # CAME
                res = (update - exp_avg).pow(2) + eps2
                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0-beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0-beta3)
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul(exp_avg)
                else:
                    update = exp_avg.clone()

                # Learning rate mask
                if state["step"] <= group["warmup_steps"]:
                    last_polarity = state['last_polarity']
                    current_polarity = (grad > 0)
                    sign_agree = torch.where(last_polarity == current_polarity, 1, -1)
                    state['last_polarity'] = current_polarity
                    lr_mask = F.dequantize_blockwise(state['lr_mask_q'], state['lr_mask_q_scale'], blocksize=2048)
                    new_lr = torch.where(
                        sign_agree > 0,
                        lr_mask + self.config.lr_bump,
                        lr_mask - self.config.lr_bump
                    )
                    new_lr = torch.clamp(new_lr, min=self.config.min_lr, max=self.config.max_lr)
                else:
                    new_lr = F.dequantize_blockwise(state['lr_mask_q'], state['lr_mask_q_scale'], blocksize=2048)

                # Update quantized learning rate mask
                q_lr_mask, q_lr_mask_scale = F.quantize_blockwise(new_lr, blocksize=2048)
                state['lr_mask_q'] = q_lr_mask
                state['lr_mask_q_scale'] = q_lr_mask_scale
                state['avg_lr'] = torch.mean(new_lr).item()

                if group["lr"] < 1e-6:
                    new_lr = new_lr * (group["lr"] / 1e-6)

                update = update.mul(new_lr)

                # SPD
                do_spd = False
                if state["step"] <= group["warmup_steps"]:
                    pre = state["pre"] if state["pre"] is not None else torch.zeros_like(p)
                    condition = -torch.sum(grad * (p - pre))
                    if condition < 0.0:
                        do_spd = True
                        new_p = p - update
                        ratio = self._ratio(new_p, p, pre)
                        new_p = new_p - group["weight_decay"] * ratio * (new_p - pre)
                        p.copy_(new_p)
                if not do_spd:
                    p.add_(-update)

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
                         if kk not in ('lr_mask', 'exp_avg_q', 'exp_avg_q_scale', 'lr_mask_q', 'lr_mask_q_scale')}
            # Save quantized tensors
            if 'exp_avg_q' in v and 'exp_avg_q_scale' in v:
                save_state['exp_avg_q'] = v['exp_avg_q']
                save_state['exp_avg_q_scale'] = v['exp_avg_q_scale']
            if 'lr_mask_q' in v and 'lr_mask_q_scale' in v:
                save_state['lr_mask_q'] = v['lr_mask_q']
                save_state['lr_mask_q_scale'] = v['lr_mask_q_scale']
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
                                   if kk not in ('exp_avg_q', 'exp_avg_q_scale', 'lr_mask_q', 'lr_mask_q_scale')}
        super().load_state_dict(basic_sd)
        # Restore quantized tensors
        param_map = [p for g in self.param_groups for p in g['params']]
        for idx, p in enumerate(param_map):
            idx_str = str(idx)
            if idx_str not in state_dict['state']:
                continue
            src = state_dict['state'][idx_str]
            st = self.state[p]
            if 'exp_avg_q' in src and 'exp_avg_q_scale' in src:
                st['exp_avg_q'] = src['exp_avg_q']
                st['exp_avg_q_scale'] = src['exp_avg_q_scale']
            if 'lr_mask_q' in src and 'lr_mask_q_scale' in src:
                st['lr_mask_q'] = src['lr_mask_q']
                st['lr_mask_q_scale'] = src['lr_mask_q_scale']