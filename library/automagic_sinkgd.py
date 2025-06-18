import torch
import torch.optim as optim
from typing import Optional, Callable, Tuple
import torch.nn.functional as F
from torch.nn.functional import normalize

@torch.jit.script
def normalize_iteration(X, sqrt_n: float, sqrt_m: float, eps: float):
    row_norm = torch.linalg.vector_norm(X, dim=1, keepdim=True) + eps
    X = X * (sqrt_n / row_norm)
    col_norm = torch.linalg.vector_norm(X, dim=0, keepdim=True) + eps
    X = X * (sqrt_m / col_norm)
    return X

@torch.jit.script
def automagic_lr_adjust(
    grad: torch.Tensor,
    last_polarity: torch.Tensor,
    lr_bump_pos: float,
    lr_bump_neg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_polarity = grad > 0
    polarity_match = last_polarity == current_polarity

    # 建立 dummy tensor，確保型態正確
    dummy = torch.tensor(0.0, device=grad.device, dtype=grad.dtype)
    row_adj, col_adj, avg_adj = dummy, dummy, dummy
    if grad.dim() == 2:
        row_dim = 1
        row_num_pos = polarity_match.sum(dim=row_dim, keepdim=True).to(dtype=grad.dtype)
        row_num_neg = (~polarity_match).sum(dim=row_dim, keepdim=True).to(dtype=grad.dtype)
        row_total = torch.tensor(polarity_match.shape[row_dim], dtype=grad.dtype)
        row_adj = (row_num_pos * lr_bump_pos - row_num_neg * lr_bump_neg) / row_total * 0.5

        col_dim = 0
        col_num_pos = polarity_match.sum(dim=col_dim, keepdim=True).to(dtype=grad.dtype)
        col_num_neg = (~polarity_match).sum(dim=col_dim, keepdim=True).to(dtype=grad.dtype)
        col_total = torch.tensor(polarity_match.shape[col_dim], dtype=grad.dtype)
        col_adj = (col_num_pos * lr_bump_pos - col_num_neg * lr_bump_neg) / col_total * 0.5
    else:
        num_pos = polarity_match.sum().to(dtype=grad.dtype)
        num_neg = (polarity_match.numel() - num_pos).to(dtype=grad.dtype)
        avg_adj = (num_pos * lr_bump_pos - num_neg * lr_bump_neg) / float(polarity_match.numel())

    return row_adj, col_adj, avg_adj, current_polarity

@torch.jit.script
def _ratio(delta_new, delta_p):
    curr_norm, prev_norm = torch.norm(delta_new), torch.norm(delta_p)
    ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
    return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

# Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
@torch.jit.script
def orthograd_(param: torch.Tensor,
                  grad: torch.Tensor,
                  eps: float = 1e-30) -> torch.Tensor:
    """
    JIT 版 Orthogonal Gradient 修正
    Args:
        param: 權重張量 (與 grad 同形狀)
        grad : 梯度張量
        eps  : 穩定常數
    Returns:
        與 grad 同形狀的修正梯度
    """
    # 扁平化計算投影
    w = param.view(-1)
    g = grad.view(-1)
    g_norm = torch.norm(g, 2)

    proj = torch.dot(w, g) / (torch.dot(w, w) + eps)
    g_orth = g - proj * w

    scale = g_norm / (torch.norm(g_orth, 2) + eps)
    g_orth_scaled = g_orth * scale

    return g_orth_scaled.view_as(grad)

class Automagic_Sinkgd(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        lr_bump: float = 1e-5,
        eta: float = 2,
        beta1: float = 0.95,
        d_coef: float = 2,
        weight_decay: float = 5e-4,
        warmup_steps: int = 500,
        full_finetune: bool = False,
        orthograd: bool = False
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.full_finetune = full_finetune
        defaults = dict(
            lr=lr,
            avg_lr_max=lr,
            eta=eta,
            beta1=beta1,
            d_coef=d_coef,
            warmup_steps=warmup_steps,
            full_finetune = full_finetune,
            weight_decay=weight_decay,
            orthograd=orthograd
        )
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

    def _init_state(self, p, group=None):
        device, shape = p.device, p.shape
        state = self.state[p]
        state.setdefault("step", 0)
        state.setdefault("avg_lr_max", float(self.lr))
        state.setdefault("lr_max", float(self.lr))
        # lr_mask
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
        state.setdefault("avg_lr", torch.tensor(self.lr, device=device, dtype=torch.float16))
        state.setdefault("exp_avg", torch.zeros_like(p))
        if len(p.shape) == 2:
            state['row_lr_mask'] = torch.ones(shape[0], 1, device=device, dtype=torch.float16) * (self.lr / 2)
            state['col_lr_mask'] = torch.ones(1, shape[1], device=device, dtype=torch.float16) * (self.lr / 2)

        if group['full_finetune'] == False:
            state.setdefault("pre", None)
            # ==== ALLoRA ====
            #ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            #https://arxiv.org/abs/2410.09692
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))
        else:
            if group['d_coef'] != 1:
                pre_init = p.clone()
                state.setdefault("pre", pre_init)

    def sinkgd_preprocess(self, grad, num_iter=3):
        m, n = grad.shape
        sqrt_n = n ** 0.5
        sqrt_m = m ** 0.5
        X = grad
        eps = 1e-30
        for _ in range(num_iter):
            X = normalize_iteration(X, sqrt_n, sqrt_m, eps)
        return X

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            warmup_steps = group['warmup_steps']
            if self._step < self.warmup_steps / 2 and self.weight_decay > 0:
                grads_this_group = []
                for p in group["params"]:
                    if p.grad is not None:
                        grads_this_group.append(p.grad.view(-1))
                if len(grads_this_group) == 0:
                    continue
                all_group_grads = torch.cat(grads_this_group)
                abs_all_group_grads = torch.abs(all_group_grads)

                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False) + 1e-12

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                state['step'] += 1
                step = state['step']
                self._step = state["step"] + 1
                grad = p.grad.data
                beta1 = group['beta1']
                exp_avg = state['exp_avg']
                if state['step'] == 1:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    continue

                update = exp_avg.abs() * grad.sign()
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if grad.ndim == 2:
                    # === 正交梯度 ===
                    #Grokking at the Edge of Numerical Stability

                    #https://arxiv.org/abs/2501.04697
                    #https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev
                    if group["orthograd"] and state["step"] > group["warmup_steps"] / 2:
                        grad = orthograd_(p.data, update)
                    update = self.sinkgd_preprocess(update)

                condition = 0.0
                if group['d_coef'] != 1 and (state["step"] < group["warmup_steps"] / 2):
                    delta_p = p - state["pre"] if state["pre"] else p
                    pre = state["pre"] if state["pre"] else torch.zeros_like(p)
                    condition = -torch.sum(p.grad * delta_p)
                else:
                    if 'pre' in state:
                        del state["pre"]

                lr_decay = 1.0
                if state["step"] > group["warmup_steps"]:
                    if group["lr"] > state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    if group["lr"] < state["lr_max"]:
                        lr_decay = max((group["lr"] / state["lr_max"]), 0.1)

                if state["step"] < group["warmup_steps"]:
                    lr_bump = self.lr_bump
                    min_lr = max(self.min_lr, state['avg_lr_max'] / 10)
                    max_lr = self.max_lr
                    if state["step"] < group["warmup_steps"] / 2:
                        lr_bump_pos = self.lr_bump * group['d_coef'] if condition > 0.0 else self.lr_bump
                        lr_bump_neg = self.lr_bump * group['d_coef'] if condition < 0.0 else self.lr_bump
                    else:
                        lr_bump_pos = lr_bump_neg = self.lr_bump

                    row_adj, col_adj, avg_adj, current_polarity= automagic_lr_adjust(grad, state["last_polarity"], lr_bump_pos, lr_bump_neg)

                    if grad.ndim == 2:
                        half_min, half_max = min_lr * 0.5, max_lr * 0.5
                        state['row_lr_mask'].add_(row_adj).clamp_(half_min, half_max)
                        state['col_lr_mask'].add_(col_adj).clamp_(half_min, half_max)
                        new_lr = state['row_lr_mask'] + state['col_lr_mask']
                        state['avg_lr'] = new_lr.mean().item()
                    else:
                        state['avg_lr'].add_(avg_adj).clamp_(min=min_lr, max=max_lr)
                        new_lr = state['avg_lr']

                    state['avg_lr_max'] = max(float(state['avg_lr']), state['avg_lr_max'])
                    state['last_polarity'] = current_polarity
                else:
                    state.pop('last_polarity', None)
                    avg_lr = float(state['avg_lr'])
                    decay_rate = avg_lr * ((avg_lr / state['avg_lr_max']) / group["warmup_steps"])

                    if grad.ndim == 2:
                        state['row_lr_mask'].sub_(decay_rate)
                        state['col_lr_mask'].sub_(decay_rate)
                        new_lr = state['row_lr_mask'] + state['col_lr_mask']
                    else:
                        if "new_avg_lr" not in state:
                            state['new_avg_lr'] = state['avg_lr']
                        state['new_avg_lr'] = state['new_avg_lr'] - decay_rate
                        new_lr = state['new_avg_lr']

                    new_lr = torch.clamp(new_lr, min=state['avg_lr_max'] / 10, max=state['avg_lr_max'] * lr_decay)

                allora =  state["row_scaling"] if "row_scaling" in state else 1
                new_lr = new_lr * allora

                # Weight decay applied to y
                if group['weight_decay'] != 0 and state["step"] < group["warmup_steps"] / 2:
                    #Adaptive Weight Decay for Deep Neural Networks
                    #https://arxiv.org/abs/1907.08931
                    param_abs_grad = torch.abs(p.grad).mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - new_lr * group["weight_decay"] * theta)

                update.mul_(new_lr)
                p.add_(-update)

        return loss
