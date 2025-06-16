import torch
import torch.optim as optim
from typing import Optional, Callable
import torch.nn.functional as F
from torch.nn.functional import normalize

class Automagic_Sinkgd(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        lr_bump: float = 1e-5,
        eps: float = 1e-8,
        beta1: float = 0.9,
        eta: float = 2,
        d_coef: float = 2,
        weight_decay: float = 5e-4,
        warmup_steps: int = 500,
        full_finetune: bool = False,
        orthograd: bool = False,
        r=0.0,
        weight_lr_power=2.0
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
            d_coef=d_coef,
            warmup_steps=warmup_steps,
            full_finetune = full_finetune,
            eps=eps,
            beta1=beta1,
            weight_decay=weight_decay,
            r=r,
            train_mode=False,
            orthograd=orthograd,
            weight_lr_power=weight_lr_power
        )
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

    def _init_state(self, p, group=None):
        device, shape = p.device, p.shape
        state = self.state[p]
        state.setdefault("step", 0)
        state.setdefault("weight_sum", 0.0)
        state.setdefault("avg_lr_max", float(self.lr))
        state.setdefault("lr_max", float(self.lr))
        # lr_mask
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
        state.setdefault("exp_avg", torch.zeros_like(p))
        if len(p.shape) == 2:
            state['row_lr_mask'] = torch.ones(shape[0], 1, device=device, dtype=torch.float16) * (self.lr / 2)
            state['col_lr_mask'] = torch.ones(1, shape[1], device=device, dtype=torch.float16) * (self.lr / 2)
        else:
            state.setdefault("avg_lr", torch.tensor(self.lr, device=device, dtype=torch.float16))
            if len(p.shape) == 1:
                state.setdefault("lr_mask", torch.zeros_like(p))
        state['last_grad'] = torch.zeros_like(p)

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

    @torch.no_grad()
    def eval(self):
        """Switch to evaluation mode - use averaged parameters"""
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta1']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x (evaluation parameters)
                        p.lerp_(end=state['z'].to(p.device), weight=1-1/beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """Switch to training mode - use training parameters"""
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta1']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y (training parameters)
                        p.lerp_(end=state['z'].to(p.device), weight=1-beta1)
                group['train_mode'] = True

    def sinkgd_preprocess(self, grad, num_iter=5):
        """
        SinkGD 預處理：交替行、列 L2 正規化
        grad: torch.Tensor, 形狀(m, n)
        num_iter: 交替正規化次數
        """
        m, n = grad.shape
        X = grad.clone()
        eps = 1e-30
        for _ in range(num_iter):
            # 行 L2-normalize to sqrt(n)
            row_norm = X.norm(dim=1, keepdim=True) + eps
            X = X / row_norm * (n ** 0.5)
            # 列 L2-normalize to sqrt(m)
            col_norm = X.norm(dim=0, keepdim=True) + eps
            X = X / col_norm * (m ** 0.5)
        return X

    # Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
    def orthograd_(self, p, grad):
        G_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)
        g_norm = g.norm(2)

        proj = torch.dot(w, g) / torch.dot(w, w).add(1e-30)
        g_orth = g.sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g_norm / g_orth.norm(2).add(1e-30))

        return g_orth_scaled.view(G_shape)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step"""

        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                          "Please call .train() before training and .eval() before evaluation.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            warmup_steps = group['warmup_steps']
            r = group['r']
            weight_lr_power = group['weight_lr_power']

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

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    self._init_state(p, group)
                state['step'] += 1
                step = state['step']
                self._step = state["step"] + 1
                beta1 = group['beta1']
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                if grad.ndim == 2:
                    # === 正交梯度 ===
                    #Grokking at the Edge of Numerical Stability

                    #https://arxiv.org/abs/2501.04697
                    #https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev
                    if group["orthograd"] and state["step"] > group["warmup_steps"] / 2:
                        grad = self.orthograd_(p, grad)
                    update = self.sinkgd_preprocess(grad + exp_avg)
                else:
                    update = exp_avg.abs() * grad.sign()

                condition = 0.0
                if group['d_coef'] != 1 and (state["step"] < group["warmup_steps"] / 2):
                    delta_p = p - state["pre"] if state["pre"] else p
                    pre = state["pre"] if state["pre"] else torch.zeros_like(p)
                    condition = -torch.sum(p.grad * delta_p)
                else:
                    if 'pre' in state:
                        del state["pre"]
                lr_decay = 1.0
                if state["step"] < group["warmup_steps"] or grad.ndim == 1:
                    lr_bump = self.lr_bump
                    min_lr = self.min_lr
                    max_lr = self.max_lr
                    current_polarity = grad > 0
                    last_polarity = state['last_polarity']
                    if state["step"] < group["warmup_steps"] / 2:
                        lr_bump_pos = self.lr_bump * group['d_coef'] if condition > 0.0 else self.lr_bump
                        lr_bump_neg = self.lr_bump * group['d_coef'] if condition < 0.0 else self.lr_bump
                    else:
                        lr_bump_pos = lr_bump_neg = self.lr_bump
                    polarity_match = last_polarity == current_polarity
                    lr_adjustment = torch.where(polarity_match, lr_bump_pos, -lr_bump_neg)

                    if grad.ndim == 2:
                        half_min, half_max = min_lr * 0.5, max_lr * 0.5
                        row_adj = lr_adjustment.mean(dim=1, keepdim=True) * 0.5
                        state['row_lr_mask'].add_(row_adj).clamp_(half_min, half_max)
                        col_adj = lr_adjustment.mean(dim=0, keepdim=True) * 0.5
                        state['col_lr_mask'].add_(col_adj).clamp_(half_min, half_max)
                        new_lr = state['row_lr_mask'] + state['col_lr_mask']
                    else:
                        state['avg_lr'].add_(lr_adjustment.mean()).clamp_(min=min_lr, max=max_lr)
                        new_lr = state['avg_lr']
                        if grad.ndim == 1:
                            state['lr_mask'].add_(lr_adjustment).clamp_(min=min_lr, max=max_lr)
                            if state["step"] > group["warmup_steps"]:
                                new_lr = torch.minimum(state['avg_lr'], state['lr_mask']).clamp(max=state['avg_lr_max'])
                                if group["lr"] > state["lr_max"]:
                                    state["lr_max"] = group["lr"]
                                if group["lr"] < state["lr_max"]:
                                    lr_decay = max((group["lr"] / state["lr_max"]), 0.1)
                    lr_mean = new_lr.mean().item()
                    state['avg_lr_max'] = max(lr_mean, state['avg_lr_max'])
                    state['last_polarity'] = current_polarity
                else:
                    state.pop('last_polarity', None)

                    if grad.ndim == 2:
                        new_lr = state['row_lr_mask'] + state['col_lr_mask']
                    else:
                        new_lr = state['avg_lr']
                    if group["lr"] > state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    if group["lr"] < state["lr_max"]:
                        lr_decay = max((group["lr"] / state["lr_max"]), 0.1)

                    avg_lr_max = state['avg_lr_max']

                allora =  state["row_scaling"] if "row_scaling" in state else 1

                # ==== VRAdam ====
                #A Physics-Inspired Optimizer: Velocity Regularized Adam
                #https://arxiv.org/abs/2505.13196
                vr = 1 / (1+ min(3 * (exp_avg ** 2).sum(),10))

                lr_tweak = lr_decay * allora * vr
                new_lr = new_lr * lr_tweak

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
