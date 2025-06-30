import torch
import torch.optim as optim
from typing import Optional, Callable

class Automagic_Splus(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        beta1=0.9,
        beta2=0.999,
        eps=1e-30,
        inverse_every=100,
        nonstandard_constant=0.001,
        lr: float = 1e-5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        lr_bump: float = 1e-5,
        eta: float = 2,
        d_coef: float = 2,
        weight_decay=5e-4,
        weight_decay2=1.0,
        max_dim=10000,
        warmup_steps: int = 500,
        full_finetune: bool = False,):

        defaults = dict(
            lr=lr,
            lr_max=lr,
            eta=eta,
            d_coef=d_coef,
            warmup_steps=warmup_steps,
            full_finetune = full_finetune,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            inverse_every=inverse_every,
            nonstandard_constant=nonstandard_constant,
            weight_decay=weight_decay,
            weight_decay2=weight_decay2,
            max_dim=max_dim
        )
        super(Automagic_Splus, self).__init__(params, defaults)
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.full_finetune = full_finetune
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

    def sinkgd_preprocess(self, grad, num_iter=3):
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

    def _init_state(self, p, group=None):
        device, shape = p.device, p.shape
        state = self.state[p]
        state.setdefault("step", 0)
        state['exp_avg'] = torch.zeros_like(p.data)
        # lr_mask
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
        state.setdefault("lr_mask", torch.ones(shape, device=device, dtype=torch.float16) * self.lr)
        state.setdefault("avg_lr", self.lr)
        state.setdefault("avg_lr_max", self.lr)
        state.setdefault("lr_max", self.lr)

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
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step"""

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
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads) + 1e-12

            if self._step < self.warmup_steps / 2 and self.weight_decay > 0:
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False) + 1e-12

            warmup_steps = group['warmup_steps']
            is_early_warmup = self._step < warmup_steps / 2
            is_post_warmup = self._step > warmup_steps
            use_weight_decay = is_early_warmup and self.weight_decay > 0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    self._init_state(p, group)
                    if len(grad.shape) == 2:
                        m, n = grad.shape
                        if m < group['max_dim'] and n < group['max_dim']:
                            state['L'] = torch.zeros(m, m, dtype=p.dtype, device=p.device)
                            state['R'] = torch.zeros(n, n, dtype=p.dtype, device=p.device)
                            state['Q_L'] = torch.eye(m, dtype=p.dtype, device=p.device)
                            state['Q_R'] = torch.eye(n, dtype=p.dtype, device=p.device)

                state['step'] += 1
                step = state['step']
                self._step = state["step"]

                # Update exp_avg
                state['exp_avg'] = group['beta1'] * state['exp_avg'] + (1 - group['beta1']) * grad
                exp_avg = state['exp_avg']
                # Compute update direction
                if len(p.shape) == 2 and 'L' in state:
                    exp_avg = self.sinkgd_preprocess(exp_avg)
                    m, n = p.shape
                    state['L'] = group['beta2'] * state['L'] + (1 - group['beta2']) * (grad @ grad.t())
                    state['R'] = group['beta2'] * state['R'] + (1 - group['beta2']) * (grad.t() @ grad)

                    if (step % group['inverse_every'] == 0) or (step == 1):
                        L_reg = state['L'] + group['eps'] * torch.eye(m, dtype=p.dtype, device=p.device)
                        R_reg = state['R'] + group['eps'] * torch.eye(n, dtype=p.dtype, device=p.device)
                        _, state['Q_L'] = torch.linalg.eigh(L_reg)
                        _, state['Q_R'] = torch.linalg.eigh(R_reg)

                    Q_L = state['Q_L']
                    Q_R = state['Q_R']
                    exp_avg_rot = Q_L.t() @ exp_avg @ Q_R
                    exp_avg_sign = exp_avg_rot.sign()
                    update = Q_L @ exp_avg_sign @ Q_R.t()
                else:
                    update = exp_avg.abs().mul_(grad.sign())

                condition = 0.0
                if group['d_coef'] != 1 and is_early_warmup:
                    delta_p = p - state["pre"] if state["pre"] is not None else p
                    condition = -torch.sum(p.grad * delta_p)
                else:
                    if 'pre' in state:
                        del state["pre"]

                allora = state.get("row_scaling", torch.tensor(1.0))
                lr_decay = 1.0
                if is_early_warmup:
                    last_polarity = state['last_polarity']
                    lr_mask = state['lr_mask']
                    lr_bump, d_coef= self.lr_bump, group["d_coef"]

                    current_polarity = grad > 0
                    same = (last_polarity == current_polarity).to(torch.float16)
                    state['last_polarity'] = current_polarity
                    if is_early_warmup:
                        if condition >= 0.0:
                            lr_adjustment = (d_coef * same - (1 - same)) * lr_bump
                        elif condition < 0.0:
                            lr_adjustment = (same - d_coef * (1 - same)) * lr_bump
                    else:
                        lr_adjustment = (same * 2 - 1) * lr_bump
                    lr_mask.add_(lr_adjustment).clamp_(min=self.min_lr, max=self.max_lr)
                    state['avg_lr'] = state['lr_mask'].mean().item()
                    if self._step % 25 == 0:
                        lr_mask_f = lr_mask.float()
                        lr_medians = torch.quantile(
                            lr_mask_f, torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1], device=lr_mask.device)
                        )
                        diff = torch.stack([torch.abs(lr_mask_f - m) for m in lr_medians], dim=-1)
                        nearest_idx = torch.argmin(diff, dim=-1)
                        lr_mask_flat = lr_mask.flatten()
                        nearest_idx_flat = nearest_idx.flatten()
                        lr_mask_flat = lr_medians[nearest_idx_flat]
                        state['lr_mask'] = lr_mask_flat.view_as(lr_mask).to(torch.float16)
                        state['avg_lr_max'] = max(state['avg_lr'], state['avg_lr_max'])
                    new_lr = state['lr_mask']
                    new_lr = new_lr * allora
                else:
                    if group["lr"] > state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    lr_decay = max(group["lr"] / state["lr_max"], 0.1)
                    if "last_polarity" in state:
                        del state['last_polarity']
                        lr_mask = state['lr_mask']
                        lr_mask.mul_(allora)
                        lr_mask_f = lr_mask.float()
                        lr_medians = torch.quantile(
                            lr_mask_f, torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1], device=lr_mask.device)
                        )
                        diff = torch.stack([torch.abs(lr_mask_f - m) for m in lr_medians], dim=-1)
                        nearest_idx = torch.argmin(diff, dim=-1)
                        lr_mask_flat = lr_mask.flatten()
                        nearest_idx_flat = nearest_idx.flatten()
                        lr_mask_flat = lr_medians[nearest_idx_flat]
                        state['lr_mask'] = lr_mask_flat.view_as(lr_mask).to(torch.float16)
                        state['avg_lr'] = state['lr_mask'].mean().item()
                        state['avg_lr_max'] = max(state['avg_lr'], state['avg_lr_max'])
                    new_lr = state['lr_mask'] * lr_decay

                update = update.mul_(new_lr)
                p.add_(-update)

                # 權重衰減處理
                if use_weight_decay:
                    param_abs_grad = torch.abs(p.grad).mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    if condition < 0.0:
                        weight_decay = state['avg_lr'] * allora.mean().item() * group["weight_decay2"] * theta
                    else:
                        weight_decay = state['avg_lr'] * allora.mean().item() * group["weight_decay"] * theta
                    p.data.mul_(1 - weight_decay)

        return loss