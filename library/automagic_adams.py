import torch
import math
from typing import List
import torch.nn.functional as F
from torch.nn.functional import normalize

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
    print("[Automagic_AdamS] 找不到 bitsandbytes，將以 FP16 儲存狀態。")

class Automagic_AdamS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-5,
        eps: float = 1e-8,
        clip_threshold: float = 1.0,
        betas: tuple = (0.5, 0.98, 0.99),
        alpha_decay: float = 0.9995,
        eta: float = 2,
        d_coef: float = 2,
        weight_decay: float = 5e-4,
        weight_decay2: float = 1.0,
        warmup_steps: int = 250,
        full_finetune: bool = False,
        use_8bit: bool = False,
    ):
        self.lr = lr
        self.use_8bit = bool(use_8bit and bnb is not None)
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.full_finetune = full_finetune
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            alpha_decay=alpha_decay,
            eta=eta,
            d_coef=d_coef,
            weight_decay=weight_decay,
            weight_decay2=weight_decay2,
            warmup_steps=warmup_steps,
            full_finetune=full_finetune,
        )
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

    def _q(self, t: torch.Tensor):
        """Quantize tensor to (int8, scale) 兩元組。"""
        if not self.use_8bit:
            return t
        q, s = bnb.functional.quantize_8bit(t)
        return (q, s)

    def _dq(self, q_or_t):
        """還原成 FP16/FP32 張量。"""
        if not self.use_8bit:
            return q_or_t
        q, s = q_or_t
        return bnb.functional.dequantize_8bit(q, s)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    def _get_group_lr(self, group):
        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])
        return float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.lr


    # Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
    def orthograd_(self, p, grad, state):
        w = p.view(-1)
        w_norm = w.norm(2)
        if w_norm < 1e-30:
            return grad
        G_shape = grad.shape
        g = grad.view(-1)
        g_norm = g.norm(2)
        dot_wd = torch.dot(w, g)
        if G_shape[0] * G_shape[1] > 50 ** 2:
            ema_decay = 0.9
            cos_val = dot_wd / (w_norm * g_norm)
            if "cos_sim" not in state or state["cos_sim"] == 0:
                state["cos_sim"] = cos_val.item()
            else:
                state["cos_sim"] = (ema_decay * state["cos_sim"] + (1 - ema_decay) * cos_val.item())

        if state["cos_sim"] < - 0.8 or G_shape[0] * G_shape[1] <= 50 ** 2:
            dot_ww = torch.dot(w, w)
            proj = dot_wd / (dot_ww + 1e-30)
            g_orth = g - w * proj
            g_orth_scaled = g_orth * (g_norm / (g_orth.norm(2) + 1e-30))
            return g_orth_scaled.view(G_shape)
        else:
            return grad

    def _ratio(self, delta_new, delta_p):
        curr_norm, prev_norm = torch.norm(delta_new), torch.norm(delta_p)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def soft_collision_update(self, weight: torch.Tensor,
                             grad: torch.Tensor,
                             coll_coef: float = 0.1) -> torch.Tensor:

        w_norm = F.normalize(weight, dim=1)           # (N, D)
        cos_w = w_norm @ w_norm.t()                   # (N, N)
        cos_w.fill_diagonal_(0.0)
        g_norm = F.normalize(grad, dim=1)
        cos_g = g_norm @ g_norm.t()
        cos_g.fill_diagonal_(0.0)
        coeff = cos_w * cos_g
        delta_g = - coeff @ grad
        new_grad = grad + coll_coef * delta_g
        return new_grad

    def _init_state(self, p, group=None):
        device, shape = p.device, p.shape
        state = self.state[p]
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)
        state.setdefault("decay_step", 0)
        state.setdefault("cos_sim", 0)
        # lr_mask
        lr_init = torch.ones(shape, device=device, dtype=torch.float16) * self.lr
        state.setdefault("lr_mask", self._q(lr_init))
        state.setdefault("avg_lr", float(self.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
        exp_init = torch.zeros_like(p)
        state.setdefault("exp_avg", self._q(exp_init))
        if group['full_finetune'] == False:
            state.setdefault("pre", None)
            # ==== ALLoRA ====
            #ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            #https://arxiv.org/abs/2410.09692
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))
        else:
            pre_init = p.clone()
            state.setdefault("pre", self._q(pre_init))

    def power_iteration(self, W, num_iters=3):
        device = W.device
        v = torch.randn(W.shape[1], 1, device=device)
        v = v / v.norm()
        for _ in range(num_iters):
            v = W.t() @ (W @ v)
            v = v / v.norm()
        sigma = (W @ v).norm()
        return sigma.item()

    @torch.no_grad()
    def step(self, closure=None):
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
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads) + 1e-12

            use_warmup, use_weight_decay = False, False
            if self._step % 500 <= self.warmup_steps:
                use_warmup = True
                if self.weight_decay > 0:
                    use_weight_decay = True
                    mean_norm = abs_all_group_grads.mean()
                    std_norm = abs_all_group_grads.std(unbiased=False) + 1e-12

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1
                self._step = state["step"] + 1

                # === grad 初始化 ===
                grad = p.grad

                # ==== AGR自適應梯度正則 ====
                #Adaptive Gradient Regularization: A Faster and Generalizable Optimization Technique for Deep Neural Networks
                #https://arxiv.org/pdf/2407.16944
                abs_grad = torch.abs(grad)
                agr = abs_grad / sum_abs_all_group_grads
                grad = grad * (1 - agr)
                beta1, beta2, beta3 = group["betas"]
                eps = group["eps"]
                alpha = (1 - beta1) / (1 - beta3)
                exp_avg = state['exp_avg']

                # ==== Simplified-AdEMAMix ====
                #Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants
                #https://arxiv.org/abs/2502.02431
                #https://github.com/DepenM/Simplified-AdEMAMix
                exp_avg.mul_(beta3).add_(grad)
                alpha_grad = alpha * grad
                final_exp_avg = (beta1 * exp_avg).mul_(alpha_grad)
                final_exp_avg_p2 = final_exp_avg ** 2

                # ==== AdamS ====
                #AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training
                #https://arxiv.org/abs/2502.02431
                exp_avg_sq = final_exp_avg_p2.mul(beta2).add_(alpha_grad ** 2, alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt_().add_(eps)
                update = final_exp_avg / denom

                #=== Cautious ===
                #Cautious Optimizers: Improving Training with One Line of Code
                #https://arxiv.org/abs/2411.16085
                #https://github.com/kyleliang919/C-Optim
                mask = (update * grad > 0).to(torch.float16)
                mask_ratio = mask.mean()
                mask.div_(mask_ratio.clamp_(min=1e-3))
                update = update * mask

                condition = 0.0
                if use_warmup:
                    if 'pre' not in state:
                        state.setdefault("pre", None)
                    delta_p = p - state["pre"] if state["pre"] else p
                    pre = state["pre"] if state["pre"] else torch.zeros_like(p)
                    condition = -torch.sum(p.grad * delta_p)
                else:
                    if 'pre' in state:
                        del state["pre"]

                # ==== Automagic lrmask ====
                # https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py
                lr_decay = 1
                if state["step"] > group["warmup_steps"]:
                    if group["lr"] > state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    elif group["lr"] < state["lr_max"]:
                        lr_decay = group["lr"] / state["lr_max"]

                if use_warmup:
                    if 'last_polarity' not in state:
                        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
                    last_polarity = state['last_polarity']
                    lr_mask = state['lr_mask']
                    lr_bump, d_coef= self.lr_bump, group["d_coef"]
                    lr_bump = lr_bump * min(state["step"] / 200, 1)
                    #Prodigy: An Expeditiously Adaptive Parameter-Free Learner
                    #https://arxiv.org/pdf/2306.06101
                    #https://github.com/konstmish/prodigy
                    current_polarity = grad > 0
                    same = (last_polarity == current_polarity).to(torch.float16)
                    state['last_polarity'] = current_polarity
                    if condition > 0.0:
                        lr_adjustment = (d_coef * same - (1 - same)) * lr_bump
                    elif condition < 0.0:
                        lr_adjustment = (same - d_coef * (1 - same)) * lr_bump
                    else:
                        lr_adjustment = (same * 2 - 1) * lr_bump
                    lr_mask.add_(lr_adjustment).clamp_(min=self.min_lr, max=self.max_lr)
                    state['avg_lr'] = state['lr_mask'].mean().item()

                    if state['step'] % 25:
                        lr_mask_f = lr_mask.float()
                        lr_medians = torch.quantile(lr_mask_f, torch.tensor([0.9,0.7, 0.5, 0.3, 0.1], device=lr_mask.device))
                        diff = torch.stack([torch.abs(lr_mask_f - m) for m in lr_medians], dim=-1)
                        nearest_idx = torch.argmin(diff, dim=-1)
                        lr_mask_flat = lr_mask.flatten()
                        nearest_idx_flat = nearest_idx.flatten()
                        lr_mask_flat = lr_medians[nearest_idx_flat]
                        state['lr_mask'] = lr_mask_flat.view_as(lr_mask).to(torch.float16)
                        state['avg_lr'] = state['lr_mask'].mean().item()

                # ==== VRAdam ====
                #A Physics-Inspired Optimizer: Velocity Regularized Adam
                #https://arxiv.org/abs/2505.13196
                vr = 1 / (1+ min(3 * (final_exp_avg_p2).sum(),10))
                allora = state.get("row_scaling", torch.tensor(1.0))
                new_lr = state['lr_mask']

                lr_tweak = lr_decay * vr * allora
                new_lr = new_lr * lr_tweak
                update.mul_(new_lr)

                # 權重衰減處理
                if use_warmup:
                    if condition < 0.0:
                        new_p = p - update
                        delta_n = new_p - pre if state["pre"] else new_p
                        ratio = self._ratio(delta_n, delta_p)
                        new_p = new_p - group["weight_decay2"] * ratio * delta_n
                        p.copy_(new_p)
                    elif use_weight_decay:
                        param_abs_grad = torch.abs(p.grad).mean()
                        norm_grad = (param_abs_grad - mean_norm) / std_norm
                        ada_alpha = 4
                        theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                        weight_decay = state['avg_lr'] * allora.mean().item() * vr * group["weight_decay"] * theta
                        p.data.mul_(1 - weight_decay)
                        p.add_(-update)
                    else:
                        p.add_(-update)
                else:
                    p.add_(-update)

        return loss

    def state_dict(self):
        state = super().state_dict()
        state['magic_version'] = 1
        return state

    def load_state_dict(self, state_dict):
        if 'magic_version' not in state_dict or state_dict['magic_version'] != 1:
            print('[WARNING] 您載入了非預期state dict，某些動態mask參數可能未正確同步！')
        super().load_state_dict(state_dict)
