import torch
import torch.optim as optim
from typing import Optional, Callable, Tuple
import torch.nn.functional as F
from torch.nn.functional import normalize

class Automagic_Sinkgd(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        allora: bool = True,
        eta: float = 2,
        orthograd: bool = False,
        sinkgd_iters: int = 1,
        beta1: float = 0.8,
        decay: float = 0.997,
        weight_decay: float = 4e-5,
        warmup_steps: int = 200,
    ):
        self.lr = lr
        self.sinkgd_iters = sinkgd_iters
        defaults = dict(
            lr=lr,
            allora=allora,
            eta=eta,
            beta1=beta1,
            weight_decay=weight_decay,
            decay=decay,
            orthograd=orthograd
        )
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps
        self.max_lr = lr

    def _init_state(self, p, group=None):
        state = self.state[p]
        state.setdefault("step", 0)
        # ==== ALLoRA ====
        #ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
        #https://arxiv.org/abs/2410.09692
        if group['allora']:
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = (1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))).mean().item()

    @staticmethod
    @torch.jit.script
    def Orthograd(
        param: torch.Tensor,
        update: torch.Tensor,
        eps: float = 1e-30
    ):
        w = param.view(-1)
        g = update.view(-1)
        proj = torch.dot(w, g) / (torch.dot(w, w) + eps)
        g_orth = g - proj * w
        update = g_orth.view_as(update)
        return update

    # === SinkGD ===
    #Gradient Multi-Normalization for Stateless and Scalable LLM Training
    #https://arxiv.org/abs/2502.06742
    @staticmethod
    @torch.jit.script
    def SinkGD(
        update: torch.Tensor,
        num_sinkgd_iter: int = 1,
        eps: float = 1e-30
    ) -> torch.Tensor:
        if num_sinkgd_iter > 0:
            m, n = update.shape
            sqrt_n = n ** 0.5
            sqrt_m = m ** 0.5
            for _ in range(num_sinkgd_iter):
                row_norm = torch.linalg.vector_norm(update, dim=1, keepdim=True) + eps
                update = update * (sqrt_n / row_norm)
                col_norm = torch.linalg.vector_norm(update, dim=0, keepdim=True) + eps
                update = update * (sqrt_m / col_norm)
        return update

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_warmup, use_weight_decay = False, False
            if self._step <= self.warmup_steps:
                use_warmup = True
                if self.weight_decay > 0:
                    grads_this_group = []
                    for p in group["params"]:
                        if p.grad is not None:
                            grads_this_group.append(p.grad.view(-1))
                    if len(grads_this_group) == 0:
                        continue
                    all_group_grads = torch.cat(grads_this_group)
                    abs_all_group_grads = torch.abs(all_group_grads)
                    use_weight_decay = True
                    mean_norm = abs_all_group_grads.mean()
                    std_norm = abs_all_group_grads.std(unbiased=False) + 1e-12

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                if len(state) == 0:
                    self._init_state(p, group)
                    sigma = grad.std().nan_to_num()
                    grad_norm = grad.norm()
                    grad_norm_snr = (grad_norm / (sigma + 1e-8))
                    state['gsnr'] = grad_norm_snr

                state['step'] += 1
                self._step = state["step"] + 1
                beta1 = group["beta1"]
                gsnr_decay = group["decay"] ** state['step']
                gsnr = state["gsnr"]

                if beta1 > 0:
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p.data)
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                    exp_avg_bar = exp_avg.mul(beta1).add(grad, alpha = 1 - beta1)
                    # === Grams ===
                    #Grams: Gradient Descent with Adaptive Momentum Scaling
                    #https://arxiv.org/abs/2412.17107
                    #https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/grams.py
                    update = (exp_avg_bar).abs().mul_(grad.sign())
                else:
                    update = grad

                if grad.ndim == 2:
                    if group["orthograd"]:
                        update = self.Orthograd(p, update)
                    update = self.SinkGD(update, self.sinkgd_iters)
                else:
                    update = update * gsnr

                allora = state.get("row_scaling", 1.0)
                lr = group["lr"] * allora

                #Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias?
                #https://arxiv.org/abs/2504.12883
                if use_weight_decay:
                    #Adaptive Weight Decay for Deep Neural Networks
                    #https://arxiv.org/abs/1907.08931
                    param_abs_grad = torch.abs(p.grad).mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4
                    theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                    p.data.mul_(1 - (lr * group["weight_decay"] * theta))

                p.add_(-update.mul(lr))

        return loss