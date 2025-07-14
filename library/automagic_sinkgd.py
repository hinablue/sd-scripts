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
        orthograd: bool = True,
        sinkgd_iters: int = 2,
        mars: bool = True
    ):
        self.lr = lr
        self.sinkgd_iters = sinkgd_iters
        defaults = dict(
            lr=lr,
            allora=allora,
            eta=eta,
            orthograd=orthograd,
            mars=mars
        )
        super().__init__(params, defaults)

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

    # === Orthograd ===
    #Grokking at the Edge of Numerical Stability

    #https://arxiv.org/abs/2501.04697
    #https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev
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
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                state['step'] += 1
                grad = p.grad.data
                update = grad
                if grad.ndim == 2:
                    update = self.SinkGD(update)
                    if group["orthograd"] and state['step'] > 100:
                        update = self.Orthograd(p, update)
                    update = self.SinkGD(update, self.sinkgd_iters - 1)

                allora = state.get("row_scaling", 1.0)
                p.add_(-update.mul(group["lr"] * allora))

        return loss