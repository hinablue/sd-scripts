import torch
import math
from typing import List, Optional, Tuple, Dict, Any
import torch.nn.functional as F
from torch.nn.functional import normalize
from collections import defaultdict
from weakref import WeakKeyDictionary

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
    print("[Automagic_AdamS] 找不到 bitsandbytes，將以 FP16 儲存狀態。")


class TensorCache:
    """
    張量快取機制 - 用於快取經常計算的中間結果，減少重複計算
    """
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.step_created: Dict[str, int] = {}

    def get(self, key: str, step: int) -> Optional[torch.Tensor]:
        """獲取快取的張量"""
        if key in self.cache:
            # 檢查是否為當前步驟創建的快取
            if self.step_created.get(key, -1) == step:
                self.access_count[key] += 1
                return self.cache[key]
        return None

    def set(self, key: str, tensor: torch.Tensor, step: int) -> None:
        """設置快取張量"""
        if len(self.cache) >= self.max_cache_size:
            self._evict_least_used()

        self.cache[key] = tensor.clone().detach()
        self.step_created[key] = step
        self.access_count[key] = 1

    def _evict_least_used(self) -> None:
        """移除最少使用的快取項目"""
        if not self.cache:
            return

        # 找到使用次數最少的項目
        min_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        del self.cache[min_key]
        del self.access_count[min_key]
        del self.step_created[min_key]

    def clear_old_cache(self, current_step: int, max_age: int = 10) -> None:
        """清除過舊的快取項目"""
        keys_to_remove = []
        for key, created_step in self.step_created.items():
            if current_step - created_step > max_age:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                del self.access_count[key]
                del self.step_created[key]


class GradientProcessor:
    """
    梯度處理器 - 負責各種梯度處理技術
    """
    def __init__(self, cache: TensorCache):
        self.cache = cache

    def apply_agr(self, grad: torch.Tensor, sum_abs_all_grads: torch.Tensor) -> torch.Tensor:
        """
        應用自適應梯度正則化 (AGR)
        Adaptive Gradient Regularization: https://arxiv.org/pdf/2407.16944
        """
        abs_grad = torch.abs(grad)
        agr = abs_grad / sum_abs_all_grads
        return grad * (1 - agr)

    def orthograd(self, p: torch.Tensor, grad: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """
        正交梯度處理
        Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability
        """
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

        if state["cos_sim"] < -0.8 or G_shape[0] * G_shape[1] <= 50 ** 2:
            dot_ww = torch.dot(w, w)
            proj = dot_wd / (dot_ww + 1e-30)
            g_orth = g - w * proj
            g_orth_scaled = g_orth * (g_norm / (g_orth.norm(2) + 1e-30))
            return g_orth_scaled.view(G_shape)
        else:
            return grad

    def soft_collision_update(self, weight: torch.Tensor, grad: torch.Tensor,
                            coll_coef: float = 0.1) -> torch.Tensor:
        """軟碰撞更新"""
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


class LearningRateManager:
    """
    學習率管理器 - 負責自適應學習率計算
    """
    def __init__(self, cache: TensorCache):
        self.cache = cache

    def compute_automagic_lr(self, grad: torch.Tensor, state: Dict[str, Any],
                           group: Dict[str, Any], condition: torch.Tensor,
                           min_lr: float, max_lr: float, lr_bump: float) -> torch.Tensor:
        """計算 Automagic 學習率遮罩"""
        cache_key = f"lr_computation_{id(grad)}"
        cached_result = self.cache.get(cache_key, state["step"])
        if cached_result is not None:
            return cached_result

        if state["step"] < group["warmup_steps"]:
            last_polarity = state['last_polarity']
            current_polarity = (grad > 0)
            sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
            state['last_polarity'] = current_polarity
            lr_mask = state['lr_mask']

            if state["step"] < group["warmup_steps"] / 2:
                lr_bump_pos = lr_bump * group['d_coef'] if condition > 0.0 else lr_bump
                lr_bump_neg = lr_bump * group['d_coef'] if condition < 0.0 else lr_bump
            else:
                lr_bump_pos, lr_bump_neg = lr_bump, lr_bump

            new_lr = torch.where(
                sign_agree > 0,
                lr_mask + lr_bump_pos,
                lr_mask - lr_bump_neg
            )

            if group["lr"] >= state["lr_max"]:
                state["lr_max"] = group["lr"]
            new_lr = torch.clamp(new_lr, min=min_lr, max=max_lr)
            state['lr_mask'] = new_lr
            state['avg_lr'] = torch.mean(new_lr).item()
        else:
            if 'last_polarity' in state:
                del state['last_polarity']
            new_lr = state['lr_mask']

        self.cache.set(cache_key, new_lr, state["step"])
        return new_lr

    def compute_lr_decay(self, group: Dict[str, Any], state: Dict[str, Any]) -> float:
        """計算學習率衰減係數"""
        lr_decay = 1.0
        if group["lr"] >= state["lr_max"]:
            state["decay_step"] = 0
            state["lr_max"] = group["lr"]
        elif group["lr"] < state["lr_max"]:
            # Neural Thermodynamic Laws for Large Language Model Training
            # https://arxiv.org/abs/2505.10559
            state["decay_step"] += 1
            decay_progress = min(state["decay_step"], 3000) / 3000
            allowed_min_ratio = 1.0 - decay_progress
            lr_decay = max(max(group["lr"] / state["lr_max"], allowed_min_ratio), 0.1)
        return lr_decay


class WeightDecayProcessor:
    """
    權重衰減處理器 - 負責各種權重衰減技術
    """
    def __init__(self, cache: TensorCache):
        self.cache = cache

    def apply_adaptive_weight_decay(self, p: torch.Tensor, grad: torch.Tensor,
                                  new_lr: torch.Tensor, weight_decay: float,
                                  mean_norm: float, std_norm: float) -> None:
        """
        應用自適應權重衰減
        Adaptive Weight Decay for Deep Neural Networks: https://arxiv.org/abs/1907.08931
        """
        param_abs_grad = torch.abs(grad).mean()
        norm_grad = (param_abs_grad - mean_norm) / std_norm
        ada_alpha = 4
        theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
        p.data.mul_(1 - new_lr * weight_decay * theta)

    def _ratio(self, delta_new: torch.Tensor, delta_p: torch.Tensor) -> torch.Tensor:
        """計算投影比率"""
        curr_norm, prev_norm = torch.norm(delta_new), torch.norm(delta_p)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def apply_spd(self, p: torch.Tensor, update: torch.Tensor, pre: torch.Tensor,
                 delta_p: torch.Tensor, weight_decay: float) -> torch.Tensor:
        """
        應用選擇性投影衰減 (SPD)
        Rethinking Weight Decay for Robust Fine-Tuning: https://arxiv.org/abs/2411.01713
        """
        new_p = p - update
        delta_new = new_p - pre
        ratio = self._ratio(delta_new, delta_p)
        return new_p - weight_decay * ratio * delta_new


class Automagic_AdamS(torch.optim.Optimizer):
    """
    Automagic AdamS 優化器 - 整合多種最新優化技術的高效優化器

    結合了以下技術：
    - AdamS: Momentum Itself Can Be A Normalizer
    - Simplified-AdEMAMix: Connections between Schedule-Free Optimizers
    - AGR: Adaptive Gradient Regularization
    - 正交梯度: Grokking at the Edge of Numerical Stability
    - Cautious Optimizers: Improving Training with One Line of Code
    - VRAdam: A Physics-Inspired Optimizer
    - ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 3e-6,
        eps: float = 1e-8,
        clip_threshold: float = 1.0,
        betas: tuple = (0.5, 0.98, 0.99),
        alpha_decay: float = 0.9995,
        eta: float = 2,
        d_coef: float = 2,
        weight_decay: float = 1.0,
        weight_decay2: float = 4e-5,
        warmup_steps: int = 500,
        full_finetune: bool = False,
        use_8bit: bool = False,
        cache_size: int = 1000,
    ):
        """
        初始化優化器

        Args:
            params: 要優化的參數
            lr: 基礎學習率
            min_lr: 最小學習率
            max_lr: 最大學習率
            lr_bump: 學習率調整幅度
            eps: 數值穩定性參數
            clip_threshold: 梯度裁剪閾值
            betas: 動量參數 (beta1, beta2, beta3)
            alpha_decay: Alpha 衰減率
            eta: ALLoRA 參數
            d_coef: 動態係數
            weight_decay: 權重衰減1
            weight_decay2: 權重衰減2
            warmup_steps: 預熱步數
            full_finetune: 是否完全微調
            use_8bit: 是否使用8位量化
            cache_size: 快取大小
        """
        self.lr = lr
        self.use_8bit = bool(use_8bit and bnb is not None)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.full_finetune = full_finetune
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

        # 初始化快取和處理器
        self.cache = TensorCache(max_cache_size=cache_size)
        self.grad_processor = GradientProcessor(self.cache)
        self.lr_manager = LearningRateManager(self.cache)
        self.wd_processor = WeightDecayProcessor(self.cache)

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

    def _q(self, t: torch.Tensor) -> torch.Tensor:
        """量化張量到 (int8, scale) 兩元組"""
        if not self.use_8bit:
            return t
        q, s = bnb.functional.quantize_8bit(t)
        return (q, s)

    def _dq(self, q_or_t):
        """還原成 FP16/FP32 張量"""
        if not self.use_8bit:
            return q_or_t
        q, s = q_or_t
        return bnb.functional.dequantize_8bit(q, s)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        """計算 RMS"""
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    def _get_group_lr(self, group: Dict[str, Any]) -> float:
        """獲取群組平均學習率"""
        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])
        return float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.lr

    def _init_state(self, p: torch.Tensor, group: Dict[str, Any]) -> None:
        """初始化參數狀態"""
        device, shape = p.device, p.shape
        state = self.state[p]

        # 基本狀態初始化
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)
        state.setdefault("decay_step", 0)
        state.setdefault("cos_sim", 0)

        # 學習率遮罩初始化
        lr_init = torch.ones(shape, device=device, dtype=torch.float16) * self.lr
        state.setdefault("lr_mask", self._q(lr_init))
        state.setdefault("avg_lr", float(self.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # 指數移動平均初始化
        exp_init = torch.zeros_like(p)
        state.setdefault("exp_avg", self._q(exp_init))

        # 根據微調模式初始化不同狀態
        if group['full_finetune'] == False:
            state.setdefault("pre", None)
            # ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))
        else:
            pre_init = p.clone()
            state.setdefault("pre", self._q(pre_init))

    def power_iteration(self, W: torch.Tensor, num_iters: int = 3) -> float:
        """功率迭代法計算最大奇異值"""
        device = W.device
        v = torch.randn(W.shape[1], 1, device=device)
        v = v / v.norm()
        for _ in range(num_iters):
            v = W.t() @ (W @ v)
            v = v / v.norm()
        sigma = (W @ v).norm()
        return sigma.item()

    def _compute_adam_s_update(self, grad: torch.Tensor, exp_avg: torch.Tensor,
                              betas: Tuple[float, float, float], eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算 AdamS 更新
        AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training
        """
        beta1, beta2, beta3 = betas
        alpha = (1 - beta1) / (1 - beta3)

        # Simplified-AdEMAMix
        exp_avg.mul_(beta3).add_(grad)
        alpha_grad = alpha * grad
        alpha_grad_p2 = alpha_grad ** 2
        final_exp_avg = beta1 * exp_avg + alpha * grad
        final_exp_avg_p2 = final_exp_avg ** 2

        # AdamS
        exp_avg_sq = final_exp_avg_p2.mul_(beta2).add_(alpha_grad_p2, alpha=1.0 - beta2)
        denom = exp_avg_sq.sqrt().add_(eps)
        update = final_exp_avg / denom

        return update, final_exp_avg_p2

    def _apply_cautious_optimizer(self, update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        應用謹慎優化器
        Cautious Optimizers: Improving Training with One Line of Code
        """
        mask = (update * grad > 0).to(grad.dtype)
        mask_ratio = mask.mean()
        mask.div_(mask_ratio.clamp_(min=1e-3))
        return update * mask

    @torch.no_grad()
    def step(self, closure=None):
        """執行一步優化"""
        loss = closure() if closure is not None else None

        # 清理舊快取
        self.cache.clear_old_cache(self._step)

        for group in self.param_groups:
            # 收集群組梯度用於 AGR
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))

            if len(grads_this_group) == 0:
                continue

            all_group_grads = torch.cat(grads_this_group)
            abs_all_group_grads = torch.abs(all_group_grads)
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads) + 1e-12

            # 計算統計資訊用於自適應權重衰減
            mean_norm = std_norm = None
            if self._step < self.warmup_steps / 2 and self.weight_decay > 0:
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

                # 初始化梯度
                grad = p.grad.clone()

                # 應用 AGR 自適應梯度正則
                grad = self.grad_processor.apply_agr(grad, sum_abs_all_group_grads)

                beta1, beta2, beta3 = group["betas"]
                eps = group["eps"]
                exp_avg = state['exp_avg']

                # 應用正交梯度
                interval = int(math.ceil(0.5 / (1 - beta3)))
                if p.ndim == 2 and grad.ndim == 2:
                    if state["cos_sim"] < -0.8 or p.data.shape[0] * p.data.shape[1] <= 50 ** 2:
                        grad = self.grad_processor.orthograd(p, grad, state)
                    elif interval > 0 and state["step"] % interval == 0:
                        exp_avg = self.grad_processor.orthograd(p, exp_avg, state)

                # 計算 AdamS 更新
                update, final_exp_avg_p2 = self._compute_adam_s_update(grad, exp_avg, group["betas"], eps)

                # 應用謹慎優化器
                update = self._apply_cautious_optimizer(update, grad)

                # 計算條件用於 SPD 和 Prodigy
                if state["step"] < group["warmup_steps"]:
                    delta_p = p - state["pre"] if state["pre"] else p
                    pre = state["pre"] if state["pre"] else torch.zeros_like(p)
                    condition = -torch.sum(p.grad * delta_p)
                else:
                    if 'pre' in state:
                        del state["pre"]
                    condition = torch.tensor(0.0)

                # 計算自適應學習率
                new_lr = self.lr_manager.compute_automagic_lr(
                    grad, state, group, condition, self.min_lr, self.max_lr, self.lr_bump
                )

                # 計算學習率衰減
                lr_decay = self.lr_manager.compute_lr_decay(group, state)

                # 應用 ALLoRA 縮放
                allora = state.get("row_scaling", 1)

                # 應用 VRAdam 速度正則化
                vr = 1 / (1 + min(3 * final_exp_avg_p2.sum(), 10))

                # 組合所有學習率調整
                lr_tweak = lr_decay * allora * vr
                new_lr = new_lr * lr_tweak
                update.mul_(new_lr)

                # 應用權重衰減
                do_spd = False
                if state["step"] < group["warmup_steps"]:
                    if p.ndim == 2 and p.data.shape[0] * p.data.shape[1] <= 50 ** 2:
                        if state["step"] < group["warmup_steps"] / 2 and mean_norm is not None:
                            self.wd_processor.apply_adaptive_weight_decay(
                                p, p.grad, new_lr, group["weight_decay2"], mean_norm, std_norm
                            )
                    else:
                        # 應用 SPD 選擇性投影權重衰減
                        if condition < 0.0:
                            do_spd = True
                            new_p = self.wd_processor.apply_spd(
                                p, update, pre, delta_p, group["weight_decay"]
                            )
                            p.copy_(new_p)

                # 更新參數（如果沒有使用 SPD）
                if not do_spd:
                    p.add_(-update)

        return loss

    def state_dict(self):
        """返回狀態字典"""
        state = super().state_dict()
        state['magic_version'] = 2  # 更新版本號
        state['cache_info'] = {
            'cache_size': len(self.cache.cache),
            'max_cache_size': self.cache.max_cache_size
        }
        return state

    def load_state_dict(self, state_dict):
        """載入狀態字典"""
        if 'magic_version' not in state_dict:
            print('[WARNING] 您載入了舊版本的 state dict，某些功能可能不完全相容！')
        elif state_dict['magic_version'] != 2:
            print(f'[WARNING] State dict 版本不匹配 (expected: 2, got: {state_dict["magic_version"]})，某些動態參數可能未正確同步！')

        # 移除快取資訊（不需要載入）
        if 'cache_info' in state_dict:
            del state_dict['cache_info']

        super().load_state_dict(state_dict)

    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取快取統計資訊"""
        return {
            'cache_size': len(self.cache.cache),
            'max_cache_size': self.cache.max_cache_size,
            'total_access_count': sum(self.cache.access_count.values()),
            'cache_keys': list(self.cache.cache.keys())
        }

    def clear_cache(self) -> None:
        """清空快取"""
        self.cache.cache.clear()
        self.cache.access_count.clear()
        self.cache.step_created.clear()
