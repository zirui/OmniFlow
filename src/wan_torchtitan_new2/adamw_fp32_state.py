"""
WanNew2 optimizer helpers.

Goals:
- Keep optimizer state & master weights in FP32 (match typical bf16+FSDP baselines).
- Match HF Trainer-style weight decay exclusion:
  - exclude bias
  - exclude all 1D parameters (LayerNorm/RMSNorm scales, etc.)
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims


class AdamWFP32State(torch.optim.Optimizer):
    """
    AdamW variant that keeps:
    - exp_avg / exp_avg_sq in FP32
    - a persistent FP32 master copy of parameters

    The actual model parameters can remain bf16/fp16; we copy updated master
    weights back each step.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWFP32State does not support sparse gradients")

                grad_fp32 = p.grad.detach().float()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad_fp32, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(grad_fp32, dtype=torch.float32)
                    state["master_param"] = p.detach().clone().float()

                p_fp32: torch.Tensor = state["master_param"]
                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step: int = state["step"]

                # Decoupled weight decay on master weights
                if weight_decay != 0.0:
                    p_fp32.add_(p_fp32, alpha=-lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                step_size = lr / bias_correction1

                p_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                p.copy_(p_fp32.to(dtype=p.dtype))

        return loss


def _is_no_decay_param(name: str, p: nn.Parameter) -> bool:
    if name.endswith(".bias"):
        return True
    # HF-style: all 1D parameters (norm scales, etc.) have no weight decay.
    if getattr(p, "ndim", 0) == 1:
        return True
    # Extra safety: common name patterns
    lname = name.lower()
    if "norm" in lname or "ln" in lname or "rms" in lname:
        return True
    return False


def _group_params_with_decay(model: nn.Module, weight_decay: float) -> list[dict]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if _is_no_decay_param(name, p) else decay_params).append(p)

    groups: list[dict] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def _apply_param_groups(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    weight_decay: float,
) -> None:
    for optimizer, model in zip(optimizers.optimizers, model_parts):
        groups = _group_params_with_decay(model, weight_decay)
        for group in groups:
            for key, value in optimizer.defaults.items():
                group.setdefault(key, value)
        optimizer.param_groups = groups


def build_wan_new2_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager=None,
) -> OptimizersContainer:
    # Keep parity with native baseline: FP32 master weights for bf16 training.
    optimizers = OptimizersContainer(
        model_parts,
        AdamWFP32State,
        {
            "lr": optimizer_config.lr,
            "betas": (optimizer_config.beta1, optimizer_config.beta2),
            "eps": optimizer_config.eps,
            "weight_decay": optimizer_config.weight_decay,
        },
    )
    _apply_param_groups(optimizers, model_parts, optimizer_config.weight_decay)
    return optimizers

