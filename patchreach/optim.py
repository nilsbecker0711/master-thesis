"""The step rule, as a choice rather than a constant."""
from __future__ import annotations

import torch

OPTIMISERS = ("adam", "sign")


class SignSGD(torch.optim.Optimizer):
    """p <- p - lr * sign(grad), the update PGD takes inside its ball."""

    def __init__(self, params, lr: float = 0.01):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad.sign(), alpha=-group["lr"])
        return loss


def build(name: str, params, lr: float, betas=(0.9, 0.999)):
    """Returns the optimiser named by `name`."""
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, amsgrad=True)
    if name == "sign":
        return SignSGD(params, lr=lr)
    raise ValueError(f"optimiser must be one of {OPTIMISERS}, got {name!r}")
