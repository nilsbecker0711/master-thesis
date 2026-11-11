"""DefendedSegModel — the defence as a model wrapper, decided once."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from .base import PatchDefence, Purified


class DefendedSegModel(nn.Module):
    """Purify in pixel space, then forward."""

    def __init__(self, model: nn.Module, defence: PatchDefence,
                 mean_t: torch.Tensor, std_t: torch.Tensor):
        super().__init__()
        self.model = model
        self.defence = defence
        self.register_buffer("_mean", mean_t.detach().clone())
        self.register_buffer("_std", std_t.detach().clone())
        self.enabled = True
        self.last: Optional[Purified] = None
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def mode(self) -> str:
        return getattr(self.model, "mode", "whole")

    @contextmanager
    def bypassed(self):
        """Run the inner model with no defence."""
        was = self.enabled
        self.enabled = False
        try:
            yield self
        finally:
            self.enabled = was

    def to_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised -> [0,1] RGB."""
        return (x * self._std + self._mean).clamp(0.0, 1.0)

    def to_normed(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._mean) / self._std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last = None
            return self.model(x)
        if x.requires_grad or x.grad_fn is not None:
            raise RuntimeError(
                "DefendedSegModel received an input that carries gradient.\n\n"
                "Every defence here thresholds, so the gradient through it is "
                "ZERO. An attack optimised through this block does not crash — "
                "it gets no signal, the patch stops moving, and the run reports "
                "a robust model. Nothing downstream would flag it.\n\n"
                "  * evaluating a patch: wrap the forward in torch.no_grad().\n"
                "  * the per-image diagnostics need gradients, so pass\n"
                "    --diagnostics_on -1 when --defence is set.\n"
                "  * an ADAPTIVE attack through the defence is separate work: "
                "it needs a straight-through estimator (BPDA), and it must be "
                "reported as a different threat model, not as this one.\n")
        self.last = self.defence(self.to_pixels(x))
        return self.model(self.to_normed(self.last.x01))

    def describe(self, log=print) -> None:
        self.defence.describe(log)
