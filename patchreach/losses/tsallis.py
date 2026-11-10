"""Tsallis cross-entropy attack objective."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .adversarial import _reduce

CE_LIMIT_EPS = 1e-6

SCHEDULES = ("const", "linear")


def validate_q(q: float = 0.0, schedule: str = "const",
               q_start: float = -2.0, q_end: float = 1.0) -> bool:
    """Raise ValueError unless every q the run can actually reach is <= 1."""
    if schedule not in SCHEDULES:
        raise ValueError(f"tsallis schedule must be one of {SCHEDULES}, "
                         f"got {schedule!r}")
    live = ((("tsallis_q", q),) if schedule == "const"
            else (("tsallis_q_start", q_start), ("tsallis_q_end", q_end)))
    for name, v in live:
        if float(v) > 1.0:
            raise ValueError(
                f"{name}={v!r} is greater than 1. The Tsallis attack "
                "objective is defined for q <= 1 only: at q > 1 the gradient "
                "weight p^(1-q) has a negative exponent and blows up on "
                "ALREADY-FLIPPED low-confidence pixels, which is the opposite "
                "of the re-weighting this attack exists to apply.")
    return True


def schedule_q(step: int, total_steps: int, q: float = 0.0,
               schedule: str = "const", q_start: float = -2.0,
               q_end: float = 1.0) -> float:
    """Active q at `step` of a `total_steps` run."""
    if schedule not in SCHEDULES:
        raise ValueError(f"tsallis schedule must be one of {SCHEDULES}, "
                         f"got {schedule!r}")
    if schedule == "const":
        return float(q)
    t = float(step) / float(max(int(total_steps) - 1, 1))
    t = min(max(t, 0.0), 1.0)
    return float(q_start) + t * (float(q_end) - float(q_start))


def tsallis_from_log_py(log_py: torch.Tensor, q: float) -> torch.Tensor:
    """L_q from LOG probabilities."""
    q = float(q)
    a = 1.0 - q
    if abs(a) < CE_LIMIT_EPS:
        return -log_py
    return -torch.expm1(a * log_py) / a


def tsallis_per_pixel(logits: torch.Tensor, labels: torch.Tensor,
                      q: float) -> torch.Tensor:
    """[B,K,H,W] logits + [B,H,W] trainId labels -> [B,H,W] per-pixel L_q."""
    safe = labels.clone()
    safe[labels == 255] = 0
    log_py = F.log_softmax(logits, dim=1).gather(
        1, safe.unsqueeze(1)).squeeze(1)
    return tsallis_from_log_py(log_py, q)


class TsallisCELoss:
    """Callable with the same convention every loss in adversarial.build returns: f(logits, labels, footprint, support) -> scalar."""

    def __init__(self, q: float = 0.0, schedule: str = "const",
                 q_start: float = -2.0, q_end: float = 1.0,
                 total_steps: int = 1):
        validate_q(q, schedule, q_start, q_end)
        self.q_const = float(q)
        self.schedule = schedule
        self.q_start = float(q_start)
        self.q_end = float(q_end)
        self.total_steps = max(int(total_steps), 1)
        self._step = 0

    @property
    def q(self) -> float:
        """The q IN FORCE right now — log this, not q_start/q_end."""
        return schedule_q(self._step, self.total_steps, self.q_const,
                          self.schedule, self.q_start, self.q_end)

    @property
    def step(self) -> int:
        return self._step

    def on_step_begin(self, step: int, total_steps: Optional[int] = None):
        if total_steps is not None:
            self.total_steps = max(int(total_steps), 1)
        self._step = int(step)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor,
                 footprint: Optional[torch.Tensor] = None,
                 support: Optional[torch.Tensor] = None) -> torch.Tensor:
        valid = labels != 255
        if support is not None:
            valid = valid & support
        per_pixel = tsallis_per_pixel(logits, labels, self.q)
        return -_reduce(per_pixel, valid)

    def __repr__(self):
        if self.schedule == "const":
            return f"TsallisCELoss(q={self.q_const:g})"
        return (f"TsallisCELoss(linear {self.q_start:g} -> {self.q_end:g} "
                f"over {self.total_steps} steps, q={self.q:g})")
