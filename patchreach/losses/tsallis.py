r"""
Tsallis cross-entropy attack objective.

  tsallis   untargeted, -L_q(p_y), maximised
            (TsallisPGD — Matyasko et al., IJCNN 2026, arXiv:2605.03405)

WHAT IT IS
----------
Per pixel, with p_y the softmax probability of the GROUND-TRUTH class:

    L_q(p_y) = (1 - p_y^(1-q)) / (1 - q)

Three properties are the whole argument for it:

  * q -> 1 recovers standard cross-entropy, L_1 = -log p_y. The family is a
    one-parameter generalisation, not a different objective.
  * dL_q/dp = p^(1-q) * dL_CE/dp. Tsallis IS cross-entropy carrying a
    per-pixel gradient weight p^(1-q). Nothing else about the attack changes.
  * That weight moves where the attack spends its gradient. The gradient-norm
    lower-bound proxy p^(2(1-q)) (1-p)^2 peaks at p* = (1-q)/(2-q), so
    q = 0 concentrates effort on pixels at p_y ~ 0.5, q = -1 on p_y ~ 0.67,
    q = -3 on p_y ~ 0.8. Pushing q negative aims the attack at pixels the
    model is still CONFIDENT about, instead of at ones already half-fooled.

WHY q <= 1 ONLY
---------------
q > 1 flips the sign of the exponent: the weight becomes p^(negative), which
GROWS without bound as p_y -> 0. The attack would then pour its gradient into
pixels it has already flipped and starve the ones it has not — the weighting
inverts toward LOW-CONFIDENCE pixels, which is the opposite of the mechanism
above. It is rejected at config-validation time rather than silently running a
different attack.

RELATION TO cospgd
------------------
Both re-weight a per-pixel CE. The CosPGD weight is cos(softmax, onehot),
DETACHED — a pure scaling with no gradient path. The Tsallis weight is not a
separate factor at all: it falls out of differentiating L_q, so it is exact
rather than a heuristic and there is no detach decision to declare.

SIGN CONVENTION — identical to ce_loss and cospgd_loss. L_q is HIGH when the
attack is winning, the optimiser MINIMISES, so this returns -mean(L_q).

MASKING — this module owns no masking of its own. `valid` is built with the
same two lines ce_loss uses and reduced with the same _reduce(), so
ignore-index 255, the patch-footprint exclusion and the reach mask compose
here exactly as they do for every other loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .adversarial import _reduce

# Below this |1 - q| the closed form is 0/0 in floating point and the CE limit
# is used instead. 1e-6 is comfortably above float32 resolution near 1 while
# still far below any q anyone would deliberately schedule to.
CE_LIMIT_EPS = 1e-6

SCHEDULES = ("const", "linear")


def validate_q(q: float = 0.0, schedule: str = "const",
               q_start: float = -2.0, q_end: float = 1.0) -> bool:
    """
    Raise ValueError unless every q the run can actually reach is <= 1.

    Only the values the SCHEDULE uses are checked: q_start/q_end are inert
    under 'const' and q is inert under 'linear', so rejecting a stale value
    that never enters the loss would be a false alarm.
    """
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
    """
    Active q at `step` of a `total_steps` run.

    t = step / max(total_steps - 1, 1), CLAMPED to [0, 1]. The clamp is what
    makes this safe against a caller whose loop is 1-based or which overruns
    its announced length: without it a linear schedule walks straight past
    q_end and, for q_end = 1, into the q > 1 regime this objective rejects.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"tsallis schedule must be one of {SCHEDULES}, "
                         f"got {schedule!r}")
    if schedule == "const":
        return float(q)
    t = float(step) / float(max(int(total_steps) - 1, 1))
    t = min(max(t, 0.0), 1.0)
    return float(q_start) + t * (float(q_end) - float(q_start))


def tsallis_from_log_py(log_py: torch.Tensor, q: float) -> torch.Tensor:
    r"""
    L_q from LOG probabilities. Never from probabilities.

        L_q = (1 - p^(1-q)) / (1 - q)
            = -expm1((1-q) * log p) / (1 - q)

    expm1 rather than exp(...) - 1 because for q near 1 the exponent is near
    zero, exp() returns 1 + tiny, and the subtraction cancels every significant
    digit of `tiny` before the division by the equally-tiny (1 - q) magnifies
    whatever is left. expm1 computes the difference directly and keeps them.

    Taking log p from log_softmax rather than log(softmax(.)) matters at the
    other end: |logit| ~ 50 underflows p_y to exactly 0 in float32, whereas
    log_softmax returns a finite ~ -100 and the whole expression stays finite
    (L_q -> 1/(1-q), and the gradient w.r.t. log p -> 0).
    """
    q = float(q)
    a = 1.0 - q
    if abs(a) < CE_LIMIT_EPS:
        return -log_py                      # the q -> 1 limit, i.e. plain CE
    return -torch.expm1(a * log_py) / a


def tsallis_per_pixel(logits: torch.Tensor, labels: torch.Tensor,
                      q: float) -> torch.Tensor:
    """
    [B,K,H,W] logits + [B,H,W] trainId labels -> [B,H,W] per-pixel L_q.

    Void labels are gathered against a DUMMY class 0 so the gather stays in
    bounds — the same device ipatch_cospgd_loss uses. Those positions are
    multiplied by zero in _reduce(), so class 0 never reaches the loss or the
    gradient; the value merely has to be well-defined.
    """
    safe = labels.clone()
    safe[labels == 255] = 0
    log_py = F.log_softmax(logits, dim=1).gather(
        1, safe.unsqueeze(1)).squeeze(1)
    return tsallis_from_log_py(log_py, q)


class TsallisCELoss:
    """
    Callable with the same convention every loss in adversarial.build returns:
    f(logits, labels, footprint, support) -> scalar. `footprint` is accepted
    and ignored, exactly as the untargeted ce/cospgd lambdas ignore it.

    Stateful because q is a function of optimisation progress. The loop calls
    on_step_begin(step, total_steps) before each step; a loop that does not
    (the hasattr guard in the loops means the pre-existing losses never see
    that call at all) leaves the step at 0, so 'const' is unaffected and
    'linear' pins at q_start.
    """

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
