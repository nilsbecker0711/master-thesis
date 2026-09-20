r"""
The step rule, as a choice rather than a constant.

WHY THIS EXISTS
---------------
Adam has been the optimiser since the initial code commit (320c5cb). Two
docstrings justify it against PGD's sign-SGD -- losses/adversarial.py on the
threat model ("a patch is not an epsilon-ball perturbation"), patch/optimise.py
on the step ("Adam's update is on the order of `lr` per coordinate whatever the
gradient magnitude"). Both arguments are sound and neither was ever tested:
nothing in this repository had run sign-SGD, so the comparison the write-up
leans on when it declines to call the attack PGD was an argument, not a result.

MATCHED STEP SIZE -- the whole difficulty of the comparison
-----------------------------------------------------------
Adam's update is m_hat / sqrt(v_hat). For a coordinate whose gradient sign is
consistent across steps that ratio approaches +/-1, so the displacement is ~lr
per coordinate regardless of gradient magnitude. sign-SGD's displacement is
EXACTLY lr per coordinate. The two are therefore matched at EQUAL lr, and that
is what the ablation holds fixed.

This matters because the obvious alternatives are wrong. Matching on gradient
norm compares a normalised step against an unnormalised one. Matching on total
displacement bakes in the answer. Equal lr is the only match under which the
two rules take the same-sized step, which is the property that makes the
remaining difference attributable to the rule itself.

WHAT THIS MODULE IS NOT
-----------------------
It is not a PGD implementation. PGD is a step rule plus a projection onto an
epsilon-ball around the clean image; the projection half lives in
Patch.project() and the epsilon-ball does not exist under a patch threat model
at all. Swapping the step rule here gets you one of PGD's three ingredients.
See PatchConfig.pixel_param for the second.
"""
from __future__ import annotations

import torch

OPTIMISERS = ("adam", "sign")


class SignSGD(torch.optim.Optimizer):
    r"""
    p <- p - lr * sign(grad), the update PGD takes inside its ball.

    No momentum and no accumulator, deliberately: the point of the comparison
    is Adam's adaptivity against its absence, and a momentum term would leave
    it ambiguous which of the two was responsible.

    sign(0) = 0, so a coordinate with exactly zero gradient does not move. That
    is PyTorch's sign() and it is also what PGD does; it matters only for a
    patch pixel whose gradient has genuinely vanished, which under
    pixel_param='direct' is precisely the case this ablation exists to measure.

    Subclasses torch.optim.Optimizer rather than open-coding the update in the
    loop so that CosineAnnealingLR and ReduceLROnPlateau drive it unchanged --
    the schedules read param_groups[i]['lr'], which is where lr lives here too.
    A hand-rolled update would have silently ignored --lr_schedule and made the
    ablation a comparison of two schedules as well as two step rules.
    """

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
    """
    Returns the optimiser named by `name`.

    `betas` is exposed because the two training loops disagree -- train.py has
    always used (0.5, 0.999) and optimise.py (0.9, 0.999) -- and this function
    must not silently move either. It is read by the 'adam' branch only.
    """
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, amsgrad=True)
    if name == "sign":
        return SignSGD(params, lr=lr)
    raise ValueError(f"optimiser must be one of {OPTIMISERS}, got {name!r}")
