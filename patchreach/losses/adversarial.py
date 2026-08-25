r"""
Adversarial objectives.

  ce            untargeted, -CE(logits, GT)
  cospgd        untargeted, cos(pred,GT).detach() * CE, maximised
                (Agnihotri et al., ICML 2024)
  ipatch_cospgd TARGETED, drives every scored pixel toward target_class
                (IPatch objective + targeted CosPGD weighting)
  tsallis       untargeted, -L_q(p_y), maximised — CE carrying a per-pixel
                gradient weight p^(1-q), with q optionally scheduled across
                the run. Defined in tsallis.py because it is stateful; build()
                below dispatches it. (Matyasko et al., IJCNN 2026)

All four take an optional `support` mask restricting which pixels are scored.
That is where patch-footprint exclusion and reach-restriction compose: the
caller intersects them and passes one mask, rather than each loss knowing about
both.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _reduce(per_pixel: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Masked mean with a guarded denominator.

    THE NaN BUG THIS PREVENTS: F.cross_entropy(reduction='mean') computes
    sum/count over non-ignored pixels. Cityscapes batches DO occur where void
    regions (ego-vehicle hood, borders) plus the patch footprint cover every
    pixel — that is 0/0 = NaN. The NaN propagates through backward(), poisons
    Adam's moment estimates permanently, and the patch freezes at its init
    value for the rest of training with no error raised. Cost: one 100-epoch
    run before it was noticed.
    """
    n = valid.sum().clamp(min=1)
    return (per_pixel * valid.float()).sum() / n


def ce_loss(logits, labels, support: Optional[torch.Tensor] = None):
    """Untargeted CE. CE is HIGH when wrong -> maximise -> return -CE."""
    valid = labels != 255
    if support is not None:
        valid = valid & support
    per_pixel = F.cross_entropy(logits, labels, reduction="none",
                                ignore_index=255)
    return -_reduce(per_pixel, valid)


def cospgd_loss(logits, labels, support: Optional[torch.Tensor] = None):
    r"""
    CosPGD (Agnihotri et al., ICML 2024):

        L = mean_i [ cos(softmax(f(x)_i), onehot(y_i)) * CE(f(x)_i, y_i) ]

    The cosine SCALES the pixel-wise CE; it is a weight, never the objective.
    Pixels still predicted correctly (cos~1) keep full weight; pixels already
    fooled (cos~0) are de-emphasised. The gradient comes from CE, which does
    not saturate the way raw cosine does near alignment.

    FIDELITY NOTE — state this in the writeup: the paper does NOT detach the
    cosine; in their Algorithm 1 the gradient flows through the whole product.
    Detaching is a deliberate deviation, common in patch-attack
    implementations, that removes gradient contributions through the weight.

    A second deviation: they use sign-SGD with epsilon-projection (it is PGD).
    A patch is not an epsilon-ball perturbation, so we use Adam on the raw
    gradient. Both deviations are correct for this threat model but must be
    declared.

    Untargeted: MAXIMISE weighted CE -> return -L.
    """
    C = logits.shape[1]
    valid = labels != 255
    if support is not None:
        valid = valid & support

    safe = labels.clone()
    safe[labels == 255] = 0
    ref = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()

    cos_w = F.cosine_similarity(F.softmax(logits, 1), ref, dim=1).detach()
    ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    return -_reduce(cos_w * ce, valid)


def ipatch_cospgd_loss(logits, target_class: int,
                       footprint: Optional[torch.Tensor] = None,
                       support: Optional[torch.Tensor] = None):
    r"""
    IPatch objective + targeted CosPGD weighting:

        L = mean_i [ (1 - cos(softmax(f(x)_i), onehot(t))) * CE(f(x)_i, t) ]

    IPatch's KL(onehot(Y_t), Y) reduces exactly to CE(logits, t) and never
    references the ground truth. Every scored pixel is pushed toward the SAME
    class, so per-pixel gradients ALIGN and add — unlike untargeted objectives
    where pixel A says "not-road", pixel B says "not-sky", and the two cancel.

    The CosPGD weight FLIPS for targeted attacks: (1 - cos_to_target), so
    already-converted pixels are de-emphasised and effort concentrates on the
    stubborn remainder.

    `footprint` is MANDATORY. Leave the patch region in the loss and the
    cheapest win is to make the PATCH look like target_class: loss collapses,
    patch becomes camouflage, remote effect stays exactly zero.

    Adam MINIMISES this -> predictions driven toward target_class.
    """
    B, C, H, W = logits.shape
    target = torch.full((B, H, W), target_class, dtype=torch.long,
                        device=logits.device)
    if footprint is not None:
        target[footprint] = 255                       # camouflage prevention

    valid = target != 255
    if support is not None:
        valid = valid & support
    if valid.sum() == 0:
        raise ValueError("ipatch loss has no valid pixels — check the "
                         "footprint and reach masks do not exclude everything")

    # Ignored positions map to a DUMMY class 0, NOT clamp(max=C-1). Clamping
    # sends 255 -> class C-1 and computes the weight against a real edge class.
    # These pixels are zeroed by `valid` below so class 0 never enters the loss,
    # but the weight stays well-defined everywhere — matching the untargeted
    # path and the paper's Eq 8.
    safe = target.clone()
    safe[~valid] = 0
    ref = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()

    weight = 1.0 - F.cosine_similarity(F.softmax(logits, 1), ref, dim=1).detach()
    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=255)
    return _reduce(weight * ce, valid)


def build(loss_fn: str, target_class: int = 8,
          tsallis_q: float = 0.0, tsallis_schedule: str = "const",
          tsallis_q_start: float = -2.0, tsallis_q_end: float = 1.0,
          tsallis_total_steps: int = 1):
    """
    Returns f(logits, labels, footprint, support) -> scalar.

    ipatch ALWAYS receives the footprint regardless of the exclusion flag —
    camouflage prevention is not optional for a targeted objective.

    The tsallis_* arguments are read by the 'tsallis' branch ONLY. They carry
    defaults so every existing two-argument call site is unchanged, and no
    other branch reads them.
    """
    if loss_fn == "tsallis":
        from .tsallis import TsallisCELoss
        return TsallisCELoss(q=tsallis_q, schedule=tsallis_schedule,
                             q_start=tsallis_q_start, q_end=tsallis_q_end,
                             total_steps=tsallis_total_steps)
    if loss_fn == "ipatch_cospgd":
        return lambda lg, lb, fp, sp: ipatch_cospgd_loss(lg, target_class,
                                                         fp, sp)
    if loss_fn == "cospgd":
        return lambda lg, lb, fp, sp: cospgd_loss(lg, lb, sp)
    if loss_fn == "ce":
        return lambda lg, lb, fp, sp: ce_loss(lg, lb, sp)
    raise ValueError(f"unknown loss_fn {loss_fn!r}")
