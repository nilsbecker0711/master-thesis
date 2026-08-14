r"""
Patch placement policies.

A centred rectangle is not a threat model anyone can act on. A road-surface
marking is: Sato et al. (USENIX Sec '21), "Dirty road can attack", is exactly
that premise, and road is present in essentially every Cityscapes frame —
unlike sidewalk, which is absent from many.

COST: centre is the STRONGEST position in Yuan et al. Table 7 (Center 52.39 vs
Top Right 46.59, higher = stronger attack). Moving to the road surface trades
attack strength for plausibility. That is the same realism-vs-strength tradeoff
as the LAP alpha knob, and it should be measured, not assumed away.

CAVEAT on their Table 7: they optimise a FRESH patch per location, so it
measures "is this a good place to attack from", not "does a patch trained at A
still work at B". Placement TRANSFER is untested in the literature.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def find_semantic_placement(clean_pred: torch.Tensor, cls: int, p: int
                            ) -> Tuple[int, int]:
    """
    Top-left of the p x p window overlapping class `cls` most.

    clean_pred : [H,W] long — argmax of the CLEAN prediction, NOT the ground
                 truth. An attacker has no labels, only what the model outputs.

    Implemented as an average-pool over the class indicator, which is exactly
    "fraction of the window belonging to cls" at every position, then argmax.
    Falls back to centre when the class is absent.
    """
    H, W = clean_pred.shape
    if p >= H or p >= W:
        return 0, 0

    m = (clean_pred == cls).float().view(1, 1, H, W)
    if m.sum() == 0:
        return (H - p) // 2, (W - p) // 2

    score = F.avg_pool2d(m, kernel_size=p, stride=1)
    flat = int(score.view(-1).argmax().item())
    top, left = divmod(flat, score.shape[-1])
    return int(top), int(left)


def resolve(policy: str, H: int, W: int, p: int,
            clean_pred: Optional[torch.Tensor] = None,
            cls: int = 0,
            xy: Tuple[float, float] = (0.5, 0.5)) -> Tuple[int, int]:
    """
    Top-left corner under the configured policy.

    center   : image centre — the default and the strongest baseline.
    fixed    : `xy` as a normalised CENTRE, clipped to fit.
    semantic : largest region of `cls` in clean_pred; centre if unavailable.
    """
    if policy == "center" or (policy == "semantic" and clean_pred is None):
        return (H - p) // 2, (W - p) // 2

    if policy == "fixed":
        cy, cx = xy
        top = int(round(cy * H - p / 2))
        left = int(round(cx * W - p / 2))
        return max(0, min(top, H - p)), max(0, min(left, W - p))

    if policy == "semantic":
        return find_semantic_placement(clean_pred, cls, p)

    raise ValueError(f"unknown placement policy {policy!r}")
