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
def find_max_response_placement(score_map: torch.Tensor, p: int,
                                centre_if_constant: bool = False
                                ) -> Tuple[int, int]:
    r"""
    Top-left of the p x p window with the highest MEAN response.

        (top, left) = argmax_{(u,v)} MeanPool( score_map[u:u+p, v:v+p] )

    score_map : [H,W] float, any non-negative response — a class indicator
                (semantic placement) or a normalised sensitivity map (gradcam
                placement). An average-pool with kernel p and stride 1 IS the
                windowed mean at every position, so one argmax finishes it.

    centre_if_constant : tie-break for a CONSTANT map, where every window
                scores identically. argmax then returns index 0 — the top-left
                corner — which reads as a deliberate placement and is not.

                DEFAULT False, which reproduces the historical behaviour of
                find_semantic_placement() EXACTLY. That path can hit a constant
                map when the target class covers the whole prediction, and
                changing where the patch lands in that case would silently
                alter existing semantic-placement runs.

                The gradcam path passes True: a fully-suppressed Grad-CAM (ReLU
                zeroed every channel) produces a constant map, and there the
                map carries no localisation information at all, so falling back
                to the documented default position is the honest choice.
    """
    H, W = score_map.shape
    if p >= H or p >= W:
        return 0, 0
    if centre_if_constant and float(score_map.max() - score_map.min()) <= 0.0:
        return (H - p) // 2, (W - p) // 2

    score = F.avg_pool2d(score_map.float().view(1, 1, H, W),
                         kernel_size=p, stride=1)
    flat = int(score.view(-1).argmax().item())
    top, left = divmod(flat, score.shape[-1])
    return int(top), int(left)


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

    return find_max_response_placement(m.view(H, W), p)


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
