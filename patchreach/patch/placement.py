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

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

SCALE_REFS = ("height", "area")


def footprint_side(H: int, W: int, scale: float, ref: str = "height") -> int:
    r"""
    Side in px of the square the patch is pasted as. THE single rule — every
    renderer, placement, footprint and diagnostic calls this, so they cannot
    drift apart. It lives here because placement.py is a leaf module that both
    spec.py and conditional_generator.py already import.

    ref='height'  p = int(H * scale). DEFAULT, and every run before 2026-09.
                  Scale is a fraction of image HEIGHT, so ASPECT RATIO decides
                  how much of the frame the patch covers: at scale 0.25 it is
                  3.125% of a 512x1024 frame but 6.25% of a 1024x1024 one.
                  Width never enters.

    ref='area'    p = int(scale * sqrt(H * W)). The patch covers scale**2 of the
                  frame at ANY size and aspect ratio — 6.25% at 0.25, whether
                  the input is 512x1024, 768x768 or 1024x2048. Use this when
                  inputs differ in shape and "0.25" must mean the same thing.
                  Identical to 'height' on square inputs.

    What NEITHER rule holds constant is the PHYSICAL size in the scene: under
    scale='crop' tensor pixels are native pixels, so equal frame coverage on a
    smaller crop is a smaller object. When field of view changes, coverage and
    physical size cannot both be held — pick the one the comparison needs.

    int() floors in both branches, matching the historical rule exactly, so
    'height' is bit-identical to what every existing checkpoint was trained at.
    """
    if ref == "height":
        return int(H * scale)
    if ref == "area":
        return int(scale * math.sqrt(H * W))
    raise ValueError(f"scale_ref must be one of {SCALE_REFS}, got {ref!r}")


@torch.no_grad()
def find_max_response_placement(score_map: torch.Tensor, p: int,
                                centre_if_constant: bool = False,
                                margin: int = 0
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

    margin  : keep the window at least `margin` px from every image border.

                WHY THIS EXISTS. Without it the argmax can land flush against
                an edge — a measured run put the window at top = H - p exactly,
                because the sensitivity map's hottest ridge is the near-field
                road/sidewalk boundary along the bottom of a dashcam frame. A
                patch on the border has roughly HALF its effective receptive
                field outside the image, so it can only influence inward. That
                is a large, silent loss of geometric leverage, and it is
                invisible in the loss: the attack simply underperforms.

                DEFAULT 0, which reproduces the unmargined behaviour exactly,
                so existing semantic-placement runs are unaffected. The margin
                is clamped down if the image is too small to honour it, and a
                margin that would leave no legal window degrades to 0 rather
                than raising.
    """
    H, W = score_map.shape
    if p >= H or p >= W:
        return 0, 0
    if centre_if_constant and float(score_map.max() - score_map.min()) <= 0.0:
        return (H - p) // 2, (W - p) // 2

    # Largest margin that still leaves at least one legal window on each axis.
    m = max(0, min(int(margin), (H - p) // 2, (W - p) // 2))

    score = F.avg_pool2d(score_map.float().view(1, 1, H, W),
                         kernel_size=p, stride=1)          # [1,1,H-p+1,W-p+1]
    if m:
        score = score[:, :, m:score.shape[-2] - m, m:score.shape[-1] - m]

    # reshape, not view: the margin slice above leaves `score` non-contiguous
    # and .view() raises on it.
    flat = int(score.reshape(-1).argmax().item())
    top, left = divmod(flat, score.shape[-1])
    return int(top) + m, int(left) + m


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
            xy: Tuple[float, float] = (0.5, 0.5),
            score_map: Optional[torch.Tensor] = None,
            margin: int = 0) -> Tuple[int, int]:
    r"""
    Top-left corner under the configured policy.

    center   : image centre — the default and the strongest baseline.
    fixed    : `xy` as a normalised CENTRE, clipped to fit.
    semantic : largest region of `cls` in clean_pred; centre if unavailable.
    gradcam  : argmax over MeanPool(score_map) — the sensitivity hotspot.

    GRADCAM IS PER-IMAGE BY CONSTRUCTION and is only coherent for a per-image
    attack. A universal patch is one tensor shared across the dataset while the
    sensitivity map is not shared, so "place the shared patch at this image's
    hotspot" is not a well-defined policy for train.py. overfit.py and
    overfit_population.py optimise a fresh patch per image, so patch and
    placement are per-image together and the policy is consistent.

    Semantics are IDENTICAL to conditional_generator.resolve_batch_placement's
    gradcam branch — same helper, same centre_if_constant, same margin — so the
    per-image ablation and the generator's ablation A-vs-B measure the same
    thing. That equality is pinned by a test; it is the whole reason gradcam
    delegates here rather than reimplementing the argmax.

    MISSING MAP RAISES rather than falling back to centre. Silent fallback to
    centre is the exact bug class spec.py's ordering note exists to prevent —
    it produces a plausible number for a placement policy that never ran.
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

    if policy == "gradcam":
        if score_map is None:
            raise ValueError(
                "placement='gradcam' needs the sensitivity map. Pass "
                "score_map= to resolve(), or a `cam` to optimise.prepare(). "
                "Falling back to centre here would report a gradcam run that "
                "was actually centred.")
        return find_max_response_placement(score_map, p,
                                           centre_if_constant=True,
                                           margin=margin)

    raise ValueError(f"unknown placement policy {policy!r}")
