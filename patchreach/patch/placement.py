"""Patch placement policies."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

SCALE_REFS = ("height", "area")


def footprint_side(H: int, W: int, scale: float, ref: str = "height") -> int:
    """Side in px of the square the patch is pasted as."""
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
    """Top-left of the p x p window with the highest MEAN response."""
    H, W = score_map.shape
    if p >= H or p >= W:
        return 0, 0
    if centre_if_constant and float(score_map.max() - score_map.min()) <= 0.0:
        return (H - p) // 2, (W - p) // 2

    m = max(0, min(int(margin), (H - p) // 2, (W - p) // 2))

    score = F.avg_pool2d(score_map.float().view(1, 1, H, W),
                         kernel_size=p, stride=1)
    if m:
        score = score[:, :, m:score.shape[-2] - m, m:score.shape[-1] - m]

    flat = int(score.reshape(-1).argmax().item())
    top, left = divmod(flat, score.shape[-1])
    return int(top) + m, int(left) + m


@torch.no_grad()
def find_semantic_placement(clean_pred: torch.Tensor, cls: int, p: int
                            ) -> Tuple[int, int]:
    """Top-left of the p x p window overlapping class `cls` most."""
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
    """Top-left corner under the configured policy."""
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
