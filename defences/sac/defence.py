"""SAC — Segment and Complete (Liu et al., CVPR 2022). arXiv:2112.04532."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ..base import PatchDefence, Purified
from .unet import load_sac_unet

SAC_SQUARE_SIZES = (100, 75, 50, 25)
SAC_DECLARED_SQUARE_SIZES = (125, 100, 75, 50, 25)
SAC_INIT_L1_THRESH = 0.9
SAC_THRESH_DECAY = 0.7
SAC_BREAK_ITS = 15


def _integral(x: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] -> [B,1,H+1,W+1] summed-area table, C[i,j] = sum(x[:i,:j])."""
    c = x.double().cumsum(-2).cumsum(-1)
    return F.pad(c, (1, 0, 1, 0))


def _box_sums(x: torch.Tensor, size: int) -> torch.Tensor:
    """Sum of x over every size x size window, stride 1. [B,1,H-size+1,W-size+1]."""
    c = _integral(x)
    return (c[..., size:, size:] - c[..., :-size, size:]
            - c[..., size:, :-size] + c[..., :-size, :-size])


def _union_of_squares(accept: torch.Tensor, size: int, hw) -> torch.Tensor:
    """Union of the size x size squares whose top-left corners accept==1."""
    H, W = hw
    pad = size - 1
    ap = F.pad(accept, (pad, pad, pad, pad))
    cov = _box_sums(ap, size)
    return (cov[..., :H, :W] >= 1.0).to(accept.dtype)


def completion_pass(mask: torch.Tensor, size: int,
                    l1_thresh: float) -> Optional[torch.Tensor]:
    """One (size, l1_thresh) pass."""
    B, _, H, W = mask.shape
    if size > H or size > W:
        return None
    inside = _box_sums(mask, size)
    total = mask.double().sum(dim=(1, 2, 3)).view(B, 1, 1, 1)
    hamming = float(size * size) - inside + (total - inside)
    accept = (hamming < (1.0 - l1_thresh) * size * size).double()
    return _union_of_squares(accept, size, (H, W)).to(mask.dtype)


def shape_completion(mask: torch.Tensor,
                     square_sizes: Sequence[int] = SAC_SQUARE_SIZES,
                     init_l1_thresh: float = SAC_INIT_L1_THRESH,
                     thresh_decay: float = SAC_THRESH_DECAY,
                     break_its: int = SAC_BREAK_ITS) -> torch.Tensor:
    """SAC's robust shape completion. [B,1,H,W] float 0/1 -> same."""
    if not torch.is_floating_point(mask):
        mask = mask.float()
    B = mask.shape[0]
    out = torch.zeros_like(mask)
    still_empty = mask.flatten(1).sum(1) > 0
    thresh = init_l1_thresh
    for _ in range(max(break_its, 0)):
        if not bool(still_empty.any()):
            break
        parts = []
        for s in square_sizes:
            part = completion_pass(mask, int(s), thresh)
            if part is not None:
                parts.append(part)
        if not parts:
            raise SystemExit(
                f"\nevery --sac_square_sizes entry {tuple(square_sizes)} is "
                f"larger than the {mask.shape[-2]}x{mask.shape[-1]} frame, so "
                f"shape completion can never fire.\n")
        union = (torch.stack(parts).sum(0) >= 1.0).to(mask.dtype)
        found = union.flatten(1).sum(1) > 0
        take = still_empty & found
        if bool(take.any()):
            sel = take.view(B, 1, 1, 1).to(mask.dtype)
            out = out + sel * union
            still_empty = still_empty & ~found
        thresh = thresh * thresh_decay
    return out.clamp_(0.0, 1.0)


class SACDefence(PatchDefence):
    """Segment-and-Complete as an input-space defence."""

    name = "sac"

    def __init__(self, ckpt: str, base_filter: int = 16,
                 square_sizes: Sequence[int] = SAC_SQUARE_SIZES,
                 image_scale: float = 1.0, complete: bool = True,
                 n_patch: int = 1, device="cpu", log=print):
        super().__init__()
        if n_patch != 1:
            raise SystemExit(
                "\nSAC's multi-patch path k-means-clusters the raw mask before "
                "completing each cluster. Patch.apply() composites exactly one "
                "footprint, so n_patch > 1 is refused rather than silently "
                "ignored.\n")
        self.unet = load_sac_unet(ckpt, base_filter, device=device, log=log)
        self.square_sizes = tuple(int(s) for s in square_sizes)
        self.image_scale = float(image_scale)
        self.complete = bool(complete)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x01: torch.Tensor) -> Purified:
        H, W = x01.shape[-2:]
        xin = x01
        if self.image_scale != 1.0:
            h = max(16, int(round(H * self.image_scale / 16)) * 16)
            w = max(16, int(round(W * self.image_scale / 16)) * 16)
            xin = F.interpolate(x01, size=(h, w), mode="bilinear",
                                align_corners=False)
        logits = self.unet(xin)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                                   align_corners=False)
        prob = torch.sigmoid(logits)
        raw = (prob > 0.5).to(x01.dtype)
        mask = (shape_completion(raw, self.square_sizes) if self.complete
                else raw)
        return Purified(x01=x01 * (1.0 - mask), mask=mask, raw_mask=raw,
                        prob=prob)

    def describe(self, log=print) -> None:
        log(f"[def ] sac  squares={self.square_sizes}  "
            f"complete={self.complete}  image_scale={self.image_scale}")
