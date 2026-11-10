"""Reach-restricted optimisation."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def centroid(footprint: torch.Tensor) -> Tuple[float, float]:
    idx = footprint[0].nonzero()
    return idx[:, 0].float().mean().item(), idx[:, 1].float().mean().item()


def distance_map(H: int, W: int, cy: float, cx: float, device) -> torch.Tensor:
    ys = torch.arange(H, device=device).view(H, 1).expand(H, W).float()
    xs = torch.arange(W, device=device).view(1, W).expand(H, W).float()
    return torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)


def radial_mask(footprint: torch.Tensor, radius: float) -> torch.Tensor:
    """[H,W] bool disc around the footprint centroid."""
    _, H, W = footprint.shape
    cy, cx = centroid(footprint)
    return distance_map(H, W, cy, cx, footprint.device) <= radius


@torch.no_grad()
def empirical_mask(model, loader, patch, device, n_images=8, n_probes=16,
                   thresh=0.02, log=print):
    """Measure the reachable set directly: feed the patch region UNIFORM RANDOM noise (maximal stimulus) and record which pixels change their argmax."""
    from ..data.cityscapes import upsample_to

    log(f"\n{'='*66}")
    log(f" MEASURING EMPIRICAL REACH — {n_images} images x {n_probes} probes, "
        f"threshold {thresh}")
    log(f"{'='*66}")

    accum, seen, fp_ref, n_batches = None, 0, None, 0
    for imgs, _ in loader:
        if seen >= n_images:
            break
        imgs = imgs.to(device)
        B, _, H, W = imgs.shape
        clean = upsample_to(model(imgs), (H, W)).argmax(1)
        if accum is None:
            accum = torch.zeros(H, W, device=device)

        original = patch.param.detach().clone()
        for _ in range(n_probes):
            with torch.no_grad():
                patch.param.data = torch.randn_like(patch.param) * 3.0
            adv = upsample_to(model(patch.apply(imgs)[0]), (H, W)).argmax(1)
            if fp_ref is None:
                fp_ref = patch.apply(imgs)[1]
            accum += (adv != clean).float().mean(0)
        with torch.no_grad():
            patch.param.data = original

        seen += B
        n_batches += 1

    reach_prob = accum / max(n_probes * n_batches, 1)
    reach_mask = reach_prob >= thresh

    cy, cx = centroid(fp_ref)
    H, W = reach_prob.shape
    dist = distance_map(H, W, cy, cx, device)
    outside = ~fp_ref[0]

    log("\n  Reach profile (prediction-change rate vs distance):")
    max_d = int(dist[outside].max().item())
    step = max(50, max_d // 8)
    for lo in range(0, max_d, step):
        ring = outside & (dist >= lo) & (dist < lo + step)
        if ring.sum() == 0:
            continue
        r = reach_prob[ring].mean().item()
        log(f"    {lo:4d}-{lo+step:<4d}px : {r*100:5.2f}%  {'#' * int(r*100)}")

    n_r, n_t = int(reach_mask.sum()), int(outside.sum())
    log(f"\n  Reachable: {n_r:,} / {n_t:,} remote px "
        f"({100*n_r/max(n_t,1):.1f}%) — loss support shrinks "
        f"{n_t/max(n_r,1):.1f}x")
    log(f"{'='*66}\n")
    return reach_prob, reach_mask


def build(mode: str, footprint: torch.Tensor, img_h: int,
          radius: Optional[float] = None, **kw) -> Optional[torch.Tensor]:
    """[H,W] bool support mask, or None for mode='off'."""
    if mode == "off":
        return None
    if mode == "radial":
        return radial_mask(footprint, radius if radius is not None
                           else 0.62 * img_h)
    if mode == "empirical":
        return empirical_mask(**kw)[1]
    raise ValueError(f"unknown reach mode {mode!r}")
