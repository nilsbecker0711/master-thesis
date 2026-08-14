r"""
Reach-restricted optimisation.

RATIONALE
---------
The patch can only influence pixels inside the model's effective receptive
field. Measured here: random-patch prediction-change collapses below ~1% past
~300px at 512x1024 and ~150px at 256x512. Yet the loss averages over ALL valid
remote pixels — typically ~450k, of which only ~50-80k are reachable.

The unreachable ~85% contribute gradients that are tiny but NOT zero. They are
noise rather than signal: those gradients do not correspond to achievable
flips. They dilute the mean (shrinking the effective per-pixel contribution of
reachable pixels ~5x) and add variance to the update direction.

Restricting the loss support therefore (a) shrinks the denominator ~5x,
amplifying reachable-pixel gradients, and (b) removes unreachable contributions
outright instead of averaging them in.

THIS CANNOT EXTEND REACH. The reachable set is fixed by the architecture. What
it may buy is faster convergence and a higher plateau WITHIN the existing reach
— deeper flips near the patch, not a longer tail. If the near field is already
saturated there may be little headroom, and that null result is itself a test
of the two-factor model.

This supersedes the earlier dist_alpha experiment, which used DISTANCE as a
soft proxy for reachability and made results slightly WORSE. Distance is a poor
proxy: measured reach varies by source class and is non-monotonic in distance.
Here we either cut radially at the measured collapse point, or measure
reachability directly.

REPORTING: the loss support and the evaluation metric now cover different pixel
sets — we optimise on the reachable set but still report drop_remote over ALL
remote pixels. That is deliberate (the metric must not be tuned to the method)
and must be stated explicitly.
"""
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
    """
    Measure the reachable set directly: feed the patch region UNIFORM RANDOM
    noise (maximal stimulus) and record which pixels change their argmax.

    Target-independent and label-free by construction — it never references any
    class, only whether the prediction moved. That invariance is what makes it
    a valid control for the geometric factor.

    Returns (reach_prob [H,W] float, reach_mask [H,W] bool).
    """
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
    """
    [H,W] bool support mask, or None for mode='off'.

    The radial default of 0.62*img_h (~320px at 512, ~158px at 256) is set from
    the measured collapse point, not chosen arbitrarily.
    """
    if mode == "off":
        return None
    if mode == "radial":
        return radial_mask(footprint, radius if radius is not None
                           else 0.62 * img_h)
    if mode == "empirical":
        return empirical_mask(**kw)[1]
    raise ValueError(f"unknown reach mode {mode!r}")
