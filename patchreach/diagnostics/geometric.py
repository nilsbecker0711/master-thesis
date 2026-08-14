r"""
GEOMETRIC diagnostics — how far can the patch reach, independent of semantics.

Everything here is TARGET-INDEPENDENT, LOSS-INDEPENDENT and LABEL-FREE by
construction. That is what makes it a control rather than a correlate: it
measures the model's effective receptive field with semantics held out, so it
can be compared across architectures, datasets and objectives without
confounds. It is also why measure_erf.py can run on ADE20K weights.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from ..data.cityscapes import upsample_to
from ..losses.reach import centroid, distance_map

RINGS = [(0, 150), (150, 300), (300, 450), (450, 650), (650, 900), (900, 1200)]


@torch.no_grad()
def receptive_field(model, imgs, patch, n_probes: int = 16,
                    rings: List[Tuple[int, int]] = None, log=print):
    """
    Empirical ERF: fill the patch region with UNIFORM RANDOM noise and record
    which pixels change their argmax.

    Random noise is MAXIMAL STIMULUS — an optimised patch cannot exceed what
    the architecture will propagate, so this is an upper bound on reach. If the
    trained patch's reach curve matches this, the wall is geometric and no loss
    or weighting change can move it.

    Returns (reach_prob [H,W], ring_stats [(lo,hi,rate_pct)]).
    """
    rings = rings or RINGS
    B, _, H, W = imgs.shape
    clean = upsample_to(model(imgs), (H, W)).argmax(1)

    original = patch.param.detach().clone()
    accum = torch.zeros(H, W, device=imgs.device)
    footprint = None
    for _ in range(n_probes):
        patch.param.data = torch.randn_like(patch.param) * 3.0   # -> sigmoid ~U
        patched, fp = patch.apply(imgs)
        footprint = fp if footprint is None else footprint
        adv = upsample_to(model(patched), (H, W)).argmax(1)
        accum += (adv != clean).float().mean(0)
    patch.param.data = original

    reach = accum / n_probes
    dist = distance_map(H, W, *centroid(footprint), imgs.device)
    outside = ~footprint[0]

    stats = []
    log("\n[geometric] empirical receptive field (random-patch stimulus):")
    log("            target-independent — an upper bound on any patch's reach")
    for lo, hi in rings:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() == 0:
            continue
        r = reach[ring].mean().item() * 100
        stats.append((lo, hi, r))
        log(f"    {lo:5d}-{hi:<5d}px : {r:6.2f}%  {'#' * int(r)}")

    below = [lo for lo, hi, r in stats if r < 5.0]
    if below:
        log(f"    -> falls below 5% at ~{below[0]}px")
    return reach, stats


@torch.no_grad()
def scale_sweep(model, imgs, patch, scales=(0.15, 0.25, 0.35, 0.50),
                n_probes: int = 8, log=print):
    """
    Does reach GROW with patch size, and how fast?

    The two-factor model predicts d_max ~ sigma * sqrt(2 ln(kappa*sqrt(3)*p / m)),
    i.e. reach grows only as sqrt(log p). If the measurement matches that weak
    scaling, "just use a bigger patch" is quantitatively ruled out: covering
    another 200px would need a patch larger than the image.

    Restores the original scale before returning.
    """
    B, _, H, W = imgs.shape
    clean = upsample_to(model(imgs), (H, W)).argmax(1)
    original_scale, original_param = patch.cfg.scale, patch.param.detach().clone()

    out = {}
    log("\n[geometric] patch-scale sweep (reach vs patch size):")
    for s in scales:
        patch.cfg.scale = s
        accum = torch.zeros(H, W, device=imgs.device)
        fp = None
        for _ in range(n_probes):
            patch.param.data = torch.randn_like(patch.param) * 3.0
            patched, f = patch.apply(imgs)
            fp = f if fp is None else fp
            accum += (upsample_to(model(patched), (H, W)).argmax(1)
                      != clean).float().mean(0)
        reach = accum / n_probes
        dist = distance_map(H, W, *centroid(fp), imgs.device)
        outside = ~fp[0]

        row = []
        for lo, hi in RINGS[:5]:
            ring = outside & (dist >= lo) & (dist < hi)
            if ring.sum():
                row.append(((lo + hi) // 2, reach[ring].mean().item() * 100))
        out[s] = row
        log(f"    scale {s:.2f} (p={int(H*s)}px): "
            + "  ".join(f"{d}px={r:.1f}%" for d, r in row))

    patch.cfg.scale, patch.param.data = original_scale, original_param
    return out


def plot_erf(stats, out_path, title="Empirical receptive field"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = [(lo + hi) / 2 for lo, hi, _ in stats]
    r = [x for _, _, x in stats]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d, r, "-o", color="#c44e52")
    ax.axhline(5, color="gray", ls="--", lw=0.8, label="5% threshold")
    ax.set_xlabel("distance from patch centre (px)")
    ax.set_ylabel("prediction-change rate (%)")
    ax.set_title(f"{title}\nrandom-patch stimulus — target-independent ceiling")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
