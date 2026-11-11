"""GEOMETRIC diagnostics — how far can the patch reach, independent of semantics."""
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
import torch

from ..data.cityscapes import upsample_to
from ..losses.reach import centroid, distance_map

RINGS = [(0, 150), (150, 300), (300, 450), (450, 650), (650, 900), (900, 1200)]


@torch.no_grad()
def _probe(model, imgs, patch, clean, n_probes, H, W):
    """n_probes random parameters -> mean per-pixel argmax-change rate."""
    accum = torch.zeros(H, W, device=imgs.device)
    footprint = None
    for _ in range(n_probes):
        patch.param.data = torch.randn_like(patch.param) * 3.0
        patch.project()
        patched, fp = patch.apply(imgs)
        footprint = fp if footprint is None else footprint
        accum += (upsample_to(model(patched), (H, W)).argmax(1)
                  != clean).float().mean(0)
    return accum / n_probes, footprint


@contextmanager
def _unconstrained(patch):
    """Render the patch as RAW PIXELS for the duration of the block."""
    saved = patch.cfg.mode
    patch.cfg.mode = "raw"
    try:
        yield
    finally:
        patch.cfg.mode = saved


@torch.no_grad()
def receptive_field(model, imgs, patch, n_probes: int = 16,
                    rings: List[Tuple[int, int]] = None, log=print):
    """Empirical ERF: fill the patch region with random noise and record which pixels change their argmax."""
    rings = rings or RINGS
    B, _, H, W = imgs.shape
    clean = upsample_to(model(imgs), (H, W)).argmax(1)

    original = patch.param.detach().clone()
    bounded = patch.cfg.mode in ("csf", "universal_csf")
    can_unconstrain = (tuple(patch.param.shape)
                       == (3, patch.cfg.size, patch.cfg.size))

    null_reach, footprint = _probe(model, imgs, patch, clean, n_probes, H, W)
    if bounded and can_unconstrain:
        with _unconstrained(patch):
            ceiling_reach, _ = _probe(model, imgs, patch, clean,
                                      n_probes, H, W)
    else:
        ceiling_reach = null_reach
    patch.param.data = original

    dist = distance_map(H, W, *centroid(footprint), imgs.device)
    outside = ~footprint[0]

    split = bounded and can_unconstrain
    stats = []
    log(f"\n[geometric] empirical receptive field "
        f"(random-patch probe, n={n_probes}):")
    if split:
        log("            ceiling — unconstrained max-contrast square: the "
            "architectural limit,")
        log("                      NOT a patch this attack could build")
        log(f"            null    — random patch at the SAME tau "
            f"({patch.cfg.csf_threshold:g}) as the attack:")
        log("                      what an unoptimised perturbation of equal "
            "perceptual cost reaches")
        log(f"    {'ring(px)':>14}  {'ceiling':>9}  {'null':>9}")
    else:
        log("            unconstrained random square — target-independent, "
            "an upper bound on reach")

    for lo, hi in rings:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() == 0:
            continue
        c = ceiling_reach[ring].mean().item() * 100
        n = null_reach[ring].mean().item() * 100
        stats.append((lo, hi, c, n))
        if split:
            log(f"    {f'{lo}-{hi}':>14}  {c:8.2f}%  {n:8.2f}%")
        else:
            log(f"    {lo:5d}-{hi:<5d}px : {c:6.2f}%  {'#' * int(c)}")

    below = [lo for lo, _, c, _ in stats if c < 5.0]
    if below and stats and below[0] == stats[0][0]:
        log(f"    -> ceiling is already below 5% in the FIRST ring: this model "
            f"propagates almost")
        log(f"       nothing from a patch this size. A trained reach curve "
            f"far above this is a")
        log(f"       measurement problem, not a result — check that the probe "
            f"and the attack")
        log(f"       composite the same way.")
    elif below:
        log(f"    -> ceiling falls below 5% beyond {below[0]}px")
    else:
        log(f"    -> ceiling stays above 5% out to {stats[-1][1]}px"
            if stats else "    -> no rings measured")
    return ceiling_reach, stats


@torch.no_grad()
def scale_sweep(model, imgs, patch, scales=(0.15, 0.25, 0.35, 0.50),
                n_probes: int = 8, log=print):
    """Does reach GROW with patch size, and how fast?"""
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
        log(f"    scale {s:.2f} (p={patch.side(H, W)}px): "
            + "  ".join(f"{d}px={r:.1f}%" for d, r in row))

    patch.cfg.scale, patch.param.data = original_scale, original_param
    return out


def plot_erf(stats, out_path, title="Empirical receptive field"):
    """stats rows are (lo, hi, ceiling_pct, null_pct)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = [(lo + hi) / 2 for lo, hi, _, _ in stats]
    ceiling = [c for _, _, c, _ in stats]
    null = [n for _, _, _, n in stats]
    split = any(abs(c - n) > 1e-9 for c, n in zip(ceiling, null))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d, ceiling, "-o", color="#c44e52",
            label="ceiling (unconstrained square)" if split else None)
    if split:
        ax.plot(d, null, "-s", color="#4c72b0",
                label="null (random patch at the same tau)")
    ax.axhline(5, color="gray", ls="--", lw=0.8, label="5% threshold")
    ax.set_xlabel("distance from patch centre (px)")
    ax.set_ylabel("prediction-change rate (%)")
    ax.set_title(f"{title}\n"
                 + ("random-patch probe — architectural ceiling vs "
                    "tau-matched null" if split else
                    "unconstrained random square — target-independent ceiling"))
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
