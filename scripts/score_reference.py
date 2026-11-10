#!/usr/bin/env python
"""Score candidate reference images before committing a training run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.patch.lap import asi, agi, ade, edge_mask
from patchreach.patch.shape import load_reference_rgba, derive_shape_mask


def score(path: str, size: int, shape: str, bg: str, thresh: float,
          edge_thresh: float) -> dict:
    rgb, alpha = load_reference_rgba(path, size, torch.device("cpu"))

    try:
        sil = derive_shape_mask(rgb, alpha, shape, bg, thresh, min_frac=0.01)
        sil_err = None
    except ValueError as e:
        sil = torch.ones(size, size, dtype=torch.bool)
        sil_err = str(e).split(".")[0]

    edges = edge_mask(rgb, edge_thresh)
    free = sil & (~edges)

    grey = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).view(1, 1, size, size)
    mean = torch.nn.functional.avg_pool2d(grey, 5, 1, 2)
    var = torch.nn.functional.avg_pool2d(grey ** 2, 5, 1, 2) - mean ** 2
    local_std = var.clamp(min=0).sqrt().view(size, size)
    interior_std = (local_std[free].mean().item() if free.any() else 0.0)

    return {
        "name": Path(path).name,
        "sil_frac": sil.float().mean().item(),
        "edge_frac": edges.float().mean().item(),
        "free_frac": free.float().mean().item(),
        "interior_std": interior_std,
        "ASI": asi(rgb), "AGI": agi(rgb), "ADE": ade(rgb),
        "has_alpha": alpha is not None,
        "warn": sil_err,
    }


def verdict(r: dict) -> str:
    bad = []
    if r["free_frac"] < 0.20:
        bad.append("too constrained (mostly edge/background)")
    if r["interior_std"] < 0.05:
        bad.append("flat interior — nowhere to hide the perturbation")
    if r["sil_frac"] > 0.97 and not r["has_alpha"]:
        bad.append("no separable background — q = M(p) will be the full square")
    if r["ASI"] > 0.6:
        bad.append("very saturated — reads as artificial")
    return "  ".join(f"[!] {b}" for b in bad) if bad else "[ok]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--shape", default="auto",
                    choices=["square", "alpha", "chroma", "auto"])
    ap.add_argument("--shape_bg", default="white", choices=["white", "black"])
    ap.add_argument("--shape_thresh", type=float, default=0.15)
    ap.add_argument("--edge_thresh", type=float, default=0.1)
    a = ap.parse_args()

    rows = [score(p, a.size, a.shape, a.shape_bg, a.shape_thresh,
                  a.edge_thresh) for p in a.images]

    hdr = (f"{'reference':<26}{'sil':>7}{'edge':>7}{'free':>7}"
           f"{'int_std':>9}{'ASI':>7}{'AGI':>7}{'ADE':>7}  alpha")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['sil_frac']:7.2f}{r['edge_frac']:7.2f}"
              f"{r['free_frac']:7.2f}{r['interior_std']:9.3f}"
              f"{r['ASI']:7.3f}{r['AGI']:7.3f}{r['ADE']:7.3f}"
              f"  {'yes' if r['has_alpha'] else 'no'}")
    print()
    print("Tan et al. Fig 2 reference points:")
    print(f"{'  their cartoon (Shaymin)':<26}{'':>7}{'':>7}{'':>7}{'':>9}"
          f"{0.3704:7.3f}{0.3378:7.3f}{0.4056:7.3f}")
    print(f"{'  their TAP (noise patch)':<26}{'':>7}{'':>7}{'':>7}{'':>9}"
          f"{0.6388:7.3f}{0.2183:7.3f}{0.0499:7.3f}")
    print()
    for r in rows:
        v = verdict(r)
        if v != "[ok]" or r["warn"]:
            print(f"{r['name']}: {v}" + (f"  ({r['warn']})" if r["warn"] else ""))
    if all(verdict(r) == "[ok]" for r in rows):
        print("all candidates pass — pick the one with the highest "
              "free_frac x interior_std")


if __name__ == "__main__":
    main()
