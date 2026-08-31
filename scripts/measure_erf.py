#!/usr/bin/env python
r"""
Block 0.2 — effective receptive field across architectures. NO TRAINING.

    python scripts/measure_erf.py --arch segformer_b0 \
        --cityscapes_root $CS --img_h 512 --img_w 1024 --n_images 5

THIS IS A RESULT, NOT SETUP. It independently replicates the ERF-vulnerability
relationship using a different instrument than the source paper: they
backpropagate from a central output unit, this feeds maximal random stimulus
and measures prediction change. Two methods agreeing is far stronger than
citing theirs.

LABEL-FREE AND DATASET-INDEPENDENT: the probe never touches ground truth and
never references a class. So ADE20K weights are perfectly valid here — you can
place Swin in the three-bracket comparison without Cityscapes weights for it.

PREDICTION: ERF should order by attention mechanism — deformable-conv narrowest,
windowed attention middle, global attention widest. If it does not, the
geometric factor needs rethinking, and better to know that on day two.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import add_model_args, add_patch_args, setup_model, build_patch
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors
from patchreach.diagnostics import geometric
from patchreach.utils import get_device, seed_everything


def main():
    p = add_patch_args(add_model_args(argparse.ArgumentParser()))
    p.add_argument("--n_images", type=int, default=5)
    p.add_argument("--n_probes", type=int, default=16)
    p.add_argument("--sweep", action="store_true",
                   help="also run the patch-scale sweep")
    p.add_argument("--out_dir", default="results/erf")
    a = p.parse_args()

    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    patch = build_patch(a, device, mean_t, std_t)

    out_dir = Path(a.out_dir) / f"{a.arch}_{a.img_h}x{a.img_w}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats, sweeps = [], {}
    for i in range(min(a.n_images, len(ds))):
        img = ds[i][0].unsqueeze(0).to(device)
        print(f"\n--- image {i} ---")
        _, stats = geometric.receptive_field(model, img, patch, a.n_probes)
        all_stats.append(stats)
        if a.sweep and i == 0:
            sweeps = geometric.scale_sweep(model, img, patch)

    # average the rings across images — ERF is a model property, so the
    # image-to-image spread should be small. If it is not, say so.
    rings = all_stats[0]
    mean_stats = []
    print(f"\n{'='*66}")
    print(f" {a.arch} ({spec.bracket} attention) — ERF averaged over "
          f"{len(all_stats)} images")
    print(f"{'='*66}")
    for k, (lo, hi, _, _) in enumerate(rings):
        vals = torch.tensor([s[k][2] for s in all_stats if k < len(s)])
        nulls = torch.tensor([s[k][3] for s in all_stats if k < len(s)])
        mean_stats.append((lo, hi, float(vals.mean()), float(nulls.mean())))
        print(f"    {lo:5d}-{hi:<5d}px : {vals.mean():6.2f}% "
              f"+/- {vals.std():.2f}  (null {nulls.mean():5.2f}%)  "
              f"{'#' * int(vals.mean())}")
    below = [lo for lo, hi, r, _ in mean_stats if r < 5.0]
    if below:
        print(f"    -> below 5% at ~{below[0]}px")

    geometric.plot_erf(mean_stats, out_dir / "erf.png",
                       title=f"{a.arch} ({spec.bracket} attention)")
    with open(out_dir / "erf.json", "w") as f:
        json.dump({"arch": a.arch, "bracket": spec.bracket,
                   "img_h": a.img_h, "img_w": a.img_w,
                   "n_images": len(all_stats), "n_probes": a.n_probes,
                   "rings": [{"lo": lo, "hi": hi, "rate": r,
                              "null_rate": n}
                             for lo, hi, r, n in mean_stats],
                   "collapse_px": below[0] if below else None,
                   "sweep": {str(k): v for k, v in sweeps.items()}}, f, indent=2)
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()