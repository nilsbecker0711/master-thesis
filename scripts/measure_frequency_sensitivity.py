#!/usr/bin/env python
"""Frequency-resolved sensitivity probe — WHICH BANDS DOES THE NETWORK READ?"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (add_model_args, add_patch_args, setup_model, build_patch,
                     image_indices)
from _common import make_dataset
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import spectral
from patchreach.patch import csf as csf_mod
from patchreach.utils import get_device, seed_everything


def build_parser():
    p = add_patch_args(add_model_args(argparse.ArgumentParser(
        description="Measure segmentation sensitivity per spatial-frequency "
                    "band, at equal perceptual cost")))
    p.add_argument("--images", default="fixed10",
                   help="'fixed10' | 'all' | '2 5 45'")
    p.add_argument("--n_images", type=int, default=5,
                   help="how many of --images to actually use")
    p.add_argument("--n_probes", type=int, default=8,
                   help="noise draws per band per image")
    p.add_argument("--region", default="patch",
                   choices=list(spectral.REGION_MODES),
                   help="'patch' is the threat model and goes through the same "
                        "compositing as the real attack; 'full' is the cleaner "
                        "instrument because no receptive-field ceiling "
                        "confounds the band response. Run both.")
    p.add_argument("--normalise", default="visibility",
                   choices=list(spectral.NORMALISE_MODES),
                   help="'visibility' equalises perceptual cost across bands "
                        "(the control that makes bands comparable); 'rms' "
                        "equalises raw amplitude and is the control for it.")
    p.add_argument("--contrast_mean", default="local",
                   choices=list(spectral.CONTRAST_MEAN_MODES),
                   help="'local' measures the mean luminance of the region "
                        "(correct, and required for equal-cost to mean "
                        "anything); 'fixed' reproduces the mu=0.5 convention "
                        "the attack modes currently use.")
    p.add_argument("--target", type=float, default=None,
                   help="cost each band is normalised to. Defaults to "
                        "--csf_threshold for 'visibility', 0.02 for 'rms'.")
    p.add_argument("--min_signal", type=float, default=0.5,
                   help="floor, in %% of remote pixels, below which the run is "
                        "declared INCONCLUSIVE and no gap / scene-dependence "
                        "claim is made. Guards against reading an argmax out "
                        "of six near-zero rates.")
    p.add_argument("--n_bands", type=int, default=0,
                   help="0 uses spectral.DEFAULT_BANDS, which ARE the bands "
                        "csf.report() prints its budget over (csf.RADIAL_BANDS). "
                        ">0 replaces them with that many equal-width bands over "
                        "[0, 0.7071]. The ceiling is sqrt(2)/2 and not 0.5: 0.5 "
                        "is the per-axis Nyquist, while the radial frequency is "
                        "the length of (fy, fx) and reaches 0.707 in the "
                        "diagonal corner -- which is where the budget spends "
                        "92% of its power.")
    p.add_argument("--out_dir", default="results/spectral")
    return p


def main():
    a = build_parser().parse_args()
    target = a.target if a.target is not None else (
        a.csf_threshold if a.normalise == "visibility" else 0.02)

    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    ds = make_dataset(a, "val")
    patch = build_patch(a, device, mean_t, std_t)

    if a.n_bands > 0:
        edges = torch.linspace(0.0, csf_mod.MAX_CPP,
                               a.n_bands + 1).tolist()
        bands = list(zip(edges[:-1], edges[1:]))
    else:
        bands = list(spectral.DEFAULT_BANDS)

    geometry = csf_mod.ViewingGeometry(a.csf_pixel_size_cm,
                                       a.csf_viewing_distance_cm)

    print(f"\n{'=' * 72}")
    print(" WHAT THE EYE ALLOWS — amplitude budget by band")
    print(f"{'=' * 72}")
    header_tau = target if a.normalise == "visibility" else a.csf_threshold
    csf_report = csf_mod.report(a.patch_size, a.patch_size, geometry,
                                a.csf_model, header_tau, a.csf_min_cycles)

    print(f"\n{'=' * 72}")
    print(f" WHAT THE NETWORK READS — {a.arch} ({spec.bracket} attention)")
    print(f"{'=' * 72}")

    idxs = image_indices(a.images, len(ds))[:a.n_images]
    per_image = []
    for i in idxs:
        img = ds[i][0].unsqueeze(0).to(device)
        print(f"\n--- image {i} ---")
        with torch.no_grad():
            clean_logits = upsample_to(model(img), (a.img_h, a.img_w))
        patch.resolve_placement(a.img_h, a.img_w, clean_logits.argmax(1)[0])
        per_image.append(spectral.frequency_sensitivity(
            model, img, patch, mean_t, std_t, bands=bands, target=target,
            n_probes=a.n_probes, region=a.region, normalise=a.normalise,
            contrast_mean=a.contrast_mean, csf_model=a.csf_model,
            geometry=geometry, beta=a.csf_beta))

    summary = spectral.summarise(per_image, target, a.normalise, a.min_signal)

    out_dir = (Path(a.out_dir)
               / f"{a.arch}_{a.img_h}x{a.img_w}_{a.region}_{a.normalise}")
    out_dir.mkdir(parents=True, exist_ok=True)
    spectral.plot_bands(summary, out_dir / "frequency_sensitivity.png",
                        title=f"{a.arch} ({spec.bracket} attention)",
                        target=target)
    with open(out_dir / "frequency_sensitivity.json", "w") as f:
        json.dump({"arch": a.arch, "bracket": spec.bracket,
                   "img_h": a.img_h, "img_w": a.img_w,
                   "patch_size": a.patch_size, "patch_scale": a.patch_scale,
                   "placement": a.placement,
                   "region": a.region, "normalise": a.normalise,
                   "contrast_mean": a.contrast_mean, "target": target,
                   "min_signal": a.min_signal,
                   "n_images": len(per_image), "n_probes": a.n_probes,
                   "images": idxs,
                   "csf": csf_report,
                   "summary": summary,
                   "per_image": per_image}, f, indent=2)
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()
