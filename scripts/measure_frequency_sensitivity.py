#!/usr/bin/env python
r"""
Frequency-resolved sensitivity probe — WHICH BANDS DOES THE NETWORK READ?
NO TRAINING.

    python scripts/measure_frequency_sensitivity.py --arch segformer_b0 \
        --cityscapes_root $CS --img_h 512 --img_w 1024 --n_images 5

THIS IS A RESULT, NOT SETUP, and it tests the premise the whole CSF attack
family rests on. csf.py bounds a perturbation by what the EYE can see and
assumes the NETWORK still reads it. That assumption has never been measured,
and csf.py's own header names the way it could fail: SegFormer's patch
embedding has stride 4, so the highest frequencies may be attenuated before the
first attention block ever sees them.

The instrument is measure_erf.py's, moved from the spatial domain to the
frequency domain: maximal random stimulus, prediction change, no ground truth
and no target class. LABEL-FREE AND DATASET-INDEPENDENT, so ADE20K weights are
valid here exactly as they are for the ERF probe — which is what makes the
three-bracket comparison possible without Cityscapes weights for every arch.

THE CONTROL: every band is normalised to the same perceptual cost before it is
injected, so the question is "per unit of visibility, which band buys the most
prediction change?" rather than the uninterpretable equal-amplitude version.
Run --normalise rms to get the equal-amplitude control alongside it.

READ THE TWO TABLES TOGETHER. The run prints csf.report()'s amplitude budget by
band, then the sensitivity by the SAME bands. The exploitable gap is where a
large budget meets a high flip rate. A large budget meeting a dead band is the
premise failing.

PREDICTION: if the CSF premise holds, flip rate should stay substantial toward
Nyquist, where the budget is orders of magnitude larger. If it collapses there,
the attack is bounded by the backbone's input stride rather than by tau, and
better to know that before the tau ladder is run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (add_model_args, add_patch_args, add_image_args,
                     setup_model, build_patch, resolve_images)
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import spectral
from patchreach.patch import csf as csf_mod
from patchreach.utils import get_device, seed_everything


def build_parser():
    p = add_patch_args(add_model_args(argparse.ArgumentParser(
        description="Measure segmentation sensitivity per spatial-frequency "
                    "band, at equal perceptual cost")))
    add_image_args(p, default_images="fixed10", default_n=5,
                   n_help="Each image costs n_probes forward passes per band, "
                          "so this is the run's cost knob.")
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
                   help="0 uses spectral.DEFAULT_BANDS, which match the bands "
                        "csf.report() prints its budget over. >0 replaces them "
                        "with that many equal-width bands over [0, 0.5].")
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

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    patch = build_patch(a, device, mean_t, std_t)

    if a.n_bands > 0:
        edges = torch.linspace(0.0, 0.5, a.n_bands + 1).tolist()
        bands = list(zip(edges[:-1], edges[1:]))
    else:
        bands = list(spectral.DEFAULT_BANDS)

    geometry = csf_mod.ViewingGeometry(a.csf_pixel_size_cm,
                                       a.csf_viewing_distance_cm)

    # The budget table FIRST, over a patch-sized grid, so the sensitivity table
    # below can be read directly against it. A large budget meeting a dead band
    # is the premise failing; a large budget meeting a live band is the gap.
    print(f"\n{'=' * 72}")
    print(" WHAT THE EYE ALLOWS — amplitude budget by band")
    print(f"{'=' * 72}")
    # The table must describe THIS run. a.csf_threshold is the attack modes'
    # default and is unrelated to --target, so printing it made all four rungs
    # of a tau ladder show an identical header reading "tau = 0.25".
    header_tau = target if a.normalise == "visibility" else a.csf_threshold
    csf_report = csf_mod.report(a.patch_size, a.patch_size, geometry,
                                a.csf_model, header_tau, a.csf_min_cycles)

    print(f"\n{'=' * 72}")
    print(f" WHAT THE NETWORK READS — {a.arch} ({spec.bracket} attention)")
    print(f"{'=' * 72}")

    # Through the shared resolver, so --images random draws n images at
    # random instead of truncating a shuffled list back to its lowest indices.
    idxs = resolve_images(a.images, len(ds), a.n_images, a.sample_seed,
                          a.exclude_image)
    per_image = []
    for i in idxs:
        img = ds[i][0].unsqueeze(0).to(device)
        print(f"\n--- image {i} ---")
        # ORDERING: clean forward -> resolve_placement -> probe. Semantic
        # placement reads the CLEAN PREDICTION; backwards it silently uses
        # centre. train.py documents the same sequence.
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
