#!/usr/bin/env python
r"""
STEP 0 for the universal CSF patch — HOW DARK IS THE GROUND THE PATCH SITS ON?
NO MODEL, NO TRAINING, NO GPU.

    python scripts/measure_footprint_luminance.py \
        --cityscapes_root $CS --img_h 512 --img_w 1024 --split train

WHAT THIS SETTLES
-----------------
The universal patch is bounded by a CSF evaluated at ONE fixed reference
luminance, because a universal patch cannot look at the image it lands on. The
per-image attack that this is the control for CAN. The entire case for that
extra machinery rests on one number: how much does the amplitude budget move
across the luminance range the footprint actually encounters?

If the answer is "barely", the fixed-luminance simplification is a footnote and
the content-adaptive variant has no perceptual justification to stand on — it
would have to earn its place on attack strength alone. If the answer is "a lot",
the fixed reference is systematically wrong on most frames and the spread is
itself the result.

TWO EFFECTS, DECOMPOSED, because they fight each other and a single ratio hides
that:

  MICHELSON DENOMINATOR   contrast is A / Y_ref, so budget is EXACTLY linear in
                          Y_ref. The large term.
  BARTEN SENSITIVITY      CSF(f) shifts with retinal illuminance via the pupil.
                          Flat across the photopic range, and it moves the
                          OTHER way — a darker background is a less sensitive
                          observer, which partly cancels the first effect.

The table below prints both factors separately alongside their product. Read
the product, but quote the decomposition: "the budget moves 7x, of which 21x is
the contrast denominator and 0.34x is Barten pushing back" is a real finding.
"the budget moves 7x" is not.

PLACEMENT IS RESTRICTED TO center/fixed, AND THAT IS A COST, NOT A VERDICT.
Cityscapes has a consistent camera geometry, so fixed pixel coordinates keep the
patch genuinely universal, and the geometry here is resolved through
placement.resolve() — the SAME helper Patch.apply() uses — so the window
measured is the window the attack will occupy, not an approximation of it.

Grad-CAM placement is a legitimate later rung, but it is NOT free here and it
is not free downstream:

  COST      the CAM is per image, so measuring its footprint means a forward
            AND backward pass over the whole split. This script currently needs
            no model, no GPU and no weights; adding gradcam moves it to a GPU
            partition and from minutes to hours.
  MEANING   a universal residual placed at each image's hotspot IS content-
            adaptive — through placement rather than through the residual. That
            is a defensible middle rung between the fixed control and the
            per-image attack, but it is not the non-adaptive control, and
            reporting it as one would give away the comparison.

USE --placement fixed --placement_xy 0.75 0.5 AS THE CHEAP PROXY. The repo
documents the CAM's hottest ridge as the near-field road boundary along the
bottom of a dashcam frame, which is why --placement_margin exists at all. Road
surface straight ahead lands in the same dark asphalt without costing a model
pass, so it brackets what gradcam would measure.

sRGB IS LINEARISED BEFORE WEIGHTING. Averaging gamma-encoded RGB over-states
the luminance of a dark region, which would flatter the fixed-reference
simplification. The legacy mu = 0.5 convention is printed alongside so the size
of the existing calibration error is visible rather than inferred.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_csf_args
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors
from patchreach.patch import csf as csf_mod
from patchreach.patch import placement as placement_mod
from patchreach.utils import seed_everything

# The bands csf.report() prints its budget over. Reused verbatim so this
# script's table can be laid next to that one without re-binning.
BANDS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1),
         (0.1, 0.2), (0.2, 0.35), (0.35, 0.5001))

# Three representative point frequencies, in cycles/pixel: low, mid, and just
# short of Nyquist. Named rather than indexed so the report reads.
PROBE_CPP = (("low", 0.02), ("mid", 0.10), ("near-Nyquist", 0.49))


def build_parser():
    p = argparse.ArgumentParser(
        description="Measure the distribution of mean luminance inside the "
                    "patch footprint, and the CSF budget's sensitivity to it")
    p.add_argument("--cityscapes_root", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="train is correct: the universal patch is fitted on "
                        "train, so L_ref must be chosen from train. Reading it "
                        "off val would be tuning on the reported split.")
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument("--patch_scale", type=float, default=0.25,
                   help="patch side as a fraction of image HEIGHT — the same "
                        "rule Patch.apply() uses, p = int(H*scale)")
    p.add_argument("--placement", default="center", choices=["center", "fixed"],
                   help="gradcam/semantic are absent because they are "
                        "per-image: measuring their footprint needs a model "
                        "pass over the split, and a universal patch placed at "
                        "each image's hotspot is content-adaptive through "
                        "placement. Use 'fixed 0.75 0.5' as the cheap proxy "
                        "for where gradcam tends to land.")
    p.add_argument("--placement_xy", type=float, nargs=2, default=[0.5, 0.5],
                   help="normalised (y, x) CENTRE for --placement fixed. "
                        "(0.75, 0.5) is the road surface straight ahead.")
    p.add_argument("--n_images", type=int, default=0,
                   help="0 = the whole split. Anything else truncates, for a "
                        "smoke test only — a percentile off 50 frames is not "
                        "a distribution.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/diagnostics/footprint_luminance")
    add_csf_args(p)
    p.add_argument("--display_peak_cd_m2", type=float, default=100.0,
                   help="display peak white. Sets the absolute luminance "
                        "Barten's pupil formula needs; 100 is the sRGB "
                        "reference white. Only affects the SENSITIVITY term, "
                        "never the contrast denominator.")
    return p


# ═════════════════════════════════════════════════════════════════════════════
#  Measurement
# ═════════════════════════════════════════════════════════════════════════════

def footprint_luminance(root, split, img_h, img_w, scale, policy, xy,
                        n_images, batch_size, num_workers):
    """
    Per-image mean luminance inside the footprint.

    Returns (Y_linear [N], Y_encoded [N], window, n_total).

    Y_linear   : linearise, then weight by Rec.709. The physically correct one
                 and the only one a Michelson contrast may be divided by.
    Y_encoded  : mean of the gamma-encoded channels — WRONG, and computed only
                 so the report can show how wrong. This is what the existing
                 local_contrast_scale() effectively assumes, and it is the
                 quantity the mu = 0.5 convention is a stand-in for.
    """
    ds = CityscapesSeg(root, split, img_h, img_w)
    n_total = len(ds)
    if n_images and n_images < n_total:
        ds = Subset(ds, list(range(n_images)))

    p = int(img_h * scale)
    top, left = placement_mod.resolve(policy, img_h, img_w, p, xy=tuple(xy))
    window = {"top": top, "left": left, "size": p}

    mean_t, std_t = norm_tensors(torch.device("cpu"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    lin, enc = [], []
    for img, _ in tqdm(loader, desc=f"{split} footprint"):
        # Undo the ImageNet normalisation to recover sRGB code values. The
        # clamp matters: bilinear resize can overshoot slightly outside [0,1]
        # and srgb_to_linear is only defined on the unit interval.
        rgb = (img * std_t + mean_t).clamp(0, 1)
        crop = rgb[:, :, top:top + p, left:left + p]
        lin.append(csf_mod.relative_luminance(crop).mean(dim=(-2, -1)))
        enc.append(crop.mean(dim=(1, 2, 3)))
    return torch.cat(lin), torch.cat(enc), window, n_total


def distribution(x: torch.Tensor) -> dict:
    q = torch.tensor([0.05, 0.5, 0.95])
    p5, med, p95 = torch.quantile(x.float(), q).tolist()
    return {"n": int(x.numel()), "mean": float(x.mean()), "median": med,
            "p5": p5, "p95": p95, "min": float(x.min()), "max": float(x.max()),
            "std": float(x.std())}


# ═════════════════════════════════════════════════════════════════════════════
#  Budget sensitivity
# ═════════════════════════════════════════════════════════════════════════════

def band_budget(size, Y_ref, geometry, model, tau, min_cycles):
    """Mean budget and mean CSF per radial band, at background luminance Y_ref."""
    csf, budget = csf_mod.patch_budget_at_luminance(
        size, Y_ref, geometry, model, tau, min_cycles, units="srgb")
    f = csf_mod.radial_frequency_cpp(size, size)
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        if not bool(m.any()):
            continue
        out[f"{lo:.2f}-{hi:.2f}"] = {"csf": float(csf[m].mean()),
                                     "budget": float(budget[m].mean())}
    return out


def probe_budget(Y_ref, geometry, model, tau):
    r"""
    Budget at the three probe frequencies, DECOMPOSED.

    Evaluated pointwise rather than by band so the two effects stay separable:

        budget(f; Y) = tau * Y / (2 * CSF(f; Y))     [linear luminance units]

    The luminance factor is the Y in the numerator, the CSF factor is the
    CSF(f; Y) in the denominator, and only their product is the budget. A
    band-averaged budget cannot be decomposed this way because the two factors
    average differently.
    """
    f_cpd = geometry.to_cycles_per_degree(
        torch.tensor([v for _, v in PROBE_CPP]))
    params, L_abs = csf_mod.barten_params_at_luminance(Y_ref, geometry)
    csf = (csf_mod.barten_csf(f_cpd, params) if model == "barten"
           else csf_mod.sso_csf(f_cpd, torch.zeros_like(f_cpd)))
    lin = tau * Y_ref / (2.0 * csf.clamp(min=1e-12))
    c_ref = csf_mod.linear_to_srgb(torch.tensor(float(Y_ref)))
    srgb = lin / csf_mod.srgb_slope(c_ref).clamp(min=1e-6)
    return {"Y_ref": float(Y_ref), "L_cd_m2": L_abs,
            "pupil_mm": csf_mod.pupil_diameter_mm(L_abs),
            "E_troland": params.E, "srgb_slope": float(csf_mod.srgb_slope(c_ref)),
            "probes": {name: {"f_cpp": v, "f_cpd": float(f_cpd[i]),
                              "csf": float(csf[i]),
                              "budget_linear": float(lin[i]),
                              "budget_srgb": float(srgb[i])}
                       for i, (name, v) in enumerate(PROBE_CPP)}}


# ═════════════════════════════════════════════════════════════════════════════
#  Reporting
# ═════════════════════════════════════════════════════════════════════════════

def plot_histogram(y_lin, y_enc, stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(y_lin.numpy(), bins=60, color="#3b6ea5")
    for k, c in (("p5", "#c44"), ("median", "#222"), ("p95", "#c44")):
        ax[0].axvline(stats[k], color=c, ls="--", lw=1)
    ax[0].set_xlabel("mean relative luminance Y in footprint (linear)")
    ax[0].set_ylabel("images")
    ax[0].set_title(f"n = {stats['n']}   median {stats['median']:.4f}   "
                    f"5-95 pct {stats['p5']:.4f}-{stats['p95']:.4f}")

    ax[1].hist(y_enc.numpy(), bins=60, color="#a5713b")
    ax[1].axvline(0.5, color="#c44", ls="--", lw=1.5)
    ax[1].set_xlabel("mean sRGB code value in footprint (gamma-encoded)")
    ax[1].set_title("dashed line = the mu = 0.5 the legacy budget assumes")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    a = build_parser().parse_args()
    seed_everything(a.seed)
    geometry = csf_mod.ViewingGeometry(a.csf_pixel_size_cm,
                                       a.csf_viewing_distance_cm,
                                       a.display_peak_cd_m2)

    y_lin, y_enc, window, n_total = footprint_luminance(
        a.cityscapes_root, a.split, a.img_h, a.img_w, a.patch_scale,
        a.placement, a.placement_xy, a.n_images, a.batch_size, a.num_workers)

    lin_stats, enc_stats = distribution(y_lin), distribution(y_enc)
    size = window["size"]

    print(f"\n{'=' * 74}")
    print(" 1. FOOTPRINT LUMINANCE — what the patch actually lands on")
    print(f"{'=' * 74}")
    geometry.describe()
    print(f"[foot] window    : {size}x{size} at (top {window['top']}, "
          f"left {window['left']}) of {a.img_h}x{a.img_w}, "
          f"placement={a.placement}")
    print(f"[foot] split     : {a.split}  ({lin_stats['n']} of {n_total} images)")
    print(f"\n  {'':<10}{'median':>10}{'mean':>10}{'5 pct':>10}"
          f"{'95 pct':>10}{'min':>10}{'max':>10}")
    for name, s in (("Y linear", lin_stats), ("sRGB code", enc_stats)):
        print(f"  {name:<10}{s['median']:>10.4f}{s['mean']:>10.4f}"
              f"{s['p5']:>10.4f}{s['p95']:>10.4f}{s['min']:>10.4f}"
              f"{s['max']:>10.4f}")
    print(f"\n  spread p95/p5 : {lin_stats['p95'] / max(lin_stats['p5'], 1e-9):.2f}x "
          f"in linear luminance")
    print(f"  legacy mu=0.5 vs measured median code "
          f"{enc_stats['median']:.4f} -> the existing budget over-states the "
          f"contrast denominator by {0.5 / max(enc_stats['median'], 1e-9):.2f}x")

    print(f"\n{'=' * 74}")
    print(" 2. BUDGET SENSITIVITY — does the fixed reference actually matter?")
    print(f"{'=' * 74}")
    refs = {"p5": lin_stats["p5"], "median": lin_stats["median"],
            "p95": lin_stats["p95"]}
    probes = {k: probe_budget(v, geometry, a.csf_model, a.csf_threshold)
              for k, v in refs.items()}
    bands = {k: band_budget(size, v, geometry, a.csf_model,
                            a.csf_threshold, a.csf_min_cycles)
             for k, v in refs.items()}

    print(f"\n  observer state at each reference (tau = {a.csf_threshold:g}):")
    print(f"  {'ref':<8}{'Y':>9}{'cd/m2':>9}{'pupil mm':>10}{'E (Td)':>10}")
    for k, pr in probes.items():
        print(f"  {k:<8}{pr['Y_ref']:>9.4f}{pr['L_cd_m2']:>9.2f}"
              f"{pr['pupil_mm']:>10.2f}{pr['E_troland']:>10.1f}")
    if probes["p5"]["L_cd_m2"] < csf_mod.PHOTOPIC_FLOOR_CD_M2:
        print(f"  ! p5 sits at {probes['p5']['L_cd_m2']:.2f} cd/m2, below the "
              f"photopic floor Barten's parameters are fitted for. The number "
              f"is an extrapolation; say so when quoting it.")

    print(f"\n  per-frequency amplitude budget (sRGB code units):")
    print(f"  {'probe':<14}{'cpd':>8}{'B(p5)':>12}{'B(median)':>12}"
          f"{'B(p95)':>12}{'p95/p5':>9}")
    ratios = {}
    for name, _ in PROBE_CPP:
        lo = probes["p5"]["probes"][name]
        md = probes["median"]["probes"][name]
        hi = probes["p95"]["probes"][name]
        r = hi["budget_srgb"] / max(lo["budget_srgb"], 1e-30)
        ratios[name] = r
        print(f"  {name:<14}{lo['f_cpd']:>8.1f}{lo['budget_srgb']:>12.3e}"
              f"{md['budget_srgb']:>12.3e}{hi['budget_srgb']:>12.3e}{r:>9.2f}x")

    print(f"\n  decomposition of that p95/p5 ratio:")
    print(f"  {'probe':<14}{'total':>9}{'luminance':>12}{'Barten CSF':>12}")
    lum_factor = refs["p95"] / max(refs["p5"], 1e-30)
    slope_factor = (float(csf_mod.srgb_slope(csf_mod.linear_to_srgb(
                        torch.tensor(refs["p5"]))))
                    / max(float(csf_mod.srgb_slope(csf_mod.linear_to_srgb(
                        torch.tensor(refs["p95"])))), 1e-30))
    for name, _ in PROBE_CPP:
        csf_factor = (probes["p5"]["probes"][name]["csf"]
                      / max(probes["p95"]["probes"][name]["csf"], 1e-30))
        print(f"  {name:<14}{ratios[name]:>9.2f}{lum_factor:>12.2f}"
              f"{csf_factor:>12.2f}")
    print(f"  (a further {slope_factor:.2f}x comes from the sRGB transfer "
          f"slope, which is a unit conversion and not a perceptual effect)")

    print(f"\n  band table at L_ref = median (compare against csf.report()):")
    print(f"  {'band cyc/px':<14}{'CSF':>10}{'budget':>12}")
    for band, v in bands["median"].items():
        print(f"  {band:<14}{v['csf']:>10.2f}{v['budget']:>12.4e}")

    out_dir = Path(a.out_dir) / f"{a.split}_{a.img_h}x{a.img_w}_{a.placement}_p{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_histogram(y_lin, y_enc, lin_stats, out_dir / "footprint_luminance.png")
    with open(out_dir / "footprint_luminance.json", "w") as f:
        json.dump({"split": a.split, "img_h": a.img_h, "img_w": a.img_w,
                   "patch_scale": a.patch_scale, "window": window,
                   "placement": a.placement, "placement_xy": a.placement_xy,
                   "n_images": lin_stats["n"], "n_total": n_total,
                   "geometry": {"pixel_size_cm": geometry.pixel_size_cm,
                                "viewing_distance_cm": geometry.viewing_distance_cm,
                                "display_peak_cd_m2": geometry.display_peak_cd_m2,
                                "degrees_per_pixel": geometry.degrees_per_pixel,
                                "nyquist_cpd": geometry.nyquist_cpd},
                   "csf_model": a.csf_model, "tau": a.csf_threshold,
                   "csf_min_cycles": a.csf_min_cycles,
                   "luminance_linear": lin_stats,
                   "luminance_encoded": enc_stats,
                   "references": refs, "probes": probes, "bands": bands,
                   "ratios_p95_over_p5": ratios,
                   "luminance_factor": lum_factor,
                   "srgb_slope_factor": slope_factor,
                   "per_image_Y_linear": y_lin.tolist()}, f, indent=2)
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()
