#!/usr/bin/env python
r"""
Perceptual evaluation of saved patches — how visible is the frame's change?

    conda activate lpips_eval
    python scripts/lpips_eval.py results/overfit/segformer_csf_* \
        --metrics all --anchors --tag png --out_csv perc_csf.csv

RUNS IN ITS OWN ENVIRONMENT, ON PURPOSE. The attack env is pinned to torch 1.11
+ mmcv 1.x and the metric packages do not install cleanly into it, so this
script never touches a segmentor. It imports only the model-free half of
patchreach (patch/spec.py, patch/csf.py, data/cityscapes.py: torch +
torchvision + numpy), rebuilds the clean and the patched frame from what the
attack scripts already saved, and scores the pair. No mmseg, no GPU required.

METRICS (--metrics, 'all' for every one below). The FIRST name is the primary:
it drives the console line and the panels, and it should stay lpips_alex
unless there is a reason, because that is the number the patch-attack
literature reports (IAP, ICCV 2025, quotes LPIPS over the patch region).

    lpips_alex/vgg/squeeze  the reference `lpips` package (Zhang et al. 2018)
    dists, stlpips, lpips+  newer full-reference deep metrics via pyiqa;
    ahiq, topiq_fr, pieapp  DISTS tolerates texture resampling where LPIPS
    ssim, ms_ssim, fsim     does not, which matters here because the pipeline
                            blur and the residual are different in exactly
                            that way. ahiq and topiq_fr read HIGHER = better.
    cvvdp                   ColorVideoVDP: the only metric here that takes the
                            VIEWING CONDITIONS, built from the run's own
                            csf_pixel_size_cm / csf_viewing_distance_cm, so it
                            and the tau budget describe ONE observer. JOD,
                            10 = indistinguishable, HIGHER = better.
    niqe, brisque,          no-reference: naturalness proper, scored on each
    clipiqa, musiq          frame alone. Reported as _clean / _patched /
                            _delta, because only the delta is about the patch.

A metric that fails on one region (a transformer out of memory on the full
1024x2048 frame is the realistic case) is dropped with a message; the rest of
the evaluation continues.

OUTPUT: <run>/perc_eval/[<tag>/] holds every evaluation of that run —
per_image.csv, summary.json (with a lower_better map, so no table can rank
cvvdp or ahiq upside down), panels and per-view PNGs.

TWO SOURCES (--source, default auto = png where the run has them):

    png         the a_clean.png / c_patched.png the attack scripts already
                wrote (diagnostics/panels/, panels/img*/). Exactly the frames
                the segmentor saw, lossless 8-bit. No checkpoint, no dataset,
                no --cityscapes_root. Only images the run wrote diagnostics
                for, and no CSF visibility columns.
    checkpoint  rebuild the frames from final/best.pt + the val image (below).
                Every image, plus the visibility columns from Patch.stats().

Both feed the same scoring, and a test holds them equal on the same patch.

WHICH PATCH THE PNGs SHOW is checked, not assumed: overfit.py wrote the final
patch into its diagnostics before 2aee5ef and best.pt since. Each checkpoint
is re-rendered onto a_clean.png and compared with c_patched.png; the result
is the png_shows column (best / final / best=final / unknown). Under --source
auto, an explicit --checkpoint best|final the PNGs do not show switches that
run to the checkpoint source.

WHAT THE CHECKPOINT SOURCE READS — every layout the attack scripts write:

    overfit.py            <run>/config.json + <run>/{final,best}.pt
                           <run>/seed*/{final,best}.pt     (--seeds > 1)
    overfit_population.py  <run>/config.json + <run>/patches/img%04d/best.pt
    train.py               <run>/config.json + <run>/{final,best}.pt
                           ONE shared patch, scored on every image in --images

The reference that mode='csf' sits on is NOT in the checkpoint (Patch.save
stores the parameter, the config and the placement), so for --from_image runs
it is re-derived from the same val image and placement, exactly as
overfit_population.py does before its diagnostics. Skip that and render()
falls back to 0.5 grey — an opaque grey square scored as a "CSF patch".

THREE REGIONS PER METRIC, because one of them alone misleads. Columns are
<metric>_<region>, e.g. lpips_alex_crop, dists_full, cvvdp_ctx:

    _crop        clean vs patched, the p x p FOOTPRINT only.  <- the result,
                 and the quantity the patch literature reports
    _ctx         the footprint plus a margin (--context x p per side), so a
                 visible seam between patch and surround is counted too.
    _full        the whole frame. These metrics are spatial MEANS, so a 256 px
                 patch in a 1024x2048 frame is diluted ~32x. Quote it only
                 against things that cover the SAME area (see ANCHORS).

and two controls, csf modes only, each on crop / ctx / full:

    <metric>_floor_*  clean vs the ZERO-RESIDUAL composite: the image's own crop
                   pasted back as the patch, zero optimisation -- the base
                   downsampled to --patch_size and resized back up to p.
                   Nonzero whenever patch_size != int(img_h * patch_scale),
                   and then part of lpips_* is resampling blur rather than the
                   residual. If floor is a large fraction of the patched
                   number, that number is about the pipeline, not the attack.

                   NOT the same as an overfit.py run with --steps 0: csf
                   initialises the residual AT the tau budget (randn * 10,
                   projected), so step 0 is random noise at tau, not the crop.

    <metric>_resid_*  clean vs clean + (patched - base): the residual exactly as
                   pasted, on the SHARP original. Removes the blur without
                   rerunning the attack -- what the CSF residual alone costs.

ANCHORS (--anchors). LPIPS has no absolute threshold, so the script can score
known reference perturbations on the same frame with the same nets:

    <metric>_anchor_linf{e}_global  random +-e/255 sign noise on the whole
                   frame. e = 8 is the conventional "imperceptible" L_inf
                   budget, and Laidlaw et al. (ICLR 2021) measured human
                   perceptibility of L_inf 8/255 attacks directly; this puts
                   that budget on YOUR LPIPS scale, for YOUR image.
    <metric>_anchor_linf{e}_patch   the same noise confined to the footprint:
                   the local counterpart, diluted exactly like the patch is.

QUANTISATION. The patched frame is rounded to 8 bit before scoring (turn off
with --no_quantize). That is what a display shows and what a saved PNG holds —
and near threshold a CSF residual is only a few code values, so a float-only
score would credit the attack with sub-quantum structure no observer can ever
receive. `frac_px_changed` records how much of the footprint survives.

NONE OF THESE IS A HUMAN STUDY. They are learned or modelled distances, and
they are an independent check on the CSF budget precisely because nothing in
the attack optimised against them. A low value says "close to the original
under this metric"; it does not say "invisible". Put them next to the realised
visibility columns, not in place of them — and score a raw (opaque) patch run
through this same script so the csf numbers have something to be low relative
to.

RUNNING MANY METRICS IS NOT FREE OF RISK: picking the friendliest one after
seeing the numbers is metric shopping. Fix the primary metric first, report
the rest as a robustness check, and say in the text that all of them were
computed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.data.cityscapes import CityscapesSeg, norm_tensors
from patchreach.patch.placement import footprint_side

# Kept in sync with scripts/_common.py by hand. _common is not imported
# because it pulls in the model registry, and with it mmseg — which is the
# whole reason this script has its own environment.
FIXED10 = [430, 46, 441, 331, 45, 162, 195, 424, 91, 353]

# CSF visibility columns copied from Patch.stats(), so LPIPS can be read
# against the budget the patch was trained under.
VIS_KEYS = ("visibility", "visibility_local", "realised_visibility",
            "resid_rms")


# ── discovery ────────────────────────────────────────────────────────────────

@dataclass
class Job:
    run: Path             # the run directory (holds config.json)
    ckpt: Path
    image: int            # val index
    label: str            # seed / image tag inside the run


def _pick_ckpt(d: Path, which: str) -> Optional[Path]:
    order = {"final": ["final.pt"], "best": ["best.pt"],
             "auto": ["final.pt", "best.pt"]}[which]
    for name in order:
        if (d / name).exists():
            return d / name
    return None


def discover(run: Path, which: str, images: List[int]) -> List[Job]:
    """
    One Job per (checkpoint, image) pair the run defines.

    overfit_population.py only saves best.pt per image (save_best=True, no
    final), so --checkpoint final finds nothing there; 'auto' falls back.
    """
    run = run.resolve()
    cfg = json.loads((run / "config.json").read_text())

    # population: one patch per image, each under patches/img%04d/
    pop = sorted(p for p in (run / "patches").glob("img*") if p.is_dir())
    if pop:
        jobs = []
        for d in pop:
            ck = _pick_ckpt(d, which)
            if ck is not None:
                jobs.append(Job(run, ck, int(d.name[3:]), d.name))
        return jobs

    # overfit.py: ONE image, recorded in config.json. Repeats nest by seed.
    if "image" in cfg and cfg.get("image") is not None and "epochs" not in cfg:
        dirs = sorted(run.glob("seed*")) or [run]
        jobs = []
        for d in dirs:
            ck = _pick_ckpt(d, which)
            if ck is not None:
                jobs.append(Job(run, ck, int(cfg["image"]),
                                d.name if d != run else "run"))
        return jobs

    # train.py: ONE shared patch, scored across the requested val images.
    ck = _pick_ckpt(run, which)
    return [Job(run, ck, i, f"img{i:04d}") for i in images] if ck else []


def parse_images(arg: str, n_images: int) -> List[int]:
    if arg == "fixed10":
        idx = FIXED10
    elif arg.startswith("first"):
        idx = list(range(int(arg[5:])))
    else:
        idx = [int(x) for x in arg.replace(",", " ").split()]
    return idx[:n_images] if n_images > 0 else idx


# ── geometry ─────────────────────────────────────────────────────────────────

def footprint_box(placement, H: int, W: int, scale: float,
                  ref: str = "height"):
    """
    (top, left, p) exactly as Patch.apply() resolves it: centred when no
    placement was stored, then clamped inside the frame.

    p comes from placement.footprint_side, the rule Patch.apply() itself uses.
    A private int(H*scale) here would crop the height-rule box around a patch
    pasted at the area-rule size (--patch_scale_ref area), so LPIPS would score
    the wrong region on every non-square input without any error.
    """
    p = footprint_side(H, W, scale, ref)
    top, left = (placement if placement is not None
                 else ((H - p) // 2, (W - p) // 2))
    top = max(0, min(int(top), H - p))
    left = max(0, min(int(left), W - p))
    return top, left, p


def expand_box(top: int, left: int, p: int, margin: int, H: int, W: int):
    """(t, b, l, r) of the footprint grown by `margin`, clipped to the frame."""
    return (max(0, top - margin), min(H, top + p + margin),
            max(0, left - margin), min(W, left + p + margin))


def quantize8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255.0).round() / 255.0


# ── one image ────────────────────────────────────────────────────────────────

def composite(patch, img_norm, mean_t, std_t, quantize: bool,
              base_only: bool = False) -> torch.Tensor:
    """
    patch.apply() on one image, returned in [0,1].

    base_only swaps render() for the bare reference: the same interpolate and
    paste, with the residual removed. That is the lpips_floor control — every
    step of the composite except the attack.
    """
    if base_only:
        ref = patch.reference.clamp(0, 1)
        patch.render = lambda: ref          # instance attribute shadows method
    try:
        with torch.no_grad():
            patched, _ = patch.apply(img_norm)
    finally:
        if base_only:
            del patch.render
    out = (patched * std_t + mean_t).clamp(0, 1)
    return quantize8(out) if quantize else out


@dataclass
class Metric:
    r"""
    One perceptual metric, normalised to a single call signature.

    kind='fr'  fn(ref, dist) -> float, both [1,3,H,W] in [0,1]: how far the
               MODIFIED frame (dist) is from the clean one (ref). The order
               matters: pyiqa and cvvdp both take the distorted image FIRST,
               and ahiq/topiq_fr/pieapp are not symmetric, so the wrapper --
               not the caller -- is where the two conventions meet.
    kind='nr'  fn(a) -> float: a no-reference score of ONE image. These answer
               a different question -- "does this look like a photograph?" --
               and are reported for clean and patched separately, because only
               their DIFFERENCE is about the patch.
    """
    name: str
    fn: object
    kind: str = "fr"
    lower_better: bool = True
    # Sizes this metric has already failed at. PER SIZE, not per metric: niqe
    # needs blocks bigger than a 64px crop but is perfectly happy on the full
    # frame, and a transformer that runs out of memory at 1024x2048 still
    # scores the footprint. Dropping the metric outright would lose the
    # region it can do.
    failed: set = field(default_factory=set)


# Full-reference metrics from pyiqa, by our column name -> pyiqa model name.
# LPIPS stays on the reference `lpips` package (same weights, and it is what
# every number in this repo so far was measured with); everything else comes
# from pyiqa, which wraps them all behind one API.
PYIQA_FR = {
    "dists": "dists",                 # structure/texture split, tolerates
                                      # texture resampling where LPIPS does not
    "stlpips": "stlpips",             # shift-tolerant LPIPS (ECCV 2022)
    "stlpips_vgg": "stlpips-vgg",
    "lpips_plus": "lpips+",           # retrained LPIPS variants
    "lpips_vgg_plus": "lpips-vgg+",
    "ahiq": "ahiq",                   # ViT+CNN, NTIRE winner (HIGHER better)
    "topiq_fr": "topiq_fr",           # semantic top-down      (HIGHER better)
    "pieapp": "pieapp",               # pairwise-preference trained
    "ssim": "ssim",                   # classics, for readers who expect them
    "ms_ssim": "ms_ssim",
    "fsim": "fsim",
}

# No-reference metrics: naturalness proper, scored on each frame alone.
PYIQA_NR = {
    "niqe": "niqe",                   # natural-scene statistics (LOWER better)
    "brisque": "brisque",
    "clipiqa": "clipiqa",             # CLIP-based     (HIGHER better)
    "musiq": "musiq",                 # transformer IQA(HIGHER better)
}

LPIPS_NETS = {"lpips_alex": "alex", "lpips_vgg": "vgg",
              "lpips_squeeze": "squeeze"}

# CVVDP is listed separately: it is the only metric here that takes the
# VIEWING CONDITIONS as input (pixels per degree), which is the same
# assumption the CSF budget is built on -- see build_cvvdp().
ALL_METRICS = (list(LPIPS_NETS) + list(PYIQA_FR) + ["cvvdp"]
               + list(PYIQA_NR))
DEFAULT_METRICS = ["lpips_alex", "lpips_vgg", "dists", "cvvdp"]


def ppd_from_cfg(cfg: dict) -> float:
    """
    Pixels per degree from the run's own viewing geometry.

        theta_pix = 2 * atan(pixel_size / (2 * distance))      [degrees]
        ppd       = 1 / theta_pix

    The same two numbers PatchConfig hands to csf.ViewingGeometry, so CVVDP
    and the tau budget describe one observer rather than two.
    """
    px = float(cfg.get("csf_pixel_size_cm", 0.0114) or 0.0114)
    d = float(cfg.get("csf_viewing_distance_cm", 50.0) or 50.0)
    return 1.0 / (2.0 * math.degrees(math.atan(0.5 * px / d)))


class Cvvdp:
    r"""
    ColorVideoVDP, rebuilt when the viewing geometry changes.

    Returns JOD: 10 = indistinguishable, lower = more visible difference, so
    HIGHER IS BETTER here while every LPIPS-like column is the other way
    round. One JOD step is calibrated to ~75% preference.

    Unlike the deep-feature metrics this one knows how far away the observer
    sits, which is why it can agree or disagree with the CSF budget on its own
    terms rather than only at pixel scale.
    """

    def __init__(self, device):
        self.device = device
        self.ppd = None
        self.metric = None

    def configure(self, ppd: float, log=print):
        if self.metric is not None and abs(ppd - self.ppd) < 1e-6:
            return self.metric
        import pycvvdp
        geom = pycvvdp.vvdp_display_geometry((1, 1), ppd=ppd)
        photo = pycvvdp.vvdp_display_photo_eotf(Y_peak=100.0)
        self.metric = pycvvdp.cvvdp(display_photometry=photo,
                                    display_geometry=geom, heatmap=None,
                                    quiet=True, device=self.device)
        self.ppd = ppd
        log(f"[cvvdp] {ppd:.1f} px/deg from the run's viewing geometry")
        return self.metric

    def __call__(self, ref, dist):
        # uint8 is one of the two datatypes predict() accepts, and our frames
        # are already 8-bit: no second quantisation happens here. predict()
        # takes the TEST frame first and the reference second.
        u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8)
        jod, _ = self.metric.predict(u8(dist), u8(ref), dim_order="BCHW")
        return float(jod)


def build_metrics(names, device, spatial: bool = False):
    """
    name -> Metric, skipping (with a message) anything not installed.

    spatial=True returns the LPIPS map used for the panels; only the lpips
    package supports it, so the name list is ignored beyond its first net.
    """
    if spatial:
        import lpips
        net = next((LPIPS_NETS[n] for n in names if n in LPIPS_NETS), "alex")
        m = lpips.LPIPS(net=net, spatial=True, verbose=False).to(device).eval()
        return {f"lpips_{net}": Metric(f"lpips_{net}", m)}

    out = {}
    for name in names:
        try:
            if name in LPIPS_NETS:
                import lpips as _lpips
                m = _lpips.LPIPS(net=LPIPS_NETS[name], verbose=False
                                 ).to(device).eval()
                out[name] = Metric(name, lambda ref, dist, m=m: float(
                    m(ref * 2 - 1, dist * 2 - 1).mean()))   # lpips wants [-1,1]
            elif name == "cvvdp":
                out[name] = Metric(name, Cvvdp(device), lower_better=False)
            elif name in PYIQA_FR or name in PYIQA_NR:
                import pyiqa
                model = PYIQA_FR.get(name) or PYIQA_NR[name]
                m = pyiqa.create_metric(model, device=device)
                kind = "fr" if name in PYIQA_FR else "nr"
                fn = ((lambda ref, dist, m=m: float(m(dist, ref)))
                      if kind == "fr" else (lambda x, m=m: float(m(x))))
                out[name] = Metric(name, fn, kind=kind,
                                   lower_better=bool(m.lower_better))
            else:
                print(f"[skip] unknown metric {name!r}; known: "
                      f"{', '.join(ALL_METRICS)}")
        except ImportError as e:
            print(f"[skip] {name}: {e}. pip install pyiqa / cvvdp into the "
                  f"evaluation env (see envs/lpips_eval.yml).")
        except Exception as e:                      # a model that will not load
            print(f"[skip] {name}: {type(e).__name__}: {e}")
    if not out:
        sys.exit("[perc] no metric could be built -- nothing to score.")
    return out


def score_pair(metrics: Dict[str, Metric], a: torch.Tensor,
               b: torch.Tensor) -> Dict[str, float]:
    """
    Every full-reference metric on one (clean, modified) pair.

    A metric that fails at one size -- a transformer running out of memory on
    the full 1024x2048 frame is the realistic case -- is skipped at that size
    for the rest of the run rather than killing an evaluation that is
    otherwise fine.
    """
    out = {}
    with torch.no_grad():
        for name, m in metrics.items():
            if m.kind != "fr" or tuple(a.shape[-2:]) in m.failed:
                continue
            try:
                out[name] = m.fn(a, b)              # a = clean, b = modified
            except Exception as e:
                _note_failure(m, a, e)
    return out


def score_single(metrics: Dict[str, Metric], x: torch.Tensor
                 ) -> Dict[str, float]:
    """Every no-reference metric on ONE frame."""
    out = {}
    with torch.no_grad():
        for name, m in metrics.items():
            if m.kind != "nr" or tuple(x.shape[-2:]) in m.failed:
                continue
            try:
                out[name] = m.fn(x)
            except Exception as e:
                _note_failure(m, x, e)
    return out


def _note_failure(m: Metric, x: torch.Tensor, e: Exception):
    hw = tuple(x.shape[-2:])
    m.failed.add(hw)
    print(f"  [metric] {m.name} failed at {hw[0]}x{hw[1]}: "
          f"{type(e).__name__}: {e} -- skipping that size from here on")


def resampled_base(clean: torch.Tensor, box, size: int,
                   quantize: bool) -> torch.Tensor:
    r"""
    The zero-residual composite, from the clean frame alone.

    Exactly the round trip mode='csf' --from_image puts the base through:
    set_reference_from_image() resizes the p x p crop to size x size, apply()
    resizes the render back to p x p. Both bilinear, align_corners=False.
    No checkpoint needed -- only the geometry.
    """
    top, left, p = box
    crop = clean[:, :, top:top + p, left:left + p]
    ref = F.interpolate(crop, size=(size, size), mode="bilinear",
                        align_corners=False)
    back = F.interpolate(ref, size=(p, p), mode="bilinear",
                         align_corners=False).clamp(0, 1)
    base = clean.clone()
    base[:, :, top:top + p, left:left + p] = back
    return quantize8(base) if quantize else base


def score_frames(clean, patched, base, box, metrics, context: float,
                 quantize: bool, anchors=(), image: Optional[int] = None):
    """
    Every LPIPS / pixel number for one clean-patched pair.

    clean, patched, base : [1,3,H,W] in [0,1]; base is the zero-residual
    composite or None where the mode has no base (opaque patches). Shared by
    the checkpoint and the PNG source, so both produce identical columns.

    Returns (row, views): row is flat {column: value} for the CSV; views holds
    the frames on CPU and the footprint box, for save_visuals().
    """
    H, W = clean.shape[-2:]
    top, left, p = box
    crop = (slice(None), slice(None), slice(top, top + p),
            slice(left, left + p))
    t, b, l, r = expand_box(top, left, p, int(round(context * p)), H, W)
    ctx = (slice(None), slice(None), slice(t, b), slice(l, r))
    whole = (slice(None),) * 4
    regions = (("crop", crop), ("ctx", ctx), ("full", whole))

    row = {}
    for region, sl in regions:
        for m, v in score_pair(metrics, clean[sl], patched[sl]).items():
            row[f"{m}_{region}"] = v
        # NO-REFERENCE: clean and patched scored separately. The delta is the
        # part about the patch; the absolute values say how this image sits on
        # that metric's scale to begin with, which varies hugely by scene.
        nr_c = score_single(metrics, clean[sl])
        for m, v in score_single(metrics, patched[sl]).items():
            row[f"{m}_{region}_clean"] = nr_c.get(m, float("nan"))
            row[f"{m}_{region}_patched"] = v
            row[f"{m}_{region}_delta"] = v - nr_c.get(m, float("nan"))

    # The control only exists where there is a base to remove the residual
    # from. universal_csf composites onto the clean window at full resolution,
    # so its floor is zero by construction; opaque modes have no base at all.
    if base is not None:
        # clean + (patched - base): the pasted residual on the sharp original.
        # Quantised again because the sum is a new displayed image.
        resid = clean + (patched - base)
        resid = quantize8(resid) if quantize else resid.clamp(0, 1)
        for name, other in (("floor", base), ("resid", resid)):
            for region, sl in regions:
                for m, v in score_pair(metrics, clean[sl],
                                       other[sl]).items():
                    row[f"{m}_{name}_{region}"] = v

    diff = patched[crop] - clean[crop]
    mse = float(diff.pow(2).mean())
    row["psnr_crop"] = (10 * math.log10(1.0 / mse) if mse > 0
                        else float("inf"))
    row["max_abs_diff_255"] = float(diff.abs().max() * 255.0)
    row["frac_px_changed"] = float((diff.abs().amax(1) > 0.5 / 255).float()
                                   .mean())

    if anchors:
        # Seeded per image, so every run scored on this image sees the same
        # anchor noise and the anchor columns are comparable across runs.
        g = torch.Generator().manual_seed(1234 + int(image or 0))
        sign = (torch.randint(0, 2, clean.shape, generator=g).float() * 2 - 1
                ).to(clean.device)
        inside = torch.zeros_like(clean)
        inside[crop] = 1.0
        for e in anchors:
            for where, mask in (("global", 1.0), ("patch", inside)):
                noisy = quantize8(clean + sign * mask * e / 255.0)
                for m, v in score_pair(metrics, clean, noisy).items():
                    row[f"{m}_anchor_linf{e:g}_{where}"] = v

    views = {"clean": clean[0].cpu(), "patched": patched[0].cpu(),
             "base": None if base is None else base[0].cpu(),
             "box": (top, left, p)}
    return row, views


def evaluate_image(patch, img_norm, mean_t, std_t, metrics, cfg: dict,
                   context: float, quantize: bool, anchors=(),
                   image: Optional[int] = None):
    """
    CHECKPOINT source: rebuild the frames from a saved Patch, then score.

    Adds the CSF visibility columns from Patch.stats(), which only a
    checkpoint can provide.
    """
    H, W = img_norm.shape[-2:]
    mode = patch.cfg.mode
    clean = quantize8((img_norm * std_t + mean_t).clamp(0, 1))
    if mode == "csf" and cfg.get("from_image", False):
        patch.set_reference_from_image(img_norm, mean_t, std_t)
    patched = composite(patch, img_norm, mean_t, std_t, quantize)
    base = (composite(patch, img_norm, mean_t, std_t, quantize,
                      base_only=True)
            if mode == "csf" and patch.reference is not None else None)
    box = footprint_box(patch.placement, H, W, patch.cfg.scale,
                        patch.cfg.scale_ref)

    row, views = score_frames(
        clean, patched, base, box, metrics, context, quantize, anchors,
        image if image is not None else cfg.get("image"))
    if mode in ("csf", "universal_csf"):
        with torch.no_grad():
            st = patch.stats()
        for k in VIS_KEYS:
            if k in st:
                row[k] = float(st[k])
    return row, views


# ── PNG source ───────────────────────────────────────────────────────────────
#
# The attack scripts already write the exact frames the segmentor saw, via
# torchvision save_image -- lossless 8-bit PNG, the same rounding quantize8()
# applies. Scoring those needs no checkpoint, no dataset and no Patch
# reconstruction:
#
#   overfit.py             <run>[/seed*]/diagnostics/panels/{a_clean,c_patched}.png
#   overfit_population.py  <run>/diagnostics/img%04d/panels/...
#   train.py               <run>/panels/img%d/...
#
# Only images the run wrote diagnostics for are available (--no_diagnostics
# writes none), and there are no visibility columns: those need the residual
# on the patch grid, which only the checkpoint holds.

@dataclass
class PngJob:
    clean: Path
    patched: Path
    image: int
    label: str
    ckpt: Optional[Path]      # only for the footprint position
    home: Optional[Path] = None   # dir holding best.pt / final.pt, if any


def _image_from_path(rel: Path) -> Optional[int]:
    for part in rel.parts:
        m = re.fullmatch(r"img(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def discover_pngs(run: Path, cfg: dict) -> List[PngJob]:
    run = run.resolve()
    jobs = []
    for patched in sorted(run.rglob("c_patched.png")):
        clean = patched.with_name("a_clean.png")
        if not clean.exists():
            continue
        rel = patched.parent.relative_to(run)
        image = _image_from_path(rel)
        if image is None:
            image = cfg.get("image")
        if image is None:
            print(f"  [skip] {rel}: cannot tell which val image this is")
            continue
        keep = [x for x in rel.parts if x not in ("diagnostics", "panels")]
        label = "-".join(keep) or "run"

        # The checkpoint that belongs to these frames, read ONLY for its
        # placement: overfit's seed dir or run dir, population's per-image
        # patch dir, train's run dir.
        homes = [run / "patches" / f"img{image:04d}"]
        homes += [run / x for x in keep if x.startswith("seed")]
        homes.append(run)
        ckpt = next((c for h in homes for c in (h / "final.pt", h / "best.pt")
                     if c.exists()), None)
        jobs.append(PngJob(clean, patched, int(image), label, ckpt,
                           ckpt.parent if ckpt is not None else None))
    return jobs


def load_png(path: Path, device) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr / 255.0).permute(2, 0, 1)[None].to(device)


def png_box(job: PngJob, cfg: dict, clean, patched):
    """
    Footprint (top, left, p) and where it came from.

    The PNGs do not record the patch position, so in order of reliability:
    the checkpoint's stored placement; centre, when the run was centre
    placed; else the bounding box of the pixels that changed (square-ified
    to p). The last can come out a few pixels small where the residual left
    border pixels unchanged -- `box_from` says which one was used.
    """
    H, W = clean.shape[-2:]
    # The side rule matters as much as the position: under
    # --patch_scale_ref area the patch is pasted at int(scale*sqrt(H*W)), and
    # a height-rule box would score the wrong region without any error.
    # 'height' is the default for every run written before the flag existed.
    scale = cfg.get("patch_scale", 0.25)
    ref = cfg.get("patch_scale_ref", "height")
    if job.ckpt is not None:
        ck = torch.load(job.ckpt, map_location="cpu")
        c = ck["config"]
        return (footprint_box(ck.get("placement"), H, W,
                              c.get("scale", scale), c.get("scale_ref", ref)),
                "checkpoint")
    if cfg.get("placement", "center") == "center":
        return footprint_box(None, H, W, scale, ref), "center"
    changed = (patched - clean).abs().amax(1)[0] > 0.5 / 255
    ys, xs = torch.nonzero(changed, as_tuple=True)
    if len(ys) == 0:
        return footprint_box(None, H, W, scale, ref), "center"
    return (footprint_box((int(ys.min()), int(xs.min())), H, W, scale, ref),
            "diff")


def png_shows(job: PngJob, cfg: dict, clean, patched, mean_t, std_t) -> str:
    r"""
    Which patch c_patched.png actually shows: 'best', 'final', 'best=final'
    (identical parameters), or 'unknown'.

    NOT KNOWABLE FROM THE FILE NAME. overfit.py wrote the FINAL patch into
    its diagnostics until 2aee5ef and writes best.pt since; population always
    used best.pt. Rather than trust dates, re-render each checkpoint onto
    a_clean.png -- which IS the val image, so --from_image bases come out
    identical -- and see which one reproduces c_patched.png. Needs no
    dataset. Tolerance is one code value: save_image rounds half up,
    quantize8 half to even.
    """
    if job.home is None:
        return "unknown"
    from patchreach.patch.spec import Patch
    img_norm = (clean - mean_t) / std_t
    hits = []
    for name in ("best", "final"):
        ck = job.home / f"{name}.pt"
        if not ck.exists():
            continue
        try:
            patch = Patch.load(ck, clean.device, mean_t, std_t)
            if patch.cfg.mode == "gan":
                return "unknown"
            if patch.cfg.mode == "csf" and cfg.get("from_image"):
                patch.set_reference_from_image(img_norm, mean_t, std_t)
            redo = composite(patch, img_norm, mean_t, std_t, True)
        except Exception as e:                  # provenance must never kill
            print(f"  [png ] could not re-render {ck.name}: {e}")
            continue
        if float((redo - patched).abs().max()) <= 1.0 / 255 + 1e-6:
            hits.append(name)
    return "=".join(hits) if hits else "unknown"


def evaluate_png(job: PngJob, cfg: dict, metrics, context: float,
                 quantize: bool, anchors, device, shows: str = "unknown"):
    clean = load_png(job.clean, device)
    patched = load_png(job.patched, device)
    box, box_from = png_box(job, cfg, clean, patched)
    base = (resampled_base(clean, box, int(cfg["patch_size"]), quantize)
            if cfg.get("patch_mode") == "csf" and cfg.get("from_image")
            and cfg.get("patch_size") else None)
    row, views = score_frames(clean, patched, base, box, metrics, context,
                              quantize, anchors, job.image)
    row["box_from"] = box_from
    row["png_shows"] = shows
    return row, views


# ── output ───────────────────────────────────────────────────────────────────

def _to_png(x: torch.Tensor, path: Path):
    """[3,h,w] in [0,1] -> 8-bit PNG, lossless for already-quantised input."""
    from PIL import Image
    arr = (x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def save_visuals(views: dict, spatial_metric, out_dir: Path, stem: str,
                 title: str, amplify: float, save_images: bool):
    r"""
    Panel + (optionally) every view as its own file.

    PANEL, two rows:
        footprint :  clean | patched | difference x amplify | LPIPS map
        frame     :  the same four on the whole image, footprint outlined

    The frame row is the context the crop row lacks: whether the patch reads
    as a blemish in a scene rather than as a texture seen in isolation.

    IMAGES (out_dir/images/<stem>/), at native resolution, no resampling:
        {clean,patched}_{crop,full}.png     what was scored
        diff_{crop,full}_x<amp>.png         0.5 + amp * (patched - clean)
        lpips_map_{crop,full}.png / .npy    PNG is colour-mapped for looking;
                                            the .npy holds the raw values
        base_{crop,full}.png, diff_floor_{crop,full}_x<amp>.png
                                            csf only: the zero-residual
                                            composite, i.e. what lpips_floor
                                            scored — resampling alone

    Both LPIPS maps share one colour scale, so the crop and the frame can be
    compared by eye. The map on the full frame is not the crop map pasted
    back: LPIPS features see the surround, so it can light up along a seam.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    top, left, p = views["box"]
    clean, patched = views["clean"], views["patched"]
    dev = next(spatial_metric.parameters()).device

    def lmap(a, b):
        with torch.no_grad():
            return spatial_metric(a[None].to(dev) * 2 - 1,
                                  b[None].to(dev) * 2 - 1)[0, 0].cpu()

    def crop(x):
        return x[:, top:top + p, left:left + p]

    def diff(a, b):
        return (0.5 + amplify * (b - a)).clamp(0, 1)

    m_crop = lmap(crop(clean), crop(patched))
    m_full = lmap(clean, patched)
    vmax = float(max(m_crop.max(), m_full.max()))
    amp = f"x{amplify:g}"

    fig, ax = plt.subplots(2, 4, figsize=(16, 7.2),
                           gridspec_kw={"height_ratios": [1.0, 0.55]})
    rows = ((crop(clean), crop(patched), m_crop, "footprint", False),
            (clean, patched, m_full, "full frame", True))
    for r, (c, pt, m, name, outline) in enumerate(rows):
        for a_, im, t in zip(ax[r, :3], (c, pt, diff(c, pt)),
                             ("clean", "patched", f"difference {amp}")):
            a_.imshow(im.permute(1, 2, 0).numpy())
            a_.set_title(f"{name}: {t}", fontsize=9)
        h = ax[r, 3].imshow(m.numpy(), cmap="magma", vmin=0, vmax=vmax)
        ax[r, 3].set_title(f"{name}: LPIPS map (mean {float(m.mean()):.4f})",
                           fontsize=9)
        fig.colorbar(h, ax=ax[r, 3], fraction=0.046)
        if outline:
            for a_ in ax[r]:
                a_.add_patch(Rectangle((left - 0.5, top - 0.5), p, p,
                                       fill=False, edgecolor="cyan", lw=0.8))
    for a_ in ax.ravel():
        a_.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"panel_{stem}.png", dpi=120)
    plt.close(fig)

    if not save_images:
        return
    d = out_dir / "images" / stem
    d.mkdir(parents=True, exist_ok=True)
    import numpy as np
    for name, c, pt, m in (("crop", crop(clean), crop(patched), m_crop),
                           ("full", clean, patched, m_full)):
        _to_png(c, d / f"clean_{name}.png")
        _to_png(pt, d / f"patched_{name}.png")
        _to_png(diff(c, pt), d / f"diff_{name}_{amp}.png")
        plt.imsave(d / f"lpips_map_{name}.png", m.numpy(), cmap="magma",
                   vmin=0, vmax=vmax)
        np.save(d / f"lpips_map_{name}.npy", m.numpy())
    if views["base"] is not None:
        b = views["base"]
        _to_png(crop(b), d / "base_crop.png")
        _to_png(diff(crop(clean), crop(b)), d / f"diff_floor_crop_{amp}.png")
        _to_png(b, d / "base_full.png")
        _to_png(diff(clean, b), d / f"diff_floor_full_{amp}.png")


def summarise(rows: List[dict]) -> dict:
    """Distribution per numeric column: a mean alone hides the tail."""
    out = {"n": len(rows)}
    cols = sorted({k for r in rows for k, v in r.items()
                   if isinstance(v, float) and not k.startswith("_")})
    for k in cols:
        v = sorted(r[k] for r in rows if k in r and math.isfinite(r[k]))
        if not v:
            continue
        q = (statistics.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3)
        out[k] = {"mean": statistics.fmean(v), "median": statistics.median(v),
                  "q25": q[0], "q75": q[2],
                  "p90": v[min(len(v) - 1, int(0.9 * len(v)))],
                  "max": v[-1]}
    return out


def write_csv(rows: List[dict], path: Path):
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path,
                   help="run directories (each holding config.json)")
    p.add_argument("--cityscapes_root", default=None,
                   help="override the root recorded in config.json — needed "
                        "whenever this runs on a different machine")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   metavar="M",
                   help="perceptual metrics to run, or 'all'. The FIRST one "
                        "is the primary: it drives the console line and the "
                        "panels. Known: " + ", ".join(ALL_METRICS) + ". "
                        "lpips_alex is the field's reference point and should "
                        "stay first unless you have a reason; the rest are a "
                        "robustness check, not a menu to pick from after "
                        "seeing the numbers.")
    p.add_argument("--nets", nargs="+", default=None,
                   choices=["alex", "vgg", "squeeze"],
                   help="deprecated alias: --nets alex vgg == --metrics "
                        "lpips_alex lpips_vgg")
    p.add_argument("--source", default="auto",
                   choices=["auto", "png", "checkpoint"],
                   help="png = score the a_clean.png / c_patched.png the run "
                        "already wrote (no checkpoint, no dataset); "
                        "checkpoint = rebuild the frames from final/best.pt "
                        "(adds the CSF visibility columns); auto = png where "
                        "the run has them, else checkpoint -- unless "
                        "--checkpoint best/final names a patch the PNGs are "
                        "verified NOT to show (column png_shows)")
    p.add_argument("--checkpoint", default="auto",
                   choices=["auto", "final", "best"],
                   help="auto = final.pt, else best.pt")
    p.add_argument("--images", default="fixed10",
                   help="shared-patch (train.py) runs only: 'fixed10', "
                        "'firstN', or a list of val indices")
    p.add_argument("--n_images", type=int, default=0,
                   help="cap per run, 0 = no cap")
    p.add_argument("--context", type=float, default=0.25,
                   help="lpips_ctx margin per side, as a fraction of p")
    p.add_argument("--no_quantize", action="store_true",
                   help="score float composites instead of 8-bit ones")
    p.add_argument("--panels", type=int, default=4,
                   help="save a panel (and, unless --no_save_images, "
                        "every view as its own PNG) for the first N images "
                        "per run")
    p.add_argument("--anchors", nargs="*", type=float, default=None,
                   metavar="E",
                   help="score +-E/255 sign-noise anchors (global and "
                        "footprint-only) on each frame; bare --anchors = "
                        "2 4 8 16")
    p.add_argument("--no_save_images", action="store_true",
                   help="panels only, no per-view image files")
    p.add_argument("--amplify", type=float, default=10.0,
                   help="difference gain in the panels")
    p.add_argument("--out_name", default="perc_eval",
                   help="per-run output subdirectory holding every "
                        "evaluation of that run")
    p.add_argument("--tag", default="",
                   help="names this evaluation: output goes to "
                        "<run>/<out_name>/<tag>/ and every row carries it as "
                        "eval_tag, so png vs checkpoint, best vs final or "
                        "quantized vs float passes sit side by side instead "
                        "of overwriting each other")
    p.add_argument("--out_csv", type=Path, default=None,
                   help="one combined CSV over every run, for the tables")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    if a.nets:                                   # deprecated, still honoured
        a.metrics = [f"lpips_{n}" for n in a.nets]
    if len(a.metrics) == 1 and a.metrics[0] == "all":
        a.metrics = list(ALL_METRICS)
    return a


def choose_png(a, shows: List[str]) -> bool:
    """
    Use the PNG source for a run whose PNGs show `shows` (one per image)?

    --source png  : always; a mismatch with --checkpoint is only warned about.
    --source auto : yes, unless --checkpoint best/final names a patch the
                    PNGs are VERIFIED not to show -- then the checkpoint
                    source, so a number is never labelled with the wrong
                    patch. 'unknown' (no checkpoint to compare) keeps PNGs:
                    there is nothing else to score, and the column says so.
    """
    want = a.checkpoint
    wrong = [x for x in shows
             if want != "auto" and x != "unknown" and want not in x.split("=")]
    if a.source == "png":
        if wrong:
            print(f"  [png ] note: --checkpoint {want}, but the PNGs show "
                  f"{sorted(set(wrong))}; scoring the PNGs as asked.")
        return True
    return a.source == "auto" and not wrong


# Directories the attack scripts own inside a run. Writing into one of them
# could replace a file a later analysis reads -- overfit.py --seeds and
# overfit_population.py both keep a summary.json of their own at run level.
ATTACK_DIRS = {"diagnostics", "panels", "patches", "aggregate",
               "intermediate_patches"}


def check_outputs(a):
    """
    Refuse, before any work, an output location that could touch run files.

    Every file this script writes lands in <run>/<out_name>/[<tag>/], which
    it creates, plus --out_csv. out_name and tag must each be ONE plain
    directory name, neither the run itself nor an attack-owned directory, so
    that no evaluation can land on a file a run was read for. --out_csv may
    only replace a CSV this script wrote.
    """
    parts = [a.out_name] + ([a.tag] if a.tag else [])
    for part in parts:
        if (not part or Path(part).name != part or part in (".", "..")
                or part.lower() in ATTACK_DIRS):
            sys.exit(f"[perc] refusing --out_name/--tag component {part!r}: "
                     f"results go to <run>/{'/'.join(parts)}/, each component "
                     f"one new directory name, not the run itself or one of "
                     f"{sorted(ATTACK_DIRS)}.")
    sub = Path(*parts)
    if a.out_csv is not None:
        if a.out_csv.suffix.lower() != ".csv":
            sys.exit(f"[perc] --out_csv must end in .csv, got {a.out_csv}")
        if a.out_csv.exists():
            with open(a.out_csv, newline="") as f:
                header = f.readline()
            if "eval_tag" not in header:
                sys.exit(f"[perc] {a.out_csv} exists and was not written by "
                         f"this script; refusing to overwrite it.")
    return sub


def main(argv=None):
    a = parse_args(argv)
    out_sub = check_outputs(a)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available()
                                       else "cpu"))
    mean_t, std_t = norm_tensors(device)
    metrics = build_metrics(a.metrics, device)
    primary = next((m for m in a.metrics if m in metrics), None)
    print(f"[perc] metrics: {', '.join(metrics)}   primary: {primary}")
    spatial = (list(build_metrics(a.metrics, device, spatial=True).values()
                    )[0].fn if a.panels > 0 else None)
    image_list = parse_images(a.images, a.n_images)
    anchors = (() if a.anchors is None
               else tuple(a.anchors) or (2.0, 4.0, 8.0, 16.0))

    from patchreach.patch.spec import Patch

    datasets: Dict[tuple, CityscapesSeg] = {}

    def dataset_for(cfg):
        root = a.cityscapes_root or cfg.get("cityscapes_root")
        key = (root, cfg.get("img_h", 512), cfg.get("img_w", 1024),
               cfg.get("scale", "resize"))
        if key not in datasets:
            # val, never randomly cropped: the attack scripts attack val
            # images and make_dataset() only randomises the train split.
            datasets[key] = CityscapesSeg(root, "val", key[1], key[2],
                                          scale=key[3], random_crop=False)
        return datasets[key]

    all_rows = []
    for run in a.runs:
        # A PATH THAT DOES NOT EXIST IS NOT "no config.json". The two were
        # reported with one message, and a mistyped arch name (deeplab101 for
        # the registry's deeplab_101) then looked like a run whose config had
        # gone missing -- fourteen of them in a row, exit code 0.
        if not run.exists():
            print(f"[skip] {run}: no such directory")
            if run.parent.is_dir():
                import difflib
                near = difflib.get_close_matches(
                    run.name, [p.name for p in run.parent.iterdir()
                               if p.is_dir()], n=3, cutoff=0.5)
                if near:
                    print(f"        did you mean: {', '.join(near)}")
            continue
        if not (run / "config.json").exists():
            print(f"[skip] {run}: directory exists but holds no config.json")
            continue
        cfg = json.loads((run / "config.json").read_text())
        if "cvvdp" in metrics:
            # The viewing geometry is a property of the RUN, not of this
            # invocation: the same pixel size and distance the tau budget was
            # built on. Rebuilt only when it actually changes.
            try:
                metrics["cvvdp"].fn.configure(ppd_from_cfg(cfg))
            except Exception as e:
                print(f"  [cvvdp] unavailable: {type(e).__name__}: {e}")
                metrics.pop("cvvdp")

        png_jobs = (discover_pngs(run, cfg) if a.source != "checkpoint"
                    else [])
        shows = {}
        for job in png_jobs:
            shows[id(job)] = png_shows(job, cfg, load_png(job.clean, device),
                                       load_png(job.patched, device),
                                       mean_t, std_t)
        if png_jobs and not choose_png(a, list(shows.values())):
            print(f"[run ] {run}: PNGs show "
                  f"{sorted(set(shows.values()))}, not --checkpoint "
                  f"{a.checkpoint}; using the checkpoint instead")
            png_jobs = []
        source = "png" if png_jobs else "checkpoint"
        if a.source == "png" and not png_jobs:
            print(f"[skip] {run}: no a_clean.png / c_patched.png pairs")
            continue
        jobs = (png_jobs if source == "png"
                else discover(run, a.checkpoint, image_list))
        if a.n_images > 0:
            jobs = jobs[:a.n_images]
        if not jobs:
            print(f"[skip] {run}: no {a.checkpoint} checkpoint found")
            continue

        print(f"\n[run ] {run}  ({len(jobs)} image(s) from {source}, "
              f"mode {cfg.get('patch_mode')}, tau {cfg.get('csf_threshold')})")
        rows = []
        for n, job in enumerate(jobs):
            if source == "png":
                row, views = evaluate_png(
                    job, cfg, metrics, a.context, not a.no_quantize,
                    anchors, device, shows[id(job)])
                mode, ck_name = cfg.get("patch_mode"), None
            else:
                patch = Patch.load(job.ckpt, device, mean_t, std_t)
                if patch.cfg.mode == "gan":
                    print("  [skip] gan mode needs the BigGAN generator; "
                          "not supported in the lpips env")
                    break
                img = dataset_for(cfg)[job.image][0].unsqueeze(0).to(device)
                row, views = evaluate_image(
                    patch, img, mean_t, std_t, metrics, cfg, a.context,
                    quantize=not a.no_quantize, anchors=anchors,
                    image=job.image)
                mode, ck_name = patch.cfg.mode, job.ckpt.name
            row = {"run": str(run), "tag": cfg.get("tag"), "eval_tag": a.tag,
                   "source": source, "arch": cfg.get("arch"),
                   "patch_mode": mode,
                   "csf_threshold": cfg.get("csf_threshold"),
                   "csf_param": cfg.get("csf_param"),
                   "csf_enforce": cfg.get("csf_enforce"),
                   "checkpoint": ck_name, "which": job.label,
                   "image": job.image, **row}
            rows.append(row)
            main_col = f"{primary}_full"
            print(f"  img {job.image:4d} {job.label:>12s}  "
                  f"{primary} full {row.get(main_col, float('nan')):.4f}"
                  + (f"  floor_full {row[f'{primary}_floor_full']:.4f}"
                     f"  resid_full {row[f'{primary}_resid_full']:.4f}"
                     if f"{primary}_floor_full" in row else "")
                  + f"  crop {row.get(f'{primary}_crop', float('nan')):.4f}"
                  + (f"  vis {row['visibility']:.3f}"
                     if "visibility" in row else "")
                  + f"  psnr {row['psnr_crop']:.1f} dB")
            if n < a.panels:
                save_visuals(views, spatial, run / out_sub,
                             f"{job.label}_img{job.image:04d}",
                             f"{run.name}  img {job.image}  "
                             f"{main_col}={row[main_col]:.4f}", a.amplify,
                             save_images=not a.no_save_images)

        if not rows:
            continue
        out = run / out_sub
        write_csv(rows, out / "per_image.csv")
        summ = summarise(rows)
        summ.update({"metrics": list(metrics),
                     "primary": primary,
                     # which way each column reads, so a table built from
                     # this file cannot silently rank ahiq or cvvdp upside
                     # down -- for those two, HIGHER means more similar.
                     "lower_better": {n: m.lower_better
                                      for n, m in metrics.items()},
                     "quantized": not a.no_quantize,
                     "context": a.context, "source": source,
                     "checkpoint": a.checkpoint if source == "checkpoint"
                     else None, "eval_tag": a.tag})
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        med = summ.get(f"{primary}_full", {})
        print(f"  -> {out}/  median {primary}_full "
              f"{med.get('median', float('nan')):.4f}  "
              f"(IQR {med.get('q25', float('nan')):.4f}-"
              f"{med.get('q75', float('nan')):.4f})")
        all_rows += rows

    if a.out_csv and all_rows:
        write_csv(all_rows, a.out_csv)
        print(f"\n[done] {len(all_rows)} rows -> {a.out_csv}")

    if not all_rows:
        # FAIL LOUDLY. A scoring run that scored nothing is a failure, and
        # exiting 0 let a shell loop and an sbatch job both report success
        # while producing no output at all.
        print(f"\n[lpips] nothing scored: none of the {len(a.runs)} path(s) "
              f"given yielded an image. See the [skip] lines above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
