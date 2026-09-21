#!/usr/bin/env python
r"""
LPIPS evaluation of saved patches — how natural does the frame look afterwards?

    conda activate lpips_eval
    python scripts/lpips_eval.py results/overfit/segformer_csf_* \
        --cityscapes_root $CS --nets alex vgg --out_csv lpips_csf.csv

RUNS IN ITS OWN ENVIRONMENT, ON PURPOSE. The attack env is pinned to torch 1.11
+ mmcv 1.x and `lpips` does not install cleanly into it, so this script never
touches a segmentor. It imports only the model-free half of patchreach
(patch/spec.py, patch/csf.py, data/cityscapes.py: torch + torchvision + numpy),
rebuilds the clean and the patched frame from what the attack scripts already
saved, and scores the pair. No model, no mmseg, no GPU required.

WHAT IT READS — every layout the attack scripts write:

    overfit.py             <run>/config.json + <run>/{final,best}.pt
                           <run>/seed*/{final,best}.pt     (--seeds > 1)
    overfit_population.py  <run>/config.json + <run>/patches/img%04d/best.pt
    train.py               <run>/config.json + <run>/{final,best}.pt
                           ONE shared patch, scored on every image in --images

The reference that mode='csf' sits on is NOT in the checkpoint (Patch.save
stores the parameter, the config and the placement), so for --from_image runs
it is re-derived from the same val image and placement, exactly as
overfit_population.py does before its diagnostics. Skip that and render()
falls back to 0.5 grey — an opaque grey square scored as a "CSF patch".

THREE LPIPS NUMBERS PER IMAGE, because one of them alone misleads:

    lpips_crop   clean vs patched, the p x p FOOTPRINT only.  <- the result
    lpips_ctx    the footprint plus a margin (--context x p per side), so a
                 visible seam between patch and surround is counted too.
    lpips_full   the whole frame. LPIPS is a spatial MEAN, so a 128 px patch
                 in a 512x1024 frame is diluted ~32x. Reported for
                 completeness; do not quote it as the naturalness number.

and one control, csf modes only:

    lpips_floor  clean vs the ZERO-RESIDUAL composite: the base alone,
                 downsampled to --patch_size and pasted back up to p. Nonzero
                 whenever patch_size != int(img_h * patch_scale), and then part
                 of lpips_crop is resampling blur rather than the residual.
                 If floor is a large fraction of crop, the number is about the
                 pipeline, not the attack.

QUANTISATION. The patched frame is rounded to 8 bit before scoring (turn off
with --no_quantize). That is what a display shows and what a saved PNG holds —
and near threshold a CSF residual is only a few code values, so a float-only
score would credit the attack with sub-quantum structure no observer can ever
receive. `frac_px_changed` records how much of the footprint survives.

LPIPS IS NOT A HUMAN STUDY. It is a learned distance calibrated on 2AFC
judgements of distortions (Zhang et al. 2018), and it is an independent check
on the CSF budget precisely because nothing in the attack optimised against it.
Low LPIPS says "close to the original under a deep-feature metric"; it does not
say "invisible". Put it next to the realised visibility columns, not in place
of them — and score a raw (opaque) patch run through this same script so the
csf number has something to be low relative to.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

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


def score_pair(metrics: Dict[str, "torch.nn.Module"], a: torch.Tensor,
               b: torch.Tensor) -> Dict[str, float]:
    """LPIPS per network. Inputs [1,3,h,w] in [0,1]; lpips wants [-1,1]."""
    out = {}
    with torch.no_grad():
        for net, m in metrics.items():
            out[net] = float(m(a * 2 - 1, b * 2 - 1).mean())
    return out


def evaluate_image(patch, img_norm, mean_t, std_t, metrics, cfg: dict,
                   context: float, quantize: bool):
    """
    Every number for one (patch, image) pair, plus the crops for a panel.

    Returns (row, crops) where row is flat {column: value} for the CSV.
    """
    H, W = img_norm.shape[-2:]
    mode = patch.cfg.mode
    from_image = bool(cfg.get("from_image", False))

    clean = quantize8((img_norm * std_t + mean_t).clamp(0, 1))
    if mode == "csf" and from_image:
        patch.set_reference_from_image(img_norm, mean_t, std_t)
    patched = composite(patch, img_norm, mean_t, std_t, quantize)

    top, left, p = footprint_box(patch.placement, H, W, patch.cfg.scale,
                                 patch.cfg.scale_ref)
    crop = (slice(None), slice(None), slice(top, top + p),
            slice(left, left + p))
    t, b, l, r = expand_box(top, left, p, int(round(context * p)), H, W)
    ctx = (slice(None), slice(None), slice(t, b), slice(l, r))

    row = {}
    for net, v in score_pair(metrics, clean[crop], patched[crop]).items():
        row[f"lpips_crop_{net}"] = v
    for net, v in score_pair(metrics, clean[ctx], patched[ctx]).items():
        row[f"lpips_ctx_{net}"] = v
    for net, v in score_pair(metrics, clean, patched).items():
        row[f"lpips_full_{net}"] = v

    # The control only exists where there is a base to remove the residual
    # from. universal_csf composites onto the clean window at full resolution,
    # so its floor is zero by construction; opaque modes have no base at all.
    if mode == "csf" and patch.reference is not None:
        base = composite(patch, img_norm, mean_t, std_t, quantize,
                         base_only=True)
        for net, v in score_pair(metrics, clean[crop], base[crop]).items():
            row[f"lpips_floor_{net}"] = v

    diff = patched[crop] - clean[crop]
    mse = float(diff.pow(2).mean())
    row["psnr_crop"] = (10 * math.log10(1.0 / mse) if mse > 0
                        else float("inf"))
    row["max_abs_diff_255"] = float(diff.abs().max() * 255.0)
    row["frac_px_changed"] = float((diff.abs().amax(1) > 0.5 / 255).float()
                                   .mean())

    if mode in ("csf", "universal_csf"):
        with torch.no_grad():
            st = patch.stats()
        for k in VIS_KEYS:
            if k in st:
                row[k] = float(st[k])
    return row, (clean[crop][0].cpu(), patched[crop][0].cpu())


# ── output ───────────────────────────────────────────────────────────────────

def save_panel(clean, patched, spatial_metric, path: Path, title: str,
               amplify: float):
    """clean | patched | amplified difference | LPIPS spatial map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        smap = spatial_metric(clean[None] * 2 - 1, patched[None] * 2 - 1
                              )[0, 0].cpu()
    diff = (0.5 + amplify * (patched - clean)).clamp(0, 1)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for a_, im, t in zip(ax[:3], (clean, patched, diff),
                         ("clean", "patched", f"difference x{amplify:g}")):
        a_.imshow(im.permute(1, 2, 0).numpy())
        a_.set_title(t)
    h = ax[3].imshow(smap.numpy(), cmap="magma")
    ax[3].set_title(f"LPIPS map (mean {float(smap.mean()):.4f})")
    fig.colorbar(h, ax=ax[3], fraction=0.046)
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


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

def build_metrics(nets, device, spatial: bool = False):
    try:
        import lpips
    except ImportError:
        sys.exit("[lpips] package not installed in this environment. "
                 "Create the env from envs/lpips_eval.yml.")
    return {n: lpips.LPIPS(net=n, spatial=spatial, verbose=False
                           ).to(device).eval() for n in nets}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path,
                   help="run directories (each holding config.json)")
    p.add_argument("--cityscapes_root", default=None,
                   help="override the root recorded in config.json — needed "
                        "whenever this runs on a different machine")
    p.add_argument("--nets", nargs="+", default=["alex"],
                   choices=["alex", "vgg", "squeeze"],
                   help="alex is the LPIPS authors' recommended forward "
                        "metric; vgg is closer to a perceptual loss")
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
                   help="save a visual panel for the first N images per run")
    p.add_argument("--amplify", type=float, default=10.0,
                   help="difference gain in the panels")
    p.add_argument("--out_name", default="lpips",
                   help="per-run output subdirectory")
    p.add_argument("--out_csv", type=Path, default=None,
                   help="one combined CSV over every run, for the tables")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available()
                                       else "cpu"))
    mean_t, std_t = norm_tensors(device)
    metrics = build_metrics(a.nets, device)
    spatial = (build_metrics(a.nets[:1], device, spatial=True)[a.nets[0]]
               if a.panels > 0 else None)
    image_list = parse_images(a.images, a.n_images)

    from patchreach.patch.spec import Patch

    datasets: Dict[tuple, CityscapesSeg] = {}
    all_rows = []
    for run in a.runs:
        if not (run / "config.json").exists():
            print(f"[skip] {run}: no config.json")
            continue
        cfg = json.loads((run / "config.json").read_text())
        jobs = discover(run, a.checkpoint, image_list)
        if a.n_images > 0:
            jobs = jobs[:a.n_images]
        if not jobs:
            print(f"[skip] {run}: no {a.checkpoint} checkpoint found")
            continue

        root = a.cityscapes_root or cfg.get("cityscapes_root")
        key = (root, cfg.get("img_h", 512), cfg.get("img_w", 1024),
               cfg.get("scale", "resize"))
        if key not in datasets:
            # val, never randomly cropped: the attack scripts attack val
            # images and make_dataset() only randomises the train split.
            datasets[key] = CityscapesSeg(root, "val", key[1], key[2],
                                          scale=key[3], random_crop=False)
        ds = datasets[key]

        print(f"\n[run ] {run}  ({len(jobs)} image(s), "
              f"mode {cfg.get('patch_mode')}, tau {cfg.get('csf_threshold')})")
        rows = []
        for n, job in enumerate(jobs):
            patch = Patch.load(job.ckpt, device, mean_t, std_t)
            if patch.cfg.mode == "gan":
                print("  [skip] gan mode needs the BigGAN generator; not "
                      "supported in the lpips env")
                break
            img = ds[job.image][0].unsqueeze(0).to(device)
            row, (c_crop, p_crop) = evaluate_image(
                patch, img, mean_t, std_t, metrics, cfg, a.context,
                quantize=not a.no_quantize)
            row = {"run": str(run), "tag": cfg.get("tag"),
                   "arch": cfg.get("arch"), "patch_mode": patch.cfg.mode,
                   "csf_threshold": cfg.get("csf_threshold"),
                   "csf_param": cfg.get("csf_param"),
                   "csf_enforce": cfg.get("csf_enforce"),
                   "checkpoint": job.ckpt.name, "which": job.label,
                   "image": job.image, **row}
            rows.append(row)
            main_col = f"lpips_crop_{a.nets[0]}"
            print(f"  img {job.image:4d} {job.label:>8s}  "
                  f"{main_col} {row[main_col]:.4f}"
                  + (f"  floor {row[f'lpips_floor_{a.nets[0]}']:.4f}"
                     if f"lpips_floor_{a.nets[0]}" in row else "")
                  + (f"  vis {row['visibility']:.3f}"
                     if "visibility" in row else "")
                  + f"  psnr {row['psnr_crop']:.1f} dB")
            if n < a.panels:
                save_panel(c_crop, p_crop, spatial,
                           run / a.out_name / f"panel_{job.label}_"
                           f"img{job.image:04d}.png",
                           f"{run.name}  img {job.image}  "
                           f"{main_col}={row[main_col]:.4f}", a.amplify)

        if not rows:
            continue
        out = run / a.out_name
        write_csv(rows, out / "per_image.csv")
        summ = summarise(rows)
        summ.update({"nets": a.nets, "quantized": not a.no_quantize,
                     "context": a.context, "checkpoint": a.checkpoint})
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        med = summ.get(f"lpips_crop_{a.nets[0]}", {})
        print(f"  -> {out}/  median lpips_crop_{a.nets[0]} "
              f"{med.get('median', float('nan')):.4f}  "
              f"(IQR {med.get('q25', float('nan')):.4f}-"
              f"{med.get('q75', float('nan')):.4f})")
        all_rows += rows

    if a.out_csv and all_rows:
        write_csv(all_rows, a.out_csv)
        print(f"\n[done] {len(all_rows)} rows -> {a.out_csv}")


if __name__ == "__main__":
    main()
