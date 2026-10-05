#!/usr/bin/env python
r"""
best.pt, measured: the numbers the run log only ever printed for final.pt.

    python scripts/best_checkpoint_stats.py --cityscapes_root $CS
    python scripts/best_checkpoint_stats.py --cityscapes_root $CS \
        --runs results/overfit_464/cospgd/setr_pup_csf_cospgd_img464_t1

WHY. overfit.py writes best.pt whenever the remote drop sets a new record, but
every aggregate it PRINTS -- whole-image mIoU, remote mIoU, flip rate, realised
visibility -- belongs to the patch the run ended on. The single best.pt number
that survives into results.json is `best_drop_remote`. A run that degraded
after its peak therefore reports a drop from one patch and a flip rate and a
visibility from another, and those cannot honestly share a row of a table.

This recomputes the WHOLE row from best.pt, on the image the patch was trained
on, with the architecture, resolution, scale and inference mode that run
recorded. Nothing is retrained and nothing is overwritten: each run gets
<run>/best_stats.json, and one collected table lands under --out_root.

WHAT IS MEASURED, per run:
    clean/best all        whole-image mIoU and its drop
    clean/best remote     mIoU outside the footprint and its drop
    any_flip_rate         % of remote pixels whose argmax moved
    visibility            realised CSF visibility, nominal AND local (csf only)

THE FINAL COLUMNS ARE READ, NOT RECOMPUTED. results.json already holds that
run's own final numbers; recomputing them is one more chance for this table to
disagree with the log it is meant to sit beside. The recomputed
best_drop_remote IS cross-checked against the logged best_drop_remote, because
that is the one quantity which exists on both sides -- and so the only
available proof that this script reconstructed the right patch.

PLACEMENT COMES FROM THE CHECKPOINT. best.pt stores where the patch sat and
this evaluates it on the image it was trained on, so the placement is known
exactly rather than re-derived. That is also what makes --placement gradcam
runs readable here: a re-resolve would need the CAM rebuilt, and the stored
coordinate is the ground truth anyway.

THE BASE IS REBUILT FOR mode='csf'. The checkpoint stores the parameter and the
placement, never the base -- the base is the image region the patch covers.
set_reference_from_image() runs for every csf run, and it is not a flag here:
without it the base is flat grey, the patch becomes a visible square with an
invisible texture, and both the drop and the visibility would describe
something other than the residual under test.
"""
from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import torch

from _common import make_dataset, setup_model
from patchreach.data.cityscapes import norm_tensors, upsample_to
from patchreach.metrics.miou import single_image_miou, attack_rates
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, seed_everything, increment_path

NAN = float("nan")

# Reading order for the printed table: transformers first, then the ResNet
# depth ladder shallow -> deep. Anything else is appended alphabetically.
ARCH_ORDER = ["segformer_b0", "segformer_b5", "setr_pup",
              "deeplab_18", "deeplab_50", "deeplab_101"]

# Everything that has to agree before two runs can share one model build.
MODEL_KEYS = ("arch", "cfg_path", "weights", "img_h", "img_w", "inference",
              "scale", "num_classes", "low_pr")

CSV_FIELDS = [
    "run", "loss_fn", "arch", "mode", "tau", "image", "img_h", "img_w",
    "inference", "scale",
    "clean_all", "clean_remote",
    "best_all", "best_remote", "best_drop_all", "best_drop_remote",
    "best_any_flip_rate", "best_visibility", "best_visibility_local",
    "best_resid_rms", "best_frac_at_bound", "best_frac_at_clip",
    "final_all", "final_remote", "final_drop_all", "final_drop_remote",
    "final_any_flip_rate", "final_visibility", "final_visibility_local",
    "logged_best_drop_remote", "best_minus_final_drop_remote",
    "degraded_after_peak", "checkpoint_matches_log",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"[best] unreadable JSON, treated as absent: {path}")
        return {}


def run_config(run: Path) -> dict:
    """
    The overfit run's own argparse namespace, whichever file holds it.

    config.json is written once per INVOCATION and results.json once per
    REPEAT, so a --seeds run keeps the config one level up. Both are checked,
    nearest first.
    """
    r = load_json(run / "results.json")
    if "config" in r:
        return r["config"]
    for d in (run, run.parent):
        c = load_json(d / "config.json")
        if c:
            return c
    return {}


def discover(root: Path) -> list:
    """Every directory under `root` that holds a best.pt."""
    if not root.is_dir():
        raise SystemExit(f"[best] no such directory: {root}")
    return sorted({p.parent for p in root.rglob("best.pt")})


@torch.no_grad()
def measure(model, patch, img, label, num_classes, mean_t, std_t,
            clean_logits) -> dict:
    """One (best.pt, model, image) row. `clean_logits` is cached by the caller."""
    H, W = img.shape[-2:]
    hw = label.shape[-2:]

    # The checkpoint's placement is the one the attack used on THIS image, so
    # it is read rather than re-derived. Only a checkpoint that predates the
    # field falls through to a resolve.
    if patch.placement is None:
        if patch.cfg.placement == "gradcam":
            raise SystemExit(
                "[best] this checkpoint is --placement gradcam but stores no "
                "placement, and the sensitivity map is not rebuilt here. "
                "Nothing can say where the patch sat.")
        patch.resolve_placement(H, W, clean_logits.argmax(1)[0])
    # AFTER the placement is known, never before: the base is the region at
    # the RESOLVED placement, exactly as optimise.prepare() took it.
    if patch.cfg.mode == "csf":
        patch.set_reference_from_image(img, mean_t, std_t)

    patched, fp = patch.apply(img)
    adv_logits = upsample_to(model(patched), hw)

    # classes='gt' throughout, the default attack_image() ran under, so the
    # recomputed drop is comparable to the logged one rather than merely
    # similar to it.
    clean_all = single_image_miou(clean_logits, label, num_classes)
    clean_rem = single_image_miou(clean_logits, label, num_classes, exclude=fp)
    best_all = single_image_miou(adv_logits, label, num_classes)
    best_rem = single_image_miou(adv_logits, label, num_classes, exclude=fp)
    rates = attack_rates(clean_logits, adv_logits, label, fp)
    st = patch.stats()

    return {"clean_all": clean_all, "clean_remote": clean_rem,
            "best_all": best_all, "best_remote": best_rem,
            "best_drop_all": clean_all - best_all,
            "best_drop_remote": clean_rem - best_rem,
            "best_any_flip_rate": rates["any_flip_rate"],
            "best_visibility": st.get("visibility", NAN),
            "best_visibility_local": st.get("visibility_local", NAN),
            "best_resid_rms": st.get("resid_rms", NAN),
            "best_frac_at_bound": st.get("frac_at_bound", NAN),
            "best_frac_at_clip": st.get("frac_at_clip", NAN)}


def fmt(v, w=9, d=2):
    if v is None or v != v:
        return f"{'--':>{w}}"
    return f"{v:>{w}.{d}f}"


def print_table(rows):
    """Best, then what final.pt reported for the same run."""
    aw = max(12, max(len(r["arch"]) for r in rows) + 2)
    hdr = (f"  {'loss':<8}{'arch':<{aw}}{'mode':<6}{'tau':>6}  "
           f"{'cleanAll':>9}{'bestAll':>9}{'dropAll':>9}  "
           f"{'cleanRem':>9}{'bestRem':>9}{'dropRem':>9}{'flip%':>9}  "
           f"{'vis':>9}{'visLoc':>9}")
    print(f"\n{'=' * len(hdr)}")
    print("  BEST CHECKPOINT (recomputed from best.pt)")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    for r in rows:
        tau = r["tau"] if r["tau"] is not None else "--"
        print(f"  {str(r['loss_fn']):<8}{r['arch']:<{aw}}{r['mode']:<6}"
              f"{tau:>6}  "
              + fmt(r["clean_all"]) + fmt(r["best_all"])
              + fmt(r["best_drop_all"]) + "  "
              + fmt(r["clean_remote"]) + fmt(r["best_remote"])
              + fmt(r["best_drop_remote"]) + fmt(r["best_any_flip_rate"], 9, 1)
              + "  " + fmt(r["best_visibility"], 9, 4)
              + fmt(r["best_visibility_local"], 9, 4))

    hdr2 = (f"  {'loss':<8}{'arch':<{aw}}{'mode':<6}{'tau':>6}  "
            f"{'dropRem':>9}{'flip%':>9}{'vis':>9}   "
            f"{'dropRem':>9}{'flip%':>9}{'vis':>9}   {'delta':>9}")
    print(f"\n{'=' * len(hdr2)}")
    print("  BEST vs FINAL  (the final columns are READ from results.json)")
    print(f"{'=' * len(hdr2)}")
    print(f"  {'':<8}{'':<{aw}}{'':<6}{'':>6}  {'--- best.pt ---':>27}   "
          f"{'--- final.pt ---':>29}")
    print(hdr2)
    for r in rows:
        tau = r["tau"] if r["tau"] is not None else "--"
        print(f"  {str(r['loss_fn']):<8}{r['arch']:<{aw}}{r['mode']:<6}"
              f"{tau:>6}  "
              + fmt(r["best_drop_remote"]) + fmt(r["best_any_flip_rate"], 9, 1)
              + fmt(r["best_visibility"], 9, 4) + "   "
              + fmt(r["final_drop_remote"]) + fmt(r["final_any_flip_rate"], 9, 1)
              + fmt(r["final_visibility"], 9, 4) + "   "
              + fmt(r["best_minus_final_drop_remote"]))
    print("  delta = best.pt drop_remote minus final.pt drop_remote. A "
          "positive delta is the\n  number this script exists to recover; a "
          "zero one means the run never degraded\n  and best.pt is final.pt.")


def main():
    p = argparse.ArgumentParser(
        description="recompute the full result row from each run's best.pt")
    p.add_argument("--overfit_root", default="results/overfit_464",
                   help="searched RECURSIVELY for best.pt, so both the "
                        "<root>/<loss>/<run> and <root>/<loss>/<arch>/<run> "
                        "layouts are found. Ignored when --runs is given.")
    p.add_argument("--runs", nargs="+", default=None,
                   help="explicit run directories (each holding best.pt)")
    p.add_argument("--cityscapes_root", default=None,
                   help="override the root recorded in config.json -- needed "
                        "whenever the runs were produced on another machine")
    p.add_argument("--split", default="val",
                   help="the split the runs indexed with --image. overfit.py "
                        "uses val and nothing else writes these directories.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", default="results/best_eval")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    runs = ([Path(x) for x in a.runs] if a.runs
            else discover(Path(a.overfit_root)))
    jobs = []
    for run in runs:
        if not (run / "best.pt").exists():
            print(f"[best] skip {run}: no best.pt")
            continue
        cfg = run_config(run)
        if not cfg.get("arch"):
            print(f"[best] skip {run}: no config.json/results.json to say "
                  f"which architecture, resolution and image it used")
            continue
        jobs.append((run, cfg))
    if not jobs:
        raise SystemExit("[best] nothing to evaluate.")
    print(f"[best] {len(jobs)} run(s) with a best.pt")

    seed_everything(a.seed)
    device = get_device()
    mean_t, std_t = norm_tensors(device)

    out_dir = Path(increment_path(
        Path(a.out_root) / "_".join(x for x in ["best_vs_final", a.tag] if x)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grouped so the MODEL is built once per (arch, resolution, inference)
    # rather than once per run: six architectures x two losses x two arms is
    # 24 runs and 6 builds, and the build is the expensive part.
    def key(cfg):
        return tuple(str(cfg.get(k)) for k in MODEL_KEYS)
    jobs.sort(key=lambda j: (key(j[1]), str(j[0])))

    rows, cur, model, ds, ns, clean_cache = [], None, None, None, None, {}
    for run, cfg in jobs:
        if key(cfg) != cur:
            cur = key(cfg)
            model = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ns = Namespace(
                arch=cfg["arch"], cfg_path=cfg.get("cfg_path"),
                weights=cfg.get("weights"),
                cityscapes_root=a.cityscapes_root or cfg.get("cityscapes_root"),
                img_h=cfg.get("img_h", 512), img_w=cfg.get("img_w", 1024),
                scale=cfg.get("scale", "resize"),
                inference=cfg.get("inference", "whole"),
                slide_checkpoint=cfg.get("slide_checkpoint", False),
                low_pr=cfg.get("low_pr", False),
                num_classes=cfg.get("num_classes", 19), seed=a.seed)
            model, _, _, _ = setup_model(ns)
            ds = make_dataset(ns, a.split)
            clean_cache = {}

        idx = cfg.get("image")
        if idx is None:
            print(f"[best] skip {run}: the config records no --image")
            continue
        if not 0 <= idx < len(ds):
            raise SystemExit(f"[best] image {idx} out of range for split "
                             f"{a.split} (n={len(ds)}) -- {run}")
        if idx not in clean_cache:
            im, lb = ds[idx]
            im, lb = im.unsqueeze(0).to(device), lb.unsqueeze(0).to(device)
            with torch.no_grad():
                clean_cache[idx] = (im, lb,
                                    upsample_to(model(im), lb.shape[-2:]))
        im, lb, clean_logits = clean_cache[idx]

        patch = Patch.load(str(run / "best.pt"), device, mean_t, std_t)
        if patch.cfg.mode in ("gan", "raw_ganinit"):
            print(f"[best] skip {run}: mode={patch.cfg.mode} needs the BigGAN "
                  f"generator, which this script does not load")
            continue
        m = measure(model, patch, im, lb, ns.num_classes, mean_t, std_t,
                    clean_logits)

        res = load_json(run / "results.json")
        logged = res.get("best_drop_remote")
        # The ONE number that exists on both sides. If the reconstruction
        # picked up a different patch, a different placement or a different
        # base, this is where it shows -- and the row is still written, marked,
        # rather than quietly quoted.
        ok = (logged is None or abs(m["best_drop_remote"] - logged) <= 0.5)
        if not ok:
            print(f"  WARNING {run.name}: recomputed best drop "
                  f"{m['best_drop_remote']:+.2f} but the log recorded "
                  f"{logged:+.2f}. Something about the reconstruction differs "
                  f"from the run; do not quote this row until it is explained.")

        # Recompute final's drop from its own clean/final pair where both are
        # recorded, so the final column is the SAME difference the best column
        # is, rather than a key that might predate a change to either.
        fd = res.get("drop_remote")
        if (res.get("clean_remote") is not None
                and res.get("final_remote") is not None):
            fd = res["clean_remote"] - res["final_remote"]
        row = {
            "run": str(run), "loss_fn": cfg.get("loss_fn"),
            "arch": cfg["arch"], "mode": patch.cfg.mode,
            "tau": (f"{patch.cfg.csf_threshold:g}"
                    if patch.cfg.mode in ("csf", "universal_csf") else None),
            "image": idx, "img_h": ns.img_h, "img_w": ns.img_w,
            "inference": ns.inference, "scale": ns.scale, **m,
            "final_all": res.get("final_all"),
            "final_remote": res.get("final_remote"),
            "final_drop_all": res.get("drop_all"),
            "final_drop_remote": fd,
            "final_any_flip_rate": res.get("any_flip_rate"),
            "final_visibility": res.get("final_visibility"),
            "final_visibility_local": res.get("final_visibility_local"),
            "logged_best_drop_remote": logged,
            "best_minus_final_drop_remote": (m["best_drop_remote"] - fd
                                             if fd is not None else None),
            "degraded_after_peak": res.get("degraded_after_peak"),
            "checkpoint_matches_log": ok,
        }
        rows.append(row)
        with open(run / "best_stats.json", "w") as f:
            json.dump(row, f, indent=2)

        v = row["best_visibility"]
        print(f"  {str(cfg.get('loss_fn')):<8} {cfg['arch']:<14} "
              f"{patch.cfg.mode:<4}"
              + (f" t{row['tau']:<5}" if row["tau"] else " " * 7)
              + f" drop_rem {m['best_drop_remote']:+7.2f}  "
                f"drop_all {m['best_drop_all']:+7.2f}  "
                f"flip {m['best_any_flip_rate']:5.1f}%"
              + (f"  vis {v:.4f}" if v == v else ""))

    if not rows:
        raise SystemExit("[best] no rows produced.")

    rank = {x: i for i, x in enumerate(ARCH_ORDER)}
    rows.sort(key=lambda r: (str(r["loss_fn"]), r["mode"], str(r["tau"]),
                             rank.get(r["arch"], 99), r["arch"]))
    print_table(rows)

    with open(out_dir / "rows.json", "w") as f:
        json.dump({"config": vars(a), "rows": rows}, f, indent=2)
    with open(out_dir / "best_vs_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k) for k in CSV_FIELDS} for r in rows)

    bad = [r["run"] for r in rows if not r["checkpoint_matches_log"]]
    if bad:
        print(f"\n  {len(bad)} row(s) disagree with their run log by more "
              f"than 0.5 mIoU:")
        for b in bad:
            print(f"    {b}")
    print(f"\n  -> {out_dir}/best_vs_final.csv")
    print("  -> one best_stats.json beside each best.pt")


if __name__ == "__main__":
    main()
