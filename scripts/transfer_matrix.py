#!/usr/bin/env python
r"""
The transfer matrix: does a patch survive a new IMAGE, or a new MODEL?

    python scripts/transfer_matrix.py --cityscapes_root $CS \
        --archs segformer_b0 segformer_b5 deeplabv3plus_r101 \
                internimage_t setr_pup \
        --losses cospgd ce --train_image 42 --transfer_image 420

WHY THIS EXISTS RATHER THAN A LOOP OVER evaluate.py. Each cell is ONE forward
pass and each evaluate.py invocation is one model build + checkpoint load. At
5 archs x 5 sources x 2 losses that is 50 loads to do 50 forward passes, and
the loads dominate by two orders of magnitude. Here the TARGET model is the
outer loop, so the matrix costs len(archs) loads — 5 instead of 50.

TWO CELLS, TWO QUESTIONS, AND THEY ARE NOT THE SAME QUESTION:

    source == target  ->  evaluated on --transfer_image
                          A NEW IMAGE, the same model. This is IMAGE transfer,
                          and it is the one that forecasts universal_csf: a
                          per-image patch composited onto a scene it was not
                          optimised for is a one-sample universal patch.

    source != target  ->  evaluated on --train_image
                          The SAME image, a new model. This is MODEL transfer,
                          with the image held fixed so nothing of the ~10-point
                          scene-to-scene spread leaks into the comparison.

THE DIAGONAL ON --train_image IS NOT RECOMPUTED. It is the white-box number the
overfit run already reported, so it is read from that run's results.json. Every
off-diagonal cell is read against it; recomputing it would be one more chance
for the two to disagree.

THE BASE IS REBUILT PER IMAGE. A csf checkpoint stores the parameter and the
placement, never the base — the base is the image region the patch covers. This
script calls set_reference_from_image() after resolve_placement() for exactly
the reason optimise.prepare() does, and in that order. Without it the base is
flat grey and every number describes a visible square rather than the residual
under test. That is the whole point of a transfer matrix, so it is automatic
here rather than a flag: mode='csf' always rebuilds.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import torch

from _common import add_model_args, setup_model
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.metrics.miou import single_image_miou, attack_rates
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, seed_everything, increment_path

NAN = float("nan")


def ckpt_dir(root: Path, arch: str, loss: str, image: int,
             patch_mode: str, qtag: str) -> Path:
    """Mirror the run-directory name overfit.py builds from its arguments."""
    parts = [arch, patch_mode, loss, f"img{image}"]
    if loss == "tsallis" and qtag:
        parts.append(qtag)
    return root / "_".join(parts)


def load_row(path: Path) -> dict:
    """The overfit run's own white-box numbers, or {} if it never finished."""
    f = path / "results.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return {}


@torch.no_grad()
def evaluate_cell(model, patch, img, label, num_classes, mean_t, std_t,
                  img_h, img_w, clean_logits) -> dict:
    """One (patch, model, image) cell. `clean_logits` is cached by the caller."""
    hw = label.shape[-2:]
    # ORDER IS LOAD-BEARING: resolve_placement first (semantic/gradcam read the
    # clean prediction), then the base is copied from the RESOLVED region.
    patch.resolve_placement(img_h, img_w, clean_logits.argmax(1)[0])
    if patch.cfg.mode == "csf":
        patch.set_reference_from_image(img, mean_t, std_t)
    patched, fp = patch.apply(img)
    adv_logits = upsample_to(model(patched), hw)

    clean = single_image_miou(clean_logits, label, num_classes, exclude=fp)
    adv = single_image_miou(adv_logits, label, num_classes, exclude=fp)
    rates = attack_rates(clean_logits, adv_logits, label, fp)
    return {"clean_remote": clean, "adv_remote": adv,
            "drop_remote": clean - adv,
            "rel_drop": (clean - adv) / clean if clean else NAN,
            "any_flip_rate": rates["any_flip_rate"]}


def table(cells, archs, loss, image, title, diag=None):
    """rows = trained on, cols = evaluated on."""
    w = max(13, max(len(a) for a in archs) + 2)
    print(f"\n  {title}   loss={loss}  image={image}")
    print(f"  {'':<{w}}" + "".join(f"{t[:w-2]:>{w}}" for t in archs))
    for s in archs:
        row = f"  {s:<{w}}"
        for t in archs:
            v = cells.get((loss, s, t, image), NAN)
            if s == t and diag is not None:
                v = diag.get((loss, s), v)
            row += f"{v:>{w}.2f}" if v == v else f"{'--':>{w}}"
        print(row)


def main():
    p = add_model_args(argparse.ArgumentParser())
    p.add_argument("--archs", nargs="+", required=True,
                   help="every arch is both a SOURCE (its patch) and a TARGET "
                        "(its weights). The matrix is archs x archs.")
    p.add_argument("--losses", nargs="+", default=["cospgd", "ce"],
                   help="one patch per (arch, loss). Add 'tsallis' once the q "
                        "sweep has picked a value, and pass --tsallis_qtag.")
    p.add_argument("--tsallis_qtag", default="q0",
                   help="the suffix tsallis_tag() put on the run directory, "
                        "e.g. q0 or q-2to1. Ignored for other losses.")
    p.add_argument("--train_image", type=int, default=42,
                   help="the image every patch was optimised on")
    p.add_argument("--transfer_image", type=int, default=420,
                   help="a held-out image, used only for source == target")
    p.add_argument("--overfit_root", default="results/overfit")
    p.add_argument("--patch_mode", default="csf",
                   help="only to rebuild the run-directory name overfit.py used")
    p.add_argument("--out_root", default="results/matrix")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    seed_everything(a.seed)
    device = get_device()
    mean_t, std_t = norm_tensors(device)
    root = Path(a.overfit_root)

    out_dir = Path(increment_path(Path(a.out_root) /
                                  "_".join(x for x in ["matrix", a.tag] if x)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── locate every source patch up front, so a missing run fails here and
    #    not four model loads deep ────────────────────────────────────────────
    ckpts, diag, missing = {}, {}, []
    for loss, src in product(a.losses, a.archs):
        d = ckpt_dir(root, src, loss, a.train_image, a.patch_mode, a.tsallis_qtag)
        f = d / "best.pt"
        (ckpts.setdefault(loss, {}))[src] = f if f.exists() else None
        if not f.exists():
            missing.append(str(f))
            continue
        r = load_row(d)
        if "drop_remote" in r:
            diag[(loss, src)] = r["drop_remote"]

    print(f"\n[matrix] {len(a.archs)} archs x {len(a.losses)} losses")
    print(f"[matrix] train image {a.train_image} (model transfer), "
          f"transfer image {a.transfer_image} (image transfer)")
    if missing:
        print(f"[matrix] {len(missing)} MISSING checkpoint(s) — those cells "
              f"will be blank:")
        for m in missing:
            print(f"           {m}")
    print(f"[matrix] white-box diagonal read from overfit results.json: "
          f"{len(diag)}/{len(a.losses) * len(a.archs)} found")

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    imgs = {}
    for idx in {a.train_image, a.transfer_image}:
        im, lb = ds[idx]
        imgs[idx] = (im.unsqueeze(0).to(device), lb.unsqueeze(0).to(device))

    cells, records = {}, []

    # TARGET is the outer loop: one model build per arch, not one per cell.
    for tgt in a.archs:
        a.arch = tgt
        a.cfg_path = a.weights = None          # always resolve from the registry
        model, n_ch, n_act, spec = setup_model(a)

        # Clean logits do not depend on the patch, so compute them once per
        # (target, image) rather than once per cell.
        clean = {}
        with torch.no_grad():
            for idx, (im, lb) in imgs.items():
                clean[idx] = upsample_to(model(im), lb.shape[-2:])

        for loss in a.losses:
            for src in a.archs:
                f = ckpts[loss][src]
                image = a.transfer_image if src == tgt else a.train_image
                kind = "image_transfer" if src == tgt else "model_transfer"
                if f is None:
                    continue
                patch = Patch.load(str(f), device, mean_t, std_t)
                im, lb = imgs[image]
                res = evaluate_cell(model, patch, im, lb, a.num_classes,
                                    mean_t, std_t, a.img_h, a.img_w,
                                    clean[image])
                cells[(loss, src, tgt, image)] = res["drop_remote"]
                records.append({"loss": loss, "source": src, "target": tgt,
                                "image": image, "kind": kind,
                                "checkpoint": str(f), **res})
                print(f"    {loss:8s} {src:20s} -> {tgt:20s} img{image:<4d} "
                      f"drop {res['drop_remote']:+7.2f}  "
                      f"({100*res['rel_drop']:5.1f}% of clean)  "
                      f"flip {res['any_flip_rate']:5.1f}%")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    for loss in a.losses:
        table(cells, a.archs, loss, a.train_image,
              "MODEL TRANSFER  (diagonal = white-box, from the overfit run)",
              diag=diag)
        print(f"\n  IMAGE TRANSFER  loss={loss}  image={a.transfer_image} "
              f"(same model, held-out image)")
        for s in a.archs:
            v = cells.get((loss, s, s, a.transfer_image), NAN)
            d = diag.get((loss, s), NAN)
            ratio = v / d if (d == d and d) else NAN
            print(f"    {s:22s} drop {v:7.2f}   white-box {d:7.2f}   "
                  + (f"retained {100*ratio:5.1f}%" if ratio == ratio else "retained    --"))
    print(f"{'=' * 78}")

    payload = {"config": vars(a), "cells": records,
               "white_box_diagonal": {f"{l}|{s}": v for (l, s), v in diag.items()},
               "missing": missing}
    (out_dir / "matrix.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  -> {out_dir}/matrix.json")


if __name__ == "__main__":
    main()
