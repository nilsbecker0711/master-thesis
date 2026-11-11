#!/usr/bin/env python
"""The transfer matrix: does a patch survive a new IMAGE, a new MODEL, or both?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import add_model_args, setup_model, make_dataset
from patchreach.data.cityscapes import norm_tensors, upsample_to
from patchreach.metrics.miou import single_image_miou, attack_rates
from patchreach.models.registry import REGISTRY
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, seed_everything, increment_path

NAN = float("nan")

ARCH_ORDER = ["segformer_b0", "segformer_b5", "setr_pup",
              "deeplab_18", "deeplab_50", "deeplab_101",
              "internimage_t", "unet_s5d16"]

SHARED_KEYS = ("img_h", "img_w", "inference", "scale")


def run_dir_name(arch: str, mode: str, loss: str, image: int, tau) -> str:
    """The leaf directory overfit.py built from its arguments."""
    parts = [arch, mode, loss, f"img{image}"]
    if mode == "csf":
        parts.append(f"t{tau}")
    return "_".join(parts)


def find_run(root: Path, arch: str, mode: str, loss: str, image: int, tau):
    """The run directory for one (arch, arm), or None."""
    name = run_dir_name(arch, mode, loss, image, tau)
    hits = []
    for parent in (root / loss, root / loss / arch):
        if not parent.is_dir():
            continue
        if ((parent / name) / "best.pt").exists():
            hits.append(parent / name)
            continue
        cand = sorted(d for d in parent.glob(f"{name}_*")
                      if d.is_dir() and (d / "best.pt").exists())
        if len(cand) > 1:
            raise SystemExit(
                f"[matrix] {len(cand)} directories match {parent / name}_* and "
                f"nothing distinguishes them:\n"
                + "".join(f"           {c.name}\n" for c in cand)
                + "         Rename or remove all but one.")
        hits += cand
    if len(hits) > 1:
        raise SystemExit(
            f"[matrix] {arch} resolves to {len(hits)} runs for this arm, in "
            f"both the flat and the nested layout:\n"
            + "".join(f"           {h}\n" for h in hits)
            + "         Keep one. They are different runs.")
    return hits[0] if hits else None


def discover_archs(root: Path, loss: str, mode: str, image: int, tau) -> list:
    """Every registry architecture that HAS a run for this arm."""
    parent = root / loss
    if not parent.is_dir():
        raise SystemExit(f"[matrix] no such directory: {parent}\n"
                         f"         --loss_fn {loss} under --overfit_root "
                         f"{root} finds nothing.")
    found = [a for a in REGISTRY
             if find_run(root, a, mode, loss, image, tau) is not None]
    if not found:
        have = sorted(d.name for d in parent.iterdir() if d.is_dir())[:12]
        raise SystemExit(
            f"[matrix] no run found for mode={mode}"
            + (f" tau={tau}" if mode == "csf" else "")
            + f" under {parent}/\n"
            f"         looking for <arch>_"
            f"{run_dir_name('<arch>', mode, loss, image, tau).split('_', 1)[1]}"
            f"/best.pt\n"
            f"         what is actually there:\n"
            + "".join(f"           {h}\n" for h in have)
            + ("           ...\n" if len(have) == 12 else ""))
    rank = {a: i for i, a in enumerate(ARCH_ORDER)}
    return sorted(found, key=lambda a: (rank.get(a, len(rank)), a))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"[matrix] unreadable JSON, treated as absent: {path}")
        return {}


def source_config(run: Path) -> dict:
    """The overfit run's own argparse namespace, whichever file holds it."""
    r = load_json(run / "results.json")
    if "config" in r:
        return r["config"]
    return load_json(run / "config.json")


@torch.no_grad()
def evaluate_cell(model, patch, img, label, num_classes, mean_t, std_t,
                  clean_logits) -> dict:
    """One (patch, model, image) cell."""
    hw = label.shape[-2:]
    H, W = img.shape[-2:]
    patch.resolve_placement(H, W, clean_logits.argmax(1)[0])
    if patch.cfg.mode == "csf":
        patch.set_reference_from_image(img, mean_t, std_t)
    patched, fp = patch.apply(img)
    adv_logits = upsample_to(model(patched), hw)

    clean = single_image_miou(clean_logits, label, num_classes, exclude=fp)
    adv = single_image_miou(adv_logits, label, num_classes, exclude=fp)
    rates = attack_rates(clean_logits, adv_logits, label, fp)
    st = patch.stats()
    return {"clean_remote": clean, "adv_remote": adv,
            "drop_remote": clean - adv,
            "rel_drop": (clean - adv) / clean if clean else NAN,
            "any_flip_rate": rates["any_flip_rate"],
            "visibility": st.get("visibility", NAN),
            "visibility_local": st.get("visibility_local", NAN)}


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else NAN


def cell(cells, diag, src, tgt, image, train_image):
    """The diagonal on the train image is the READ white-box number."""
    if src == tgt and image == train_image:
        return diag.get(src, NAN)
    return cells.get((src, tgt, image), NAN)


def fmt(v, w=11, d=2):
    return f"{v:>{w}.{d}f}" if v == v else f"{'--':>{w}}"


def table(cells, diag, archs, image, train_image, title, note):
    """rows = trained on, cols = evaluated on."""
    w = max(12, max(len(x) for x in archs) + 2)
    print(f"\n  {title}")
    print(f"  {note}")
    print(f"  {'':<{w}}" + "".join(f"{t[:w - 1]:>{w}}" for t in archs)
          + f"{'off-diag':>{w}}")
    for s in archs:
        row = f"  {s:<{w}}"
        for t in archs:
            row += fmt(cell(cells, diag, s, t, image, train_image), w)
        row += fmt(mean([cell(cells, diag, s, t, image, train_image)
                         for t in archs if t != s]), w)
        print(row)
    foot = f"  {'off-diag':<{w}}"
    for t in archs:
        foot += fmt(mean([cell(cells, diag, s, t, image, train_image)
                          for s in archs if s != t]), w)
    print(foot)
    print(f"  row mean = how far this SOURCE's patch carries; "
          f"column mean = how exposed this TARGET is")


def retention(archs, cells, vis, diag, train_image, transfer_image):
    """The four quadrants side by side, per source, as a fraction of white box."""
    rows = []
    for s in archs:
        wb = diag.get(s, NAN)
        img_t = cells.get((s, s, transfer_image), NAN)
        mod_t = mean([cells.get((s, t, train_image), NAN)
                      for t in archs if t != s])
        joint = mean([cells.get((s, t, transfer_image), NAN)
                      for t in archs if t != s])
        ok = wb == wb and wb
        rows.append({"source": s, "white_box": wb, "image_transfer": img_t,
                     "model_transfer": mod_t, "joint_transfer": joint,
                     "retained_image_pct": 100 * img_t / wb if ok else NAN,
                     "retained_model_pct": 100 * mod_t / wb if ok else NAN,
                     "retained_joint_pct": 100 * joint / wb if ok else NAN,
                     "vis_train": mean([vis.get((s, t, train_image), NAN)
                                        for t in archs]),
                     "vis_transfer": mean([vis.get((s, t, transfer_image), NAN)
                                           for t in archs])})
    return rows


RET_KEYS = ("white_box", "image_transfer", "model_transfer", "joint_transfer")
PCT_KEYS = ("retained_image_pct", "retained_model_pct", "retained_joint_pct")
VIS_KEYS = ("vis_train", "vis_transfer")


def print_retention(rows, tau, train_image, transfer_image):
    w = max(12, max(len(r["source"]) for r in rows) + 2)
    print("\n  RETENTION  (drop_remote, and what fraction of the white-box "
          "drop survives)")
    print(f"  {'source':<{w}}{'white box':>11}{'image':>11}{'model':>11}"
          f"{'joint':>11}   {'img%':>7}{'mod%':>7}{'joint%':>8}"
          f"   {'vis' + str(train_image):>9}{'vis' + str(transfer_image):>9}")
    body = lambda r: ("".join(fmt(r[k]) for k in RET_KEYS) + "   "
                      + "".join(fmt(r[k], n, 1)
                                for k, n in zip(PCT_KEYS, (7, 7, 8)))
                      + "   " + "".join(fmt(r[k], 9, 4) for k in VIS_KEYS))
    for r in rows:
        print(f"  {r['source']:<{w}}" + body(r))
    agg = {k: mean([r[k] for r in rows])
           for k in RET_KEYS + PCT_KEYS + VIS_KEYS}
    print(f"  {'MEAN':<{w}}" + body(agg))
    if tau is not None and agg["vis_transfer"] == agg["vis_transfer"]:
        over = [r["source"] for r in rows
                if r["vis_transfer"] > float(tau) * 1.02]
        print(f"  tau = {tau}; mean realised visibility "
              f"{agg['vis_train']:.4f} on img{train_image} -> "
              f"{agg['vis_transfer']:.4f} on img{transfer_image}")
        if over:
            print(f"  NOTE realised visibility EXCEEDS tau on "
                  f"img{transfer_image} for: {', '.join(over)} -- their image "
                  f"transfer is not measured at the budget they were trained "
                  f"under.")
    return agg


def main():
    p = add_model_args(argparse.ArgumentParser(
        description="transfer matrix for one (loss, patch mode, tau) arm"))
    p.set_defaults(img_h=None, img_w=None, inference=None, scale=None)

    p.add_argument("--loss_fn", required=True,
                   help="the loss every patch in this matrix was trained with; "
                        "also the first path component under --overfit_root")
    p.add_argument("--csf", action="store_true",
                   help="the CSF-bounded arm. Needs --tau, and looks for run "
                        "directories ending _t<tau>. Without it the arm is raw "
                        "(unconstrained) and the directories have no tau "
                        "suffix.")
    p.add_argument("--tau", default=None,
                   help="--csf only: the csf_threshold, AS WRITTEN in the run "
                        "directory name. A string, not a float, so 0.5 and "
                        "0.50 stay distinguishable and both round-trip to the "
                        "path that exists.")

    p.add_argument("--train_image", type=int, default=464,
                   help="the image every patch was optimised on")
    p.add_argument("--transfer_image", type=int, default=42,
                   help="the held-out image")
    p.add_argument("--split", default="val")
    p.add_argument("--overfit_root", default="results/overfit_464")
    p.add_argument("--archs", nargs="+", default=None,
                   help="override the architecture list. Default: every "
                        "registry architecture that HAS a checkpoint for this "
                        "arm. Naming one whose run is missing leaves its row "
                        "blank but keeps its column.")
    p.add_argument("--out_root", default="results/matrix_464")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    mode = "csf" if a.csf else "raw"
    if a.csf and a.tau is None:
        p.error("--csf needs --tau: the run directory name carries it, so "
                "without it there is no path to look up.")
    if not a.csf and a.tau is not None:
        p.error("--tau is meaningless without --csf (raw runs ignore "
                "--csf_threshold and their directories have no _t suffix). "
                "Drop --tau, or add --csf.")
    if a.tau is not None:
        try:
            float(a.tau)
        except ValueError:
            p.error(f"--tau must parse as a number, got {a.tau!r}")
    arm = f"csf_t{a.tau}" if a.csf else "raw"
    if a.train_image == a.transfer_image:
        p.error("--train_image and --transfer_image are the same image; there "
                "is no image transfer to measure.")

    root = Path(a.overfit_root)
    archs = a.archs or discover_archs(root, a.loss_fn, mode, a.train_image,
                                      a.tau)
    unknown = [x for x in archs if x not in REGISTRY]
    if unknown:
        raise SystemExit(f"[matrix] not in the registry: {', '.join(unknown)}")

    ckpts, diag, missing, src_cfg = {}, {}, [], {}
    for src in archs:
        run = find_run(root, src, mode, a.loss_fn, a.train_image, a.tau)
        if run is None:
            missing.append(str(root / a.loss_fn /
                               run_dir_name(src, mode, a.loss_fn,
                                            a.train_image, a.tau)))
            ckpts[src] = None
            continue
        ckpts[src] = run / "best.pt"
        src_cfg[src] = source_config(run)
        r = load_json(run / "results.json")
        if "best_drop_remote" in r:
            diag[src] = r["best_drop_remote"]
        elif "drop_remote" in r:
            diag[src] = r["drop_remote"]
            print(f"[matrix] {src}: no best_drop_remote in results.json -- "
                  f"falling back to drop_remote, which is final.pt's number, "
                  f"not best.pt's.")

    present = [s for s in archs if ckpts[s] is not None]
    print(f"\n[matrix] arm: loss={a.loss_fn}  mode={mode}"
          + (f"  tau={a.tau}" if a.csf else ""))
    print(f"[matrix] {len(present)}/{len(archs)} sources x {len(archs)} "
          f"targets x 2 images")
    print(f"[matrix] train image {a.train_image}, transfer image "
          f"{a.transfer_image}")
    for s in archs:
        d = diag.get(s, NAN)
        print(f"           {'ok     ' if ckpts[s] else 'MISSING'} {s:<14s} "
              f"white box " + (f"{d:+7.2f}" if d == d else "     --")
              + (f"   {ckpts[s].parent.name}" if ckpts[s] else ""))
    if not present:
        raise SystemExit("[matrix] no checkpoints found -- nothing to do.")
    if missing:
        print(f"[matrix] {len(missing)} missing run(s); their ROWS are blank "
              f"but their COLUMNS are not -- a missing source is still a "
              f"target.")

    agreed = {}
    for k in SHARED_KEYS:
        vals = {s: c.get(k) for s, c in src_cfg.items() if c.get(k) is not None}
        if len(set(vals.values())) > 1:
            raise SystemExit(
                f"[matrix] the source runs disagree on {k}, so their patches "
                f"are not commensurable:\n"
                + "".join(f"           {s:<14s} {v}\n"
                          for s, v in sorted(vals.items()))
                + f"         Pass --{k} explicitly to force one, and say so in "
                  f"the caption.")
        if vals:
            agreed[k] = next(iter(vals.values()))
    for k in SHARED_KEYS:
        if getattr(a, k) is None:
            if k not in agreed:
                raise SystemExit(f"[matrix] no source run records {k} and none "
                                 f"was passed. Pass --{k}.")
            setattr(a, k, agreed[k])
            print(f"[matrix] {k} = {agreed[k]} (adopted from the source runs)")
        elif k in agreed and getattr(a, k) != agreed[k]:
            print(f"[matrix] WARNING {k} = {getattr(a, k)} overrides the source "
                  f"runs' {agreed[k]}. The read white-box diagonal was "
                  f"measured at {agreed[k]} and is NOT comparable to these "
                  f"cells.")

    seed_everything(a.seed)
    device = get_device()
    mean_t, std_t = norm_tensors(device)

    out_dir = Path(increment_path(Path(a.out_root) / a.loss_fn /
                                 "_".join(x for x in [arm, a.tag] if x)))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = make_dataset(a, a.split)
    for idx in (a.train_image, a.transfer_image):
        if not 0 <= idx < len(ds):
            raise SystemExit(f"[matrix] image {idx} is out of range for split "
                             f"{a.split} (n={len(ds)})")
    images = [a.train_image, a.transfer_image]
    imgs = {}
    for idx in images:
        im, lb = ds[idx]
        imgs[idx] = (im.unsqueeze(0).to(device), lb.unsqueeze(0).to(device))

    cells, vis, records, clean_ref = {}, {}, [], {}

    for tgt in archs:
        a.arch = tgt
        a.cfg_path = a.weights = None
        model, n_ch, n_act, spec = setup_model(a)

        clean = {}
        with torch.no_grad():
            for idx, (im, lb) in imgs.items():
                clean[idx] = upsample_to(model(im), lb.shape[-2:])
                clean_ref[(tgt, idx)] = single_image_miou(clean[idx], lb,
                                                          a.num_classes)
        print(f"  clean mIoU  img{a.train_image} "
              f"{clean_ref[(tgt, a.train_image)]:.2f}   "
              f"img{a.transfer_image} "
              f"{clean_ref[(tgt, a.transfer_image)]:.2f}   (no patch)")

        for image in images:
            for src in present:
                if src == tgt and image == a.train_image:
                    continue
                kind = ("image_transfer" if src == tgt else
                        "model_transfer" if image == a.train_image else
                        "joint_transfer")
                f = ckpts[src]
                patch = Patch.load(str(f), device, mean_t, std_t)
                if patch.cfg.mode != mode:
                    raise SystemExit(
                        f"[matrix] {f} is mode={patch.cfg.mode!r}, expected "
                        f"{mode!r}. The directory name and the checkpoint "
                        f"disagree.")
                if mode == "csf" and abs(patch.cfg.csf_threshold
                                         - float(a.tau)) > 1e-9:
                    raise SystemExit(
                        f"[matrix] {f} has csf_threshold="
                        f"{patch.cfg.csf_threshold:g}, but --tau {a.tau} put it "
                        f"at this path. One of the two is wrong.")
                im, lb = imgs[image]
                res = evaluate_cell(model, patch, im, lb, a.num_classes,
                                    mean_t, std_t, clean[image])
                cells[(src, tgt, image)] = res["drop_remote"]
                vis[(src, tgt, image)] = res["visibility"]
                records.append({"loss": a.loss_fn, "mode": mode, "tau": a.tau,
                                "source": src, "target": tgt, "image": image,
                                "kind": kind, "checkpoint": str(f), **res})
                v = res["visibility"]
                print(f"    {kind:15s} {src:14s} -> {tgt:14s} img{image:<4d} "
                      f"drop {res['drop_remote']:+7.2f}  "
                      f"({100 * res['rel_drop']:5.1f}% of clean)  "
                      f"flip {res['any_flip_rate']:5.1f}%"
                      + (f"  vis {v:.4f}" if v == v else ""))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hdr = (f"loss={a.loss_fn}  mode={mode}"
           + (f"  tau={a.tau}" if a.csf else "")
           + f"  {a.img_h}x{a.img_w}  scale={a.scale}  "
             f"inference={a.inference}")
    print(f"\n{'=' * 100}\n  {hdr}\n{'=' * 100}")
    table(cells, diag, archs, a.train_image, a.train_image,
          f"TRAIN IMAGE {a.train_image}   drop_remote",
          "diagonal = WHITE BOX (read from the overfit run), "
          "off-diagonal = MODEL transfer")
    table(cells, diag, archs, a.transfer_image, a.train_image,
          f"TRANSFER IMAGE {a.transfer_image}   drop_remote",
          "diagonal = IMAGE transfer, off-diagonal = JOINT "
          "(new model AND new image)")
    rows = retention(archs, cells, vis, diag, a.train_image, a.transfer_image)
    agg = print_retention(rows, a.tau, a.train_image, a.transfer_image)
    print(f"{'=' * 100}")

    payload = {"config": vars(a), "arm": arm, "mode": mode,
               "archs": archs, "sources_present": present,
               "adopted_from_sources": agreed,
               "white_box_diagonal": diag,
               "retention": rows, "retention_mean": agg,
               "clean_remote_no_patch": {f"{t}|{i}": v
                                         for (t, i), v in clean_ref.items()},
               "cells": records, "missing": missing}
    (out_dir / "matrix.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  -> {out_dir}/matrix.json")


if __name__ == "__main__":
    main()
