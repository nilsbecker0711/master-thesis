#!/usr/bin/env python
"""Per-image attacks over a POPULATION of images — overfit.py at sample size."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (add_model_args, add_patch_args, setup_model, build_patch,
                     image_indices, tsallis_kwargs, tsallis_tag)
from _common import make_dataset
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import aggregate, report
from patchreach.metrics import population as pop_mod
from patchreach.patch import optimise, segmentation_cam
from patchreach.patch.spec import Patch
from patchreach.utils import (get_device, seed_everything, increment_path,
                              atomic_save)


def build_parser():
    p = add_patch_args(add_model_args(argparse.ArgumentParser(
        description="Optimise one patch per image over a population of "
                    "images and report the distribution")))
    p.add_argument("--loss_fn", default="cospgd",
                   choices=["ce", "cospgd", "ce_target",
                            "ipatch_cospgd", "tsallis"])
    p.add_argument("--target_class", type=int, default=8)
    p.add_argument("--from_image", action="store_true",
                   help="initialise the patch base with the region it covers "
                        "(csf modes)")

    p.add_argument("--images", default="random",
                   help="'random' (seeded sample of --n_images) | 'fixed10' | "
                        "'all' | an explicit list like '2 5 45'")
    p.add_argument("--n_images", type=int, default=100,
                   help="population size. 500 = the full Cityscapes val set, "
                        "which is what the field does when it can afford to; "
                        "the run reports whether this n supports the claim.")
    p.add_argument("--sample_seed", type=int, default=0,
                   help="seed for --images random. RECORD THIS — it is what "
                        "makes the subset reproducible.")

    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimiser", default="adam",
                   choices=["adam", "sign"],
                   help="step rule. 'adam' is every run to date; 'sign' "
                        "is PGD's update, matched at equal lr. THIS is "
                        "the script the optimiser ablation should use: a "
                        "single image cannot settle it, because four "
                        "identical single-image invocations already span "
                        "9.6 points of remote drop (see "
                        "attack_image's lr_schedule note). Pair with "
                        "--pixel_param direct for the full PGD recipe.")
    p.add_argument("--lr_schedule", default="cosine",
                   choices=["none", "cosine"],
                   help="Anneal lr to zero over each image's run. 'none' "
                        "restores the flat-lr regime, which was measured "
                        "swinging 9.6 mIoU points across four identical "
                        "single-image runs.")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--exclude_footprint", action="store_true", default=True)
    p.add_argument("--verbose_images", action="store_true",
                   help="full per-step logging for every image. Off by "
                        "default: at n=100 it is 100x the output of overfit.py")

    p.add_argument("--n_panels", type=int, default=3,
                   help="how many images get panels and the full diagnostic "
                        "suite. 0 disables.")
    p.add_argument("--panel_select", default="spread",
                   choices=list(pop_mod.SELECT_MODES),
                   help="which images those are. 'spread' = best + median + "
                        "worst; 'best' is what you want to look at and what a "
                        "reviewer will call cherry-picking.")
    p.add_argument("--no_aggregate", dest="aggregate", action="store_false",
                   default=True,
                   help="skip the pooled diagnostic suite. It costs a few "
                        "tensor reductions per image against a full "
                        "optimisation, so the only reason to is a rerun of an "
                        "existing directory whose aggregate state is gone.")
    p.add_argument("--classes", default="gt", choices=["gt", "union"],
                   help="which classes enter the mIoU mean. 'gt' counts a "
                        "class iff it has ground truth, so clean and attacked "
                        "share a denominator and their difference is a "
                        "measurement; 'union' is the pre-fix convention and is "
                        "reported alongside either way.")
    p.add_argument("--select_key", default="drop_remote",
                   help="metric ranking images for --panel_select")

    p.add_argument("--resume", action="store_true",
                   help="continue into an existing --out_dir, skipping images "
                        "already done")
    p.add_argument("--out_dir", default=None,
                   help="explicit output directory (required with --resume)")
    p.add_argument("--out_root", default="results/population")
    p.add_argument("--tag", default="")
    return p


def resolve_images(a, n_val: int):
    """The image list, and it is recorded in config.json either way."""
    if a.images == "random":
        g = torch.Generator().manual_seed(a.sample_seed)
        perm = torch.randperm(n_val, generator=g).tolist()
        return sorted(perm[:min(a.n_images, n_val)])
    return image_indices(a.images, n_val)[:a.n_images]


def main():
    a = build_parser().parse_args()
    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    ds = make_dataset(a, "val")
    idxs = resolve_images(a, len(ds))

    if a.out_dir:
        out_dir = Path(a.out_dir)
    else:
        arm = "_".join(x for x in
                       ["" if a.optimiser == "adam" else a.optimiser,
                        "" if a.pixel_param == "sigmoid"
                        else a.pixel_param] if x)
        tag = "_".join(x for x in [a.arch, a.patch_mode, a.loss_fn, arm,
                                   f"n{len(idxs)}", tsallis_tag(a),
                                   a.tag] if x)
        out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = out_dir / "patches"
    patch_dir.mkdir(exist_ok=True)
    state_path = out_dir / "population_state.pt"
    agg_path = out_dir / "aggregate_state.pt"

    target = a.target_class if a.loss_fn == "ipatch_cospgd" else None
    pop = pop_mod.Population(a.num_classes, device=device, target_class=target)
    diag = aggregate.PopulationDiagnostics(a.num_classes) if a.aggregate else None

    if a.resume and state_path.exists():
        pop.load_state_dict(torch.load(state_path, map_location="cpu"))
        print(f"[resume] {len(pop.records)} images already done in {out_dir}")
        if diag is not None:
            if agg_path.exists():
                diag.load_state_dict(torch.load(agg_path, map_location="cpu"))
                print(f"[resume] aggregate diagnostics carry "
                      f"{diag.n_images} of them")
            elif pop.records:
                print(f"  [!] no aggregate_state.pt beside "
                      f"{len(pop.records)} finished images. The pooled "
                      f"diagnostics will cover")
                print(f"      ONLY the images this invocation attacks, while "
                      f"summary.json covers all of them.")
                print(f"      Delete the directory and rerun, or pass "
                      f"--no_aggregate and accept the gap.")
    todo = [i for i in idxs if i not in pop.done_images]

    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(a), "images": idxs, "bracket": spec.bracket}, f,
                  indent=2)

    print(f"\n{'=' * 72}")
    print(f" {len(todo)} images to attack "
          f"({len(idxs)} requested, {len(idxs) - len(todo)} already done)")
    print(f" one patch per image, {a.steps} steps each — "
          f"{a.arch} ({spec.bracket} attention)")
    print(f"{'=' * 72}")

    cam = None
    if a.placement == "gradcam":
        cam = segmentation_cam.build(
            model, a.loss_fn if a.cam_objective == "attack" else a.cam_objective,
            a.target_class, a.cam_layer, a.cam_module, a.cam_target,
            tsallis=tsallis_kwargs(a, a.steps))
        print(f"[cam ] objective  : {a.cam_objective}  target: {a.cam_target}  "
              f"layer {a.cam_layer} of {a.cam_module}")
        print(f"[cam ] margin     : {a.placement_margin}px keep-out from the "
              f"image border")

    t0 = time.time()
    for n, i in enumerate(todo, 1):
        img, label = ds[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        patch = build_patch(a, device, mean_t, std_t)
        clean_logits = optimise.prepare(model, img, patch, a.img_h, a.img_w,
                                        a.from_image, mean_t, std_t,
                                        cam=cam, label=label)

        rec = optimise.attack_image(
            model, img, label, patch,
            loss_fn=a.loss_fn, target_class=a.target_class, steps=a.steps,
            lr=a.lr, optimiser=a.optimiser, num_classes=a.num_classes,
            exclude_footprint=a.exclude_footprint, log_every=a.log_every,
            lr_schedule=a.lr_schedule, classes=a.classes,
            clean_logits=clean_logits, out_dir=patch_dir / f"img{i:04d}",
            save_step_images=False, save_best=True,
            verbose=a.verbose_images)
        rec["image"] = i

        with torch.no_grad():
            patched, fp = patch.apply(img)
            adv_logits = upsample_to(model(patched), label.shape[-2:])
        pop.update(clean_logits, adv_logits, label, fp, rec)
        if diag is not None:
            diag.update(clean_logits, adv_logits, label, fp, rec)

        atomic_save(pop.state_dict(), state_path)
        if diag is not None:
            atomic_save(diag.state_dict(), agg_path)
        with open(out_dir / "records.jsonl", "a") as f:
            f.write(json.dumps({k: v for k, v in rec.items()
                                if k != "history"}) + "\n")

        el = time.time() - t0
        eta = el / n * (len(todo) - n)
        print(f"  [{n:4d}/{len(todo)}] image {i:4d}  "
              f"drop_remote {rec['drop_remote']:+7.2f}  "
              f"best {rec['best_drop_remote']:+7.2f}  "
              f"flip {rec['any_flip_rate']:5.1f}%   "
              f"({el/n:.0f}s/img, eta {eta/60:.0f}m)")

    if cam is not None:
        if getattr(cam, "n_degenerate", 0):
            print(f"\n  NOTE: the sensitivity map was degenerate (ReLU "
                  f"suppressed every channel) on {cam.n_degenerate} images;")
            print(f"        those fell back to CENTRE placement and are not "
                  f"gradcam-placed. They are still in the population.")
        cam.close()

    summary = pop.summarise(key=a.select_key)

    panels = pop_mod.select(pop.records, a.n_panels, a.panel_select,
                            a.select_key)
    pop_mod.plot_distribution(pop.records, out_dir / "distribution.png",
                              key=a.select_key,
                              title=f"{a.arch} — {a.patch_mode} / {a.loss_fn}",
                              highlight=[r["image"] for r in panels])

    agg = None
    if diag is not None and diag.n_images:
        agg_dir = out_dir / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        lines = []

        def alog(s=""):
            print(s)
            lines.append(str(s))

        title = (f"{a.arch} — {a.patch_mode} / {a.loss_fn}"
                 + (f" / tau {a.csf_threshold:g}"
                    if "csf" in a.patch_mode else "")
                 + f" / {a.placement}")
        agg = diag.summarise(
            tau=a.csf_threshold if "csf" in a.patch_mode else None,
            records=pop.records, log=alog)
        agg["per_class_iou_pooled"] = aggregate.pooled_per_class_iou(
            pop.clean_rem, pop.adv_rem, agg_dir / "per_class_iou.png",
            target_class=target, title=f"pooled over {diag.n_images} images, "
                                       f"remote px", log=alog)
        figs = diag.write_figures(
            agg_dir, tau=a.csf_threshold if "csf" in a.patch_mode else None,
            records=pop.records, title=title)
        if diag.n_images != len(pop.records):
            alog(f"\n  [!] the aggregate covers {diag.n_images} images but "
                 f"summary.json covers {len(pop.records)} — a resume without "
                 f"aggregate_state.pt. Do not quote the two side by side.")
        (agg_dir / "aggregate.txt").write_text("\n".join(lines),
                                               encoding="utf-8")
        with open(agg_dir / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n  aggregate -> {agg_dir}/")
        for nme in ["per_class_iou.png"] + figs:
            print(f"    {nme}")

    if panels:
        print(f"\n  diagnostics for {len(panels)} images "
              f"({a.panel_select}): {[r['image'] for r in panels]}")
    for rank, r in enumerate(panels):
        i = r["image"]
        img, label = ds[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        ck = patch_dir / f"img{i:04d}" / "best.pt"
        if not ck.exists():
            print(f"    image {i}: no checkpoint, skipped")
            continue
        patch = Patch.load(ck, device, mean_t, std_t)
        if a.from_image:
            patch.set_reference_from_image(img, mean_t, std_t)

        d = out_dir / "diagnostics" / f"img{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            clean_logits = upsample_to(model(img), label.shape[-2:])
            patched, fp = patch.apply(img)
            adv_logits = upsample_to(model(patched), label.shape[-2:])
        report.per_class_iou_figure(clean_logits, adv_logits, label,
                                    a.num_classes, d / "per_class_iou.png",
                                    target, title=f"val image {i}")
        report.run(model, img, label, patch, d, a.loss_fn, a.num_classes,
                   target, mean_t, std_t, skip_geometric=(rank > 0),
                   log=lambda *_, **__: None)
        print(f"    image {i}: drop_remote {r['drop_remote']:+.2f} -> {d}/")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"config": {**vars(a), "images": idxs},
                   "summary": summary,
                   "aggregate": ({"n_images": agg["n_images"],
                                  "file": "aggregate/aggregate.json"}
                                 if agg else None),
                   "panels": [r["image"] for r in panels],
                   "records": [{k: v for k, v in r.items() if k != "history"}
                               for r in pop.records]}, f, indent=2)
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()
