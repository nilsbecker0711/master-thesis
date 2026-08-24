#!/usr/bin/env python
r"""
Per-image attacks over a POPULATION of images — overfit.py at sample size.

    python scripts/overfit_population.py --arch segformer --cityscapes_root $CS \
        --patch_mode csf --from_image --loss_fn cospgd \
        --images random --n_images 100 --steps 300

WHAT THIS IS
------------
ONE PATCH PER IMAGE, optimised independently, exactly as scripts/overfit.py
does it — the same function, patchreach/patch/optimise.attack_image(), so the
two cannot drift. This is BASELINE B at sample size: the approximate upper
bound on what direct optimisation achieves, now with an interval around it
instead of an anecdote.

It is NOT a universal patch. train.py optimises one tensor across the dataset
and applies it unchanged to every image; that is a different threat model with
a different (much harder) problem, and the two sets of numbers must never be
merged. See the note in train_conditional_generator.py for the three-way
distinction.

HOW MANY IMAGES
---------------
Cityscapes val is 500 images. Two reference points from the literature:

  Nesti et al., WACV 2022   patches optimised on 250 images sampled from the
                            training set; "the entire validation set was used
                            to evaluate the effectiveness of the patches".
  Gu et al., ECCV 2024      "Following general practice, we evaluate over the
    (Trying Harder Pays Off) validation sets of PASCAL VOC 2012 and
                            Cityscapes" — i.e. the full set, no subsampling.

The convention is therefore the FULL validation set where the attack is cheap
enough, and a sampled subset where it is not. A per-image patch at 300 steps is
in between, so the run reports the precision it achieved and the n that would
be needed for a given interval — see population.images_needed(). Quote the
interval, not a bare mean, and record --sample_seed so the subset is
reproducible.

DIAGNOSTICS ARE NOT RUN PER IMAGE
---------------------------------
The suite costs a Grad-CAM, an ERF probe and a dozen figures. Over hundreds of
images that dominates the run and produces figures nobody opens. Only
--n_panels images get it, chosen by --panel_select, and the default is
`spread` (best + median + worst) rather than `best`: the population summary
reports a distribution, and a panel figure showing only the strongest attacks
contradicts it in exactly the way a reviewer will notice.

RESUMABLE. Every image's record and the pooled confusion matrices are
checkpointed as the run proceeds, so a job killed at the walltime continues
from where it stopped with --resume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (add_model_args, add_patch_args, setup_model, build_patch,
                     image_indices)
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import report
from patchreach.metrics import population as pop_mod
from patchreach.patch import optimise, segmentation_cam
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, seed_everything, increment_path


def build_parser():
    p = add_patch_args(add_model_args(argparse.ArgumentParser(
        description="Optimise one patch per image over a population of "
                    "images and report the distribution")))
    p.add_argument("--loss_fn", default="cospgd",
                   choices=["ce", "cospgd", "ipatch_cospgd"])
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
    # Same default as overfit.py, deliberately. optimise.py's header states the
    # rule: anything that changes the attack changes BOTH callers or neither,
    # or the population numbers stop matching the single-image numbers and
    # nothing says why.
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

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    idxs = resolve_images(a, len(ds))

    if a.out_dir:
        out_dir = Path(a.out_dir)
    else:
        tag = "_".join(x for x in [a.arch, a.patch_mode, a.loss_fn,
                                   f"n{len(idxs)}", a.tag] if x)
        out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = out_dir / "patches"
    patch_dir.mkdir(exist_ok=True)
    state_path = out_dir / "population_state.pt"

    target = a.target_class if a.loss_fn == "ipatch_cospgd" else None
    pop = pop_mod.Population(a.num_classes, device=device, target_class=target)

    if a.resume and state_path.exists():
        pop.load_state_dict(torch.load(state_path, map_location="cpu"))
        print(f"[resume] {len(pop.records)} images already done in {out_dir}")
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

    # ONE CAM for the whole run. It registers a forward hook on the backbone,
    # so rebuilding it per image would stack hundreds of live hooks on a frozen
    # model and silently slow every forward pass in the run.
    cam = None
    if a.placement == "gradcam":
        cam = segmentation_cam.build(
            model, a.loss_fn if a.cam_objective == "attack" else a.cam_objective,
            a.target_class, a.cam_layer, a.cam_module, a.cam_target)
        print(f"[cam ] objective  : {a.cam_objective}  target: {a.cam_target}  "
              f"layer {a.cam_layer} of {a.cam_module}")
        print(f"[cam ] margin     : {a.placement_margin}px keep-out from the "
              f"image border")

    t0 = time.time()
    for n, i in enumerate(todo, 1):
        img, label = ds[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        # A FRESH patch per image. Carrying one over would make image k's
        # result depend on image k-1 and turn this into a sequential universal
        # patch by accident — a different threat model, silently.
        patch = build_patch(a, device, mean_t, std_t)
        clean_logits = optimise.prepare(model, img, patch, a.img_h, a.img_w,
                                        a.from_image, mean_t, std_t,
                                        cam=cam, label=label)

        rec = optimise.attack_image(
            model, img, label, patch,
            loss_fn=a.loss_fn, target_class=a.target_class, steps=a.steps,
            lr=a.lr, num_classes=a.num_classes,
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

        # Checkpoint after EVERY image: the pooled confusion matrices are not
        # recoverable from the per-image records, so losing them to a walltime
        # kill would cost the whole run rather than the current image.
        torch.save(pop.state_dict(), state_path)
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
            # A fully-suppressed map carries no localisation information, so
            # those images were CENTRED. Saying so matters: silently counting
            # them as gradcam-placed would dilute the very effect being measured.
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
        # Patch.save() stores the parameter, the config and the placement — NOT
        # the reference. For --from_image the base is the image region, so it
        # has to be re-derived here or render() silently falls back to 0.5 grey
        # and the panels show a patch that was never optimised.
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
        # The ERF probe is a MODEL property, not an image one, so it runs once
        # rather than once per panel — report.run's own flag for exactly this.
        report.run(model, img, label, patch, d, a.loss_fn, a.num_classes,
                   target, mean_t, std_t, skip_geometric=(rank > 0),
                   log=lambda *_, **__: None)
        print(f"    image {i}: drop_remote {r['drop_remote']:+.2f} -> {d}/")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"config": {**vars(a), "images": idxs},
                   "summary": summary,
                   "panels": [r["image"] for r in panels],
                   "records": [{k: v for k, v in r.items() if k != "history"}
                               for r in pop.records]}, f, indent=2)
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()
