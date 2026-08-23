#!/usr/bin/env python
r"""
Single-image overfit — the sanity check before any multi-hour run.

    python scripts/overfit.py --arch segformer_b0 --cityscapes_root $CS \
        --patch_mode raw --loss_fn cospgd --image 2 --steps 300

If mIoU on ONE image drops sharply, the pipeline is correct — scale up. If it
does not, there is a bug, and finding it here costs two minutes instead of five
hours. This isolates "is the attack possible at all?" from "does it generalise?".

ONE IMAGE IS AN ANECDOTE. For a result rather than a check, run the same attack
over a population and report the distribution:

    python scripts/overfit_population.py --images random --n_images 100 ...

Both scripts call patchreach.patch.optimise.attack_image(), so the procedure is
identical by construction and a change to one cannot silently fail to reach the
other.

NOTE ON COMPARABILITY: a single-image overfit is a much EASIER problem than a
universal patch trained across the dataset. Yuan et al.'s numbers come from
per-image patches at 400 iterations; ours are universal. Do not compare a
result from this script against a published universal-patch number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from _common import (add_model_args, add_patch_args, setup_model, build_patch)
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import report
from patchreach.patch import optimise, segmentation_cam
from patchreach.patch.lap import magnitude_report
from patchreach.utils import get_device, seed_everything, increment_path


def main():
    p = add_patch_args(add_model_args(argparse.ArgumentParser()))
    p.add_argument("--loss_fn", default="cospgd",
                   choices=["ce", "cospgd", "ipatch_cospgd"])
    p.add_argument("--from_image", action="store_true",
                    help="Initialise the patch with the image region it will replace.")
    p.add_argument("--target_class", type=int, default=8)
    p.add_argument("--image", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--exclude_footprint", action="store_true", default=True)
    p.add_argument("--no_diagnostics", action="store_true")
    p.add_argument("--out_root", default="results/overfit")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    img, label = ds[a.image]
    img, label = img.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
    print(f"[data] image {a.image}, classes present "
          f"{sorted(label.unique().tolist())}")

    G = None
    if a.patch_mode in ("gan", "raw_ganinit"):
        from GANLatentDiscovery.loading import load_from_dir
        from GANLatentDiscovery.utils import is_conditional
        _, G, _ = load_from_dir(
            "./GANLatentDiscovery/models/pretrained/deformators/BigGAN/",
            G_weights="./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth")
        if is_conditional(G):
            G.set_classes(259)
        G.eval().to(device)
        for q in G.parameters():
            q.requires_grad_(False)

    patch = build_patch(a, device, mean_t, std_t, generator=G)

    # ORDERING: clean forward -> resolve_placement -> first apply().
    # Semantic placement reads the CLEAN PREDICTION, so it cannot be resolved
    # before the model has seen an image. Backwards, it silently uses centre.
    # --placement gradcam needs the sensitivity map. Built here rather than
    # inside prepare() so the hook is registered once and closed once; the CAM
    # holds a forward hook on the backbone and leaking one per image would
    # accumulate hooks across a population run.
    cam = None
    if a.placement == "gradcam":
        cam = segmentation_cam.build(
            model, a.loss_fn if a.cam_objective == "attack" else a.cam_objective,
            a.target_class, a.cam_layer, a.cam_module, a.cam_target)
    clean_logits = optimise.prepare(model, img, patch, a.img_h, a.img_w,
                                    a.from_image, mean_t, std_t,
                                    cam=cam, label=label, log=print)
    if cam is not None:
        cam.close()
    patch.describe(a.img_h, a.img_w)

    tag = "_".join(x for x in [a.arch, a.patch_mode, a.loss_fn,
                               f"img{a.image}", a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    res = optimise.attack_image(
        model, img, label, patch,
        loss_fn=a.loss_fn, target_class=a.target_class, steps=a.steps,
        lr=a.lr, num_classes=a.num_classes,
        exclude_footprint=a.exclude_footprint, log_every=a.log_every,
        clean_logits=clean_logits, out_dir=out_dir,
        save_step_images=True, save_best=True, verbose=True)

    # ── calibration for stage 2 ──────────────────────────────────────────────
    # L_rat at step 1 is ~0 by construction: a stage-1 run starts AT the
    # reference, so ||q - c|| = 0 and any weight derived from it is a division
    # by zero. The meaningful scale is the DRIFT the attack produced — measured
    # here, at the end. Set alpha from this row, not from the step-1 one.
    if patch.cfg.mode == "lap" and patch.reference is not None:
        print("\n[lap] END-OF-RUN magnitudes — set stage-2 alpha from THIS "
              "rat row,\n      not from the step-1 report (where L_rat is 0 "
              "by construction):")
        magnitude_report(res["last_loss"], patch.render(), patch.reference,
                         patch.active_mask(), patch.cfg)

    patch.save(out_dir / "final.pt")
    save_image(patch.render().detach().cpu(), out_dir / "final_patch.png")

    print(f"\n{'='*66}")
    print(f"  clean  all/remote : {res['clean_all']:.2f} / "
          f"{res['clean_remote']:.2f}")
    print(f"  final  all/remote : {res['final_all']:.2f} / "
          f"{res['final_remote']:.2f}")
    print(f"  drop   REMOTE     : {res['drop_remote']:+.2f}   <- the result")
    for k in ("any_flip_rate", "target_hit_rate"):
        if k in res:
            print(f"  {k:18s}: {res[k]:.1f}%")
    print(f"{'='*66}")

    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), label.shape[-2:])

    out = {"config": vars(a), **res}
    if patch.cfg.mode == "lap":
        from patchreach.patch.lap import rationality_report
        out["rationality"] = rationality_report(patch.render(),
                                                patch.reference)
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    report.per_class_iou_figure(
        clean_logits, adv_logits, label, a.num_classes,
        out_dir / "per_class_iou.png",
        a.target_class if a.loss_fn == "ipatch_cospgd" else None,
        title=f"val image {a.image}")

    if not a.no_diagnostics:
        report.run(model, img, label, patch, out_dir / "diagnostics",
                   a.loss_fn, a.num_classes,
                   a.target_class if a.loss_fn == "ipatch_cospgd" else None,
                   mean_t, std_t)
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()
