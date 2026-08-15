#!/usr/bin/env python
r"""
Single-image overfit — the sanity check before any multi-hour run.

    python scripts/overfit.py --arch segformer_b0 --cityscapes_root $CS \
        --patch_mode raw --loss_fn cospgd --image 2 --steps 300

If mIoU on ONE image drops sharply, the pipeline is correct — scale up. If it
does not, there is a bug, and finding it here costs two minutes instead of five
hours. This isolates "is the attack possible at all?" from "does it generalise?".

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
from patchreach.losses import adversarial
from patchreach.metrics.miou import single_image_miou, attack_rates
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
    with torch.no_grad():
        clean_logits = upsample_to(model(img), label.shape[-2:])
    patch.resolve_placement(a.img_h, a.img_w, clean_logits.argmax(1)[0])
    if a.from_image:
        # Base the patch on the region it will cover. Only meaningful AFTER
        # resolve_placement, and only changes behaviour for modes that treat
        # the reference as a base (csf); every pre-existing mode ignores it.
        patch.set_reference_from_image(img, mean_t, std_t)
        print(f"[patch] base      : image region at {patch.placement} "
              f"(--from_image)")
    patch.describe(a.img_h, a.img_w)

    _, fp0 = patch.apply(img)
    clean_all = single_image_miou(clean_logits, label, a.num_classes)
    clean_rem = single_image_miou(clean_logits, label, a.num_classes, exclude=fp0)
    print(f"\n[clean] all {clean_all:.2f} | remote {clean_rem:.2f}  "
          f"<- remote is the baseline that matters")

    tag = "_".join(x for x in [a.arch, a.patch_mode, a.loss_fn,
                               f"img{a.image}", a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = adversarial.build(
        a.loss_fn, a.target_class if a.loss_fn == "ipatch_cospgd" else 8)
    opt = torch.optim.Adam([patch.param], lr=a.lr, betas=(0.9, 0.999),
                           amsgrad=True)

    # Track the BEST patch, not just the final one. A saturation collapse
    # late in a run otherwise destroys the result it already achieved.
    history, best_drop = [], -1e9
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        patched, fp = patch.apply(img)
        logits = upsample_to(model(patched), label.shape[-2:])
        sup = (~fp) if a.exclude_footprint else None
        la = loss_fn(logits, label, fp, sup)
        if not torch.isfinite(la):
            raise RuntimeError(f"non-finite loss at step {step}")
        extra = patch.regularisers()
        (la + extra["total"]).backward()

        if step == 1:
            g = patch.param.grad
            print(f"[grad] {'None — GRAPH BROKEN' if g is None else f'abs mean {g.abs().mean():.3e}'}")
            if patch.cfg.mode == "lap":
                magnitude_report(la.item(), patch.render(), patch.reference,
                                 patch.active_mask(), patch.cfg)

        opt.step()
        patch.project()

        if step % a.log_every == 0 or step == 1:
            with torch.no_grad():
                cur = single_image_miou(logits.detach(), label, a.num_classes,
                                        exclude=fp)
            st = " ".join(f"{k}={v:.4f}" for k, v in patch.stats().items())
            print(f"  step {step:4d}  remote={cur:6.2f} "
                  f"({clean_rem-cur:+6.2f})  {a.loss_fn}={la.item():.4f}  {st}")
            history.append({"step": step, "miou_remote": cur,
                            "loss": la.item(), **patch.stats()})
            if clean_rem - cur > best_drop:
                best_drop = clean_rem - cur
                patch.save(out_dir / "best.pt")
                save_image(patch.render().cpu(), out_dir / "best_patch.png")
            save_image(patch.render().cpu(),
                       out_dir / f"patch_step{step:04d}.png")

    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), label.shape[-2:])
        final_all = single_image_miou(adv_logits, label, a.num_classes)
        final_rem = single_image_miou(adv_logits, label, a.num_classes,
                                      exclude=fp)
        rates = attack_rates(clean_logits, adv_logits, label, fp,
                             a.target_class if a.loss_fn == "ipatch_cospgd"
                             else None)

    # ── calibration for stage 2 ──────────────────────────────────────────────
    # L_rat at step 1 is ~0 by construction: a stage-1 run starts AT the
    # reference, so ||q - c|| = 0 and any weight derived from it is a division
    # by zero. The meaningful scale is the DRIFT the attack produced — measured
    # here, at the end. Set alpha from this row, not from the step-1 one.
    if patch.cfg.mode == "lap" and patch.reference is not None:
        print("\n[lap] END-OF-RUN magnitudes — set stage-2 alpha from THIS "
              "rat row,\n      not from the step-1 report (where L_rat is 0 "
              "by construction):")
        magnitude_report(la.item(), patch.render(), patch.reference,
                         patch.active_mask(), patch.cfg)

    if best_drop > (clean_rem - final_rem) + 1.0:
        print(f"\n  WARNING: best drop was {best_drop:+.2f} but the FINAL "
              f"patch only reaches {clean_rem-final_rem:+.2f}.\n"
              f"  The run degraded after its peak — check frac_at_clip in the "
              f"history for\n  sigmoid saturation, and use best.pt rather "
              f"than final.pt.")

    patch.save(out_dir / "final.pt")
    save_image(patch.render().cpu(), out_dir / "final_patch.png")

    print(f"\n{'='*66}")
    print(f"  clean  all/remote : {clean_all:.2f} / {clean_rem:.2f}")
    print(f"  final  all/remote : {final_all:.2f} / {final_rem:.2f}")
    print(f"  drop   REMOTE     : {clean_rem-final_rem:+.2f}   <- the result")
    for k, v in rates.items():
        print(f"  {k:18s}: {v:.1f}%")
    print(f"{'='*66}")

    res = {"config": vars(a), "clean_all": clean_all, "clean_remote": clean_rem,
           "final_all": final_all, "final_remote": final_rem,
           "drop_all": clean_all - final_all,
           "drop_remote": clean_rem - final_rem,
           "best_drop_remote": best_drop, **rates, "history": history}
    if patch.cfg.mode == "lap":
        from patchreach.patch.lap import rationality_report
        res["rationality"] = rationality_report(patch.render(), patch.reference)
    with open(out_dir / "results.json", "w") as f:
        json.dump(res, f, indent=2)

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