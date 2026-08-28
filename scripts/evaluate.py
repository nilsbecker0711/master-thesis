#!/usr/bin/env python
r"""
Evaluate a trained patch on a set of images. NO TRAINING.

    python scripts/evaluate.py --checkpoint results/runs/<id>/best.pt \
        --arch segformer_b0 --cityscapes_root $CS --img_h 512 --img_w 1024

RESOLUTION TRANSFER: pass an --img_h/--img_w different from training. Nothing
in the patch is tied to resolution — the checkpoint stores the parameter, and
the renderer resizes it — so the full transfer matrix is evaluation-only.
The OFF-DIAGONAL is the result: the diagonal says patches work, the
off-diagonal says whether they TRANSFER, and those are different claims.

Evaluation is cheap and training is not, so evaluate every configuration on the
same fixed images rather than training more.

CROSS-IMAGE VALIDATION
----------------------
The same off-diagonal argument holds across IMAGES, and it is the one a
single-image overfit needs: a patch optimised on image 420 reaching 50 points
there says nothing about whether it reaches anything anywhere else. That is one
run of this script, not n runs compared by hand:

    a random image          --images random --n_images 1 --sample_seed 7
    n random images         --images random --n_images 25 --sample_seed 0
    n fixed images          --images fixed  --n_images 25
    everything but image x  --images all    --exclude_image 420
    the whole val set       --images all    --exclude_image -1

--exclude_image is what keeps the number honest: the training image left in the
mean makes it a mixture of a training number and a transfer number. It is
applied BEFORE --n_images, so a sample of n stays a sample of n.

WHAT TRANSFERS FOR mode='csf' IS THE RESIDUAL, NOT THE PATCH. A csf patch
trained with --from_image is base + CSF-bounded residual, and the base was the
region of the training image it covered. The checkpoint stores the parameter,
not the base, so applying it to a new image means re-deriving the base THERE
and adding the same residual — which is also the only version of the attack
that stays invisible on the new image. That happens automatically for
checkpoints that recorded --from_image; older ones need it passed here, and the
run says so rather than quietly rendering a grey square.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from _common import (add_model_args, add_image_args, add_cam_args,
                     setup_model, resolve_images, sample_tag, tsallis_kwargs)
from patchreach.data.cityscapes import (CityscapesSeg, class_name,
                                        norm_tensors, upsample_to)
from patchreach.diagnostics import report
from patchreach.metrics.miou import (SegMetric, compare,
                                     single_image_miou, attack_rates)
from patchreach.metrics.curves import untargeted_reach, collapse_point
from patchreach.metrics import population as pop_mod
from patchreach.patch import optimise, segmentation_cam
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, seed_everything, increment_path


# overfit.py names its run …_img420; overfit_population.py writes
# patches/img0420/best.pt. Both are the image the patch was OVERFIT on, and
# that is the one image a cross-image number must not include.
_IMG_IN_PATH = re.compile(r"(?:^|[_\-])img0*(\d+)(?:$|[_\-.])")


def _checkpoint_image(path) -> int | None:
    """The val index a checkpoint's PATH names, or None. Advisory only."""
    for part in reversed(Path(path).parts):
        m = _IMG_IN_PATH.search(part)
        if m:
            return int(m.group(1))
    return None


def main():
    p = add_image_args(add_model_args(argparse.ArgumentParser()),
                       default_images="fixed10",
                       n_help="Cross-image validation of a single-image patch "
                              "is --images random/fixed with an n, or --images "
                              "all with --exclude_image on the image it was "
                              "trained on.")
    p.add_argument("--checkpoint", required=True)
    # Tri-state on purpose. The DEFAULT (None) reads the checkpoint, which is
    # the only source that knows the truth; the two flags exist for
    # checkpoints written before the field did.
    p.add_argument("--from_image", dest="from_image", action="store_true",
                   default=None,
                   help="take the csf base from the region of THE IMAGE BEING "
                        "EVALUATED, and add the trained residual on top. This "
                        "is what a cross-image test of a csf patch means. "
                        "Default: whatever the checkpoint recorded.")
    p.add_argument("--no_from_image", dest="from_image", action="store_false",
                   help="force the base to the checkpoint's own (0.5 grey for "
                        "a csf patch with no --reference). Turns the patch "
                        "into a visible square; it is a control, not a result.")
    p.add_argument("--loss_fn", default=None,
                   help="defaults to the value stored in the checkpoint config")
    p.add_argument("--target_class", type=int, default=None)
    p.add_argument("--no_panels", action="store_true",
                   help="skip the per-image panel figures entirely. "
                        "Equivalent to --n_panels 0.")
    p.add_argument("--n_panels", type=int, default=10,
                   help="cap on how many images get panel figures. A panel is "
                        "cheap — the clean and adv logits are already in hand "
                        "— but 500 of them are not, and --images all is now a "
                        "normal thing to ask for. They go to the FIRST "
                        "n_panels images in evaluation order, which is "
                        "arbitrary with respect to attack strength and "
                        "therefore unbiased; this script does not pick the "
                        "winners. 0 disables.")
    p.add_argument("--diagnostics_on", type=int, default=0,
                   help="run the full diagnostic suite on this image index "
                        "(-1 to skip)")

    # Needed to REBUILD the sensitivity map for a gradcam-placed checkpoint.
    # Those settings describe how the map is computed, not the patch, so they
    # are not in the checkpoint's PatchConfig and have to be restated here.
    # Defaults match the training ones.
    add_cam_args(p)

    p.add_argument("--out_root", default="results/eval")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    ck = torch.load(a.checkpoint, map_location="cpu")
    pcfg = ck["config"]
    G = None
    if pcfg.get("mode") in ("gan", "raw_ganinit"):
        from GANLatentDiscovery.loading import load_from_dir
        from GANLatentDiscovery.utils import is_conditional
        _, G, _ = load_from_dir(
            "./GANLatentDiscovery/models/pretrained/deformators/BigGAN/",
            G_weights="./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth")
        if is_conditional(G):
            G.set_classes(259)
        G.eval().to(device)
    patch = Patch.load(a.checkpoint, device, mean_t, std_t, generator=G)
    loss_fn = a.loss_fn or "cospgd"
    tgt = a.target_class

    print(f"\n[patch] loaded {a.checkpoint}")
    patch.describe(a.img_h, a.img_w)

    # ── where the base comes from on EVERY image this run touches ────────────
    # Only mode='csf' has a base; every other mode renders from the parameter
    # alone and ignores this entirely.
    from_image = (bool(pcfg.get("from_image", False)) if a.from_image is None
                  else a.from_image)
    if pcfg.get("mode") == "csf":
        if from_image:
            print("[patch] base      : re-derived from EACH evaluated image "
                  "(the residual is what transfers)")
        else:
            print("  [!] this csf patch is being rendered on the base stored "
                  "in its config —")
            print(f"      {pcfg.get('reference') or '0.5 grey'}. If it was "
                  f"trained with --from_image, that base is")
            print("      gone from the checkpoint and this run measures a "
                  "patch that was never")
            print("      optimised. Pass --from_image (checkpoints written "
                  "since this flag record it).")

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    idxs = resolve_images(a.images, len(ds), a.n_images, a.sample_seed,
                          a.exclude_image)
    held_out = sorted(x for x in a.exclude_image if x >= 0)
    print(f"\n[data] {len(idxs)} images — --images {a.images}"
          + (f" --n_images {a.n_images}" if a.n_images else "")
          + (f" --sample_seed {a.sample_seed}" if a.images == "random" else "")
          + (f", holding out {held_out}" if held_out else ""))
    print(f"[data] {idxs if len(idxs) <= 20 else str(idxs[:20])[:-1] + ', ...]'}")
    # A patch is named after the image it was overfit on, in overfit.py's run
    # tag (…_img420) and in overfit_population.py's per-image directory
    # (patches/img0420/best.pt). If that image is still in the set, the mean
    # this run prints is part training number — say so, once, in words.
    train_img = _checkpoint_image(a.checkpoint)
    if train_img is not None and train_img in idxs:
        print(f"  [!] the checkpoint path names image {train_img} and it IS "
              f"in this set, so the")
        print(f"      reported mean mixes the image the patch was trained on "
              f"with the ones it")
        print(f"      transfers to. --exclude_image {train_img} for a "
              f"cross-image number.")

    tag = "_".join(x for x in [Path(a.checkpoint).parent.name,
                               f"eval{a.img_h}x{a.img_w}", sample_tag(a),
                               a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONE CAM for the whole run — it registers a forward hook, so rebuilding it
    # per image would stack one live hook per image on a frozen model. Only
    # built for a gradcam-placed patch, where placement is a FUNCTION OF THE
    # IMAGE: on a new image the patch goes where that image is sensitive, which
    # is the cross-image analogue of re-deriving the csf base.
    cam = None
    if patch.cfg.placement == "gradcam":
        cam = segmentation_cam.build(
            model, loss_fn if a.cam_objective == "attack" else a.cam_objective,
            tgt if tgt is not None else 8, a.cam_layer, a.cam_module,
            a.cam_target, tsallis=tsallis_kwargs(pcfg, 1))
        print(f"[cam ] objective  : {a.cam_objective}  target: {a.cam_target}  "
              f"layer {a.cam_layer} of {a.cam_module}")

    m = {k: SegMetric(a.num_classes, device=device)
         for k in ("clean_all", "clean_rem", "adv_all", "adv_rem")}
    per_image = []
    n_panels = 0 if a.no_panels else max(0, a.n_panels)
    if 0 < n_panels < len(idxs):
        print(f"[figs] panels for the first {n_panels} of {len(idxs)} images "
              f"(--n_panels); every image is in the metrics")

    for i in idxs:
        img, label = ds[i]
        img, label = img.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
        hw = label.shape[-2:]
        # THE SAME FUNCTION THE TRAINING LOOP USES, for the same reason
        # optimise.py exists: clean forward -> sensitivity map ->
        # resolve_placement -> image-derived base is an ordering, and a second
        # copy of it here would drift. Everything in it is per image, which is
        # exactly what makes this a cross-image test — the patch is placed and
        # based on the image it is being applied to, and only the optimised
        # parameter is carried over.
        # .detach() because the gradcam path builds its map with a BACKWARD
        # pass, so the clean logits come back attached to a graph. Nothing here
        # differentiates them, and the figure helpers call .numpy() on them.
        lc = optimise.prepare(model, img, patch, a.img_h, a.img_w,
                              from_image, mean_t, std_t, cam=cam,
                              label=label).detach()
        with torch.no_grad():
            patched, fp = patch.apply(img)
            la = upsample_to(model(patched), hw)

        pc, pa = lc.argmax(1), la.argmax(1)
        m["clean_all"].update(pc, label)
        m["clean_rem"].update(pc, label, exclude=fp)
        m["adv_all"].update(pa, label)
        m["adv_rem"].update(pa, label, exclude=fp)

        cr = single_image_miou(lc, label, a.num_classes, exclude=fp)
        ar = single_image_miou(la, label, a.num_classes, exclude=fp)
        rates = attack_rates(lc, la, label, fp, tgt)
        d, r = untargeted_reach(lc, la, fp)
        row = {"image": i, "clean_remote": cr, "adv_remote": ar,
               "drop_remote": cr - ar, **rates,
               "collapse_px": collapse_point(d, r)}
        per_image.append(row)
        print(f"  img {i:4d}: drop_remote {cr-ar:+6.2f}  "
              f"any_flip {rates['any_flip_rate']:5.1f}%")

        # Labelled figures, for more than just the one image that gets the
        # full diagnostic suite — but capped, see --n_panels.
        if n_panels > 0 and len(per_image) <= n_panels:
            d = out_dir / "panels" / f"img{i:04d}"
            report.save_panels(img, label, patched, lc, la, fp, patch,
                               mean_t, std_t, d)
            row["per_class_iou"] = report.per_class_iou_figure(
                lc, la, label, a.num_classes, d / "per_class_iou.png", tgt,
                title=f"val image {i}")

        # Full suite (ERF probe, confusion, margins) on one image only — the
        # probe costs n_probes extra forward passes.
        if i == a.diagnostics_on:
            report.run(model, img, label, patch, out_dir / f"diag_img{i}",
                       loss_fn, a.num_classes, tgt, mean_t, std_t)

    if cam is not None:
        if getattr(cam, "n_degenerate", 0):
            # A fully-suppressed map carries no localisation information, so
            # those images were CENTRED. Counting them as gradcam-placed would
            # dilute the very effect being measured.
            print(f"\n  NOTE: the sensitivity map was degenerate on "
                  f"{cam.n_degenerate} images; those fell back to")
            print(f"        CENTRE placement. They are still in the numbers "
                  f"below.")
        cam.close()

    agg = {k: v.compute() for k, v in m.items()}
    agg.update(compare(m["clean_all"], m["adv_all"], prefix="all_"))
    agg.update(compare(m["clean_rem"], m["adv_rem"], prefix="rem_"))
    ciou, aiou = m["clean_rem"].per_class(), m["adv_rem"].per_class()
    agg["drop_remote_dataset"] = agg["rem_drop"]
    drops = torch.tensor([r["drop_remote"] for r in per_image])
    flips = torch.tensor([r["any_flip_rate"] for r in per_image])

    # The distribution, not a bare mean — the same summary and the same
    # bootstrap overfit_population.py reports, so a transfer number and a
    # per-image-optimised number are quoted in the same units with the same
    # interval and can be put side by side.
    dist = pop_mod.describe([r["drop_remote"] for r in per_image])
    pop_mod.plot_distribution(
        per_image, out_dir / "distribution.png", key="drop_remote",
        title=f"{a.arch} — {pcfg.get('mode')} patch from "
              f"{Path(a.checkpoint).parent.name}",
        subtitle=f"ONE patch applied to {len(per_image)} images "
                 f"(--images {a.images})")

    print(f"\n{'='*66}")
    print(f"  {len(idxs)} images @ {a.img_h}x{a.img_w}"
          + (f", excluding {held_out}" if held_out else ""))
    print(f"  drop_remote  per-image mean {drops.mean():+.2f} "
          f"+/- {drops.std():.2f}  (range {drops.min():+.2f} to {drops.max():+.2f})")
    if dist["n"] > 1:
        print(f"               95% CI (bootstrap) "
              f"[{dist['ci95_boot'][0]:+.2f}, {dist['ci95_boot'][1]:+.2f}]"
              f"   median {dist['median']:+.2f}")
        print(f"               worked on {dist['success_rate']['1.0']:.0f}% "
              f"of images at >1 point, "
              f"{dist['success_rate']['5.0']:.0f}% at >5")
        need = pop_mod.images_needed(dist["std"], 2.0)
        if need > dist["n"]:
            print(f"  [!] +/-2 points needs n >= {need} at this spread; "
                  f"this run has {dist['n']}.")
    print(f"  any_flip     mean {flips.mean():.1f}% +/- {flips.std():.1f}%")
    print(f"  DATASET drop_remote {agg['drop_remote_dataset']:+.2f}")
    print(f"\n  per-class IoU (remote), clean -> patched:")
    for c in range(min(a.num_classes, 19)):
        if not torch.isnan(ciou[c]):
            print(f"    {c:2d} {class_name(c):10s}: {ciou[c]:6.2f} -> "
                  f"{aiou[c]:6.2f}  ({aiou[c]-ciou[c]:+.1f})")
    print(f"{'='*66}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"checkpoint": a.checkpoint, "config": vars(a),
                   # The resolved set, not just the flags that produced it: a
                   # seeded sample is only reproducible if the seed AND the
                   # indices it expanded to are both on record.
                   "images": idxs, "excluded_images": held_out,
                   "base_from_image": from_image,
                   "train_image_in_path": train_img,
                   "patch_config": pcfg, "aggregate": agg,
                   "drop_remote_distribution": dist,
                   "drop_remote_mean": float(drops.mean()),
                   "drop_remote_std": float(drops.std()),
                   "any_flip_mean": float(flips.mean()),
                   "per_class_iou": {
                       class_name(c): {"clean": float(ciou[c]),
                                       "adv": float(aiou[c])}
                       for c in range(min(a.num_classes, 19))
                       if not torch.isnan(ciou[c])},
                   "per_image": per_image}, f, indent=2)
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()