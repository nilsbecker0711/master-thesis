#!/usr/bin/env python
"""Evaluate a trained patch on a fixed image set."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from _common import (add_model_args, setup_model, image_indices, FIXED10)
from _common import make_dataset
from patchreach.data.cityscapes import (CityscapesSeg, class_name,
                                        norm_tensors, upsample_to)
from patchreach.diagnostics import report
from patchreach.metrics.miou import (SegMetric, compare,
                                     single_image_miou, attack_rates)
from patchreach.metrics.curves import untargeted_reach, collapse_point
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, seed_everything, increment_path


def main():
    p = add_model_args(argparse.ArgumentParser())
    p.add_argument("--checkpoint", default=None,
                   help="a patchreach checkpoint (Patch.save). Optional ONLY "
                        "with --patch_png, which can stand alone for a square "
                        "raw patch pasted at the centre.")
    p.add_argument("--images", default="fixed10",
                   help="'fixed10' | 'all' | '2 5 45'")
    p.add_argument("--loss_fn", default=None,
                   help="defaults to the value stored in the checkpoint config")
    p.add_argument("--target_class", type=int, default=None)
    p.add_argument("--no_panels", action="store_true",
                   help="skip the per-image panel figures (they are cheap "
                        "— clean and adv logits are already in hand).")
    p.add_argument("--diagnostics_on", type=int, default=0,
                   help="run the full diagnostic suite on this image index "
                        "(-1 to skip)")
    p.add_argument("--from_image", action="store_true",
                   help="rebuild the base from the image being evaluated, as "
                        "overfit.py --from_image did during training. REQUIRED "
                        "for mode='csf': the checkpoint stores the parameter "
                        "and the placement, never the base, so without this "
                        "the base falls back to flat grey and the patch is a "
                        "visible square with an invisible texture.")

    p.add_argument("--patch_png", default=None,
                   help="evaluate THE PIXELS OF THIS PNG instead of the "
                        "checkpoint's parameter; mode='raw' only. The "
                        "checkpoint is still required and still supplies "
                        "size, scale, scale_ref, shape and the resolved "
                        "placement -- only the parameter is replaced. The "
                        "numbers then carry 8-bit quantisation, and the "
                        "deviation from the checkpoint's own render is "
                        "printed so it can be read rather than assumed.")

    p.add_argument("--patch_scale", type=float, default=None,
                   help="fraction of image HEIGHT the patch is pasted at, "
                        "for --patch_png WITHOUT a checkpoint. Default: the "
                        "PNG's own pixel size at this --img_h, so nothing is "
                        "resampled. Ignored when a checkpoint supplies the "
                        "geometry.")

    p.add_argument("--out_root", default="results/eval")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    if not a.checkpoint and not a.patch_png:
        p.error("need --checkpoint, or --patch_png on its own for a square "
                "raw patch pasted at the centre")

    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    if a.checkpoint is None:
        patch = Patch.from_png(a.patch_png, device, mean_t, std_t,
                               a.img_h, a.patch_scale)
        pcfg = asdict(patch.cfg)
    else:
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

    if a.checkpoint and pcfg.get("mode") == "csf" and not a.from_image:
        raise SystemExit(
            "\nThis checkpoint is mode='csf'. Its base is the image region the\n"
            "patch covers, and that base is NOT stored in the checkpoint — only\n"
            "the parameter and the placement are. Re-run with --from_image so\n"
            "the base is rebuilt from each evaluated image, exactly as\n"
            "overfit.py --from_image did during training.\n\n"
            "Without it the base is flat grey, the patch becomes a visible\n"
            "square with an invisible texture, and drop_remote describes that\n"
            "square rather than the residual under test.\n")

    png_delta = (patch.param_from_png(a.patch_png)
                 if a.patch_png and a.checkpoint else None)

    loss_fn = a.loss_fn or "cospgd"
    tgt = a.target_class

    print(f"\n[patch] loaded {a.checkpoint or a.patch_png}")
    if png_delta is not None:
        print(f"[patch] parameter REPLACED by the pixels of {a.patch_png}")
        print(f"[patch] max |png - checkpoint render| {png_delta:.6f}  "
              f"(one 8-bit step is 1/255 = {1 / 255:.6f})")
    if a.checkpoint is None:
        own = " (the PNG's own pixel size)" if a.patch_scale is None else ""
        print(f"[patch] FROM THE PNG ALONE: square raw patch at the centre, "
              f"scale {patch.cfg.scale:.6g} of height" + own)
        if patch.side(a.img_h, a.img_w) != patch.cfg.size:
            print(f"[patch] WARNING   : the PNG is {patch.cfg.size}px but "
                  f"pastes at {patch.side(a.img_h, a.img_w)}px, so it is "
                  f"RESAMPLED. Pass --patch_scale "
                  f"{patch.cfg.size / a.img_h:g} to paste it 1:1.")
    patch.describe(a.img_h, a.img_w)

    ds = make_dataset(a, "val")
    idxs = image_indices(a.images, len(ds))

    tag = "_".join(x for x in [Path(a.checkpoint or a.patch_png).parent.name,
                               f"eval{a.img_h}x{a.img_w}",
                               "png" if a.patch_png else "", a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    m = {k: SegMetric(a.num_classes, device=device)
         for k in ("clean_all", "clean_rem", "adv_all", "adv_rem")}
    per_image = []

    for i in idxs:
        img, label = ds[i]
        img, label = img.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
        hw = label.shape[-2:]
        with torch.no_grad():
            lc = upsample_to(model(img), hw)
            patch.resolve_placement(a.img_h, a.img_w, lc.argmax(1)[0])
            if a.from_image:
                patch.set_reference_from_image(img, mean_t, std_t)
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

        if not a.no_panels:
            d = out_dir / "panels" / f"img{i:04d}"
            report.save_panels(img, label, patched, lc, la, fp, patch,
                               mean_t, std_t, d)
            row["per_class_iou"] = report.per_class_iou_figure(
                lc, la, label, a.num_classes, d / "per_class_iou.png", tgt,
                title=f"val image {i}")

        if i == a.diagnostics_on:
            report.run(model, img, label, patch, out_dir / f"diag_img{i}",
                       loss_fn, a.num_classes, tgt, mean_t, std_t)

    agg = {k: v.compute() for k, v in m.items()}
    agg.update(compare(m["clean_all"], m["adv_all"], prefix="all_"))
    agg.update(compare(m["clean_rem"], m["adv_rem"], prefix="rem_"))
    ciou, aiou = m["clean_rem"].per_class(), m["adv_rem"].per_class()
    agg["drop_remote_dataset"] = agg["rem_drop"]
    drops = torch.tensor([r["drop_remote"] for r in per_image])
    flips = torch.tensor([r["any_flip_rate"] for r in per_image])

    print(f"\n{'='*66}")
    print(f"  {len(idxs)} images @ {a.img_h}x{a.img_w}")
    print(f"  drop_remote  per-image mean {drops.mean():+.2f} "
          f"+/- {drops.std():.2f}  (range {drops.min():+.2f} to {drops.max():+.2f})")
    print(f"  any_flip     mean {flips.mean():.1f}% +/- {flips.std():.1f}%")
    print(f"  DATASET drop_remote {agg['drop_remote_dataset']:+.2f}")
    print(f"\n  per-class IoU (remote), clean -> patched:")
    for c in range(min(a.num_classes, 19)):
        if not torch.isnan(ciou[c]):
            print(f"    {c:2d} {class_name(c):10s}: {ciou[c]:6.2f} -> "
                  f"{aiou[c]:6.2f}  ({aiou[c]-ciou[c]:+.1f})")
    print(f"{'='*66}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"checkpoint": a.checkpoint,
                   "param_source": a.patch_png or a.checkpoint,
                   "config": vars(a),
                   "patch_config": pcfg, "aggregate": agg,
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
