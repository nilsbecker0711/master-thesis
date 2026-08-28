#!/usr/bin/env python
r"""
Evaluate a trained patch on a fixed image set. NO TRAINING.

    python scripts/evaluate.py --checkpoint results/runs/<id>/best.pt \
        --arch segformer_b0 --cityscapes_root $CS --img_h 512 --img_w 1024 \
        --from_image

--from_image IS REQUIRED FOR mode='csf' and the script refuses without it. The
checkpoint stores the parameter and the placement; the BASE is the image region
the patch covers and is rebuilt per image, so evaluating without it silently
substitutes flat grey for scene content.

RESOLUTION TRANSFER: pass an --img_h/--img_w different from training. Nothing
in the patch is tied to resolution — the checkpoint stores the parameter, and
the renderer resizes it — so the full transfer matrix is evaluation-only.
The OFF-DIAGONAL is the result: the diagonal says patches work, the
off-diagonal says whether they TRANSFER, and those are different claims.

Evaluation is cheap and training is not, so evaluate every configuration on the
same fixed images rather than training more.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import (add_model_args, setup_model, image_indices, FIXED10)
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
    p.add_argument("--checkpoint", required=True)
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

    # REFUSE RATHER THAN RENDER A GREY SQUARE. Patch.load restores `param` and
    # `placement`; `reference` is NOT in the checkpoint, and `from_image` is a
    # script flag rather than a PatchConfig field, so it cannot round-trip.
    # A csf patch rendered with reference=None takes the 0.5 grey fallback in
    # Patch.render() — the failure --patch_mode's own help text warns about —
    # and every number below would silently describe a different patch.
    if pcfg.get("mode") == "csf" and not a.from_image:
        raise SystemExit(
            "\nThis checkpoint is mode='csf'. Its base is the image region the\n"
            "patch covers, and that base is NOT stored in the checkpoint — only\n"
            "the parameter and the placement are. Re-run with --from_image so\n"
            "the base is rebuilt from each evaluated image, exactly as\n"
            "overfit.py --from_image did during training.\n\n"
            "Without it the base is flat grey, the patch becomes a visible\n"
            "square with an invisible texture, and drop_remote describes that\n"
            "square rather than the residual under test.\n")

    loss_fn = a.loss_fn or "cospgd"
    tgt = a.target_class

    print(f"\n[patch] loaded {a.checkpoint}")
    patch.describe(a.img_h, a.img_w)

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    idxs = image_indices(a.images, len(ds))

    tag = "_".join(x for x in [Path(a.checkpoint).parent.name,
                               f"eval{a.img_h}x{a.img_w}", a.tag] if x)
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
            # placement is resolved per image; semantic placement depends on
            # scene content, so a fixed offset would be wrong here
            patch.resolve_placement(a.img_h, a.img_w, lc.argmax(1)[0])
            # AFTER resolve_placement, never before: set_reference_from_image
            # copies the region at the RESOLVED placement, so the order here
            # matches optimise.prepare() and semantic/gradcam placement stays
            # consistent between training and evaluation.
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

        # Labelled figures for EVERY evaluated image, not only the one that
        # gets the full diagnostic suite.
        if not a.no_panels:
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
        json.dump({"checkpoint": a.checkpoint, "config": vars(a),
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