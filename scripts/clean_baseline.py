#!/usr/bin/env python
r"""
Block 0.1 — clean DATASET mIoU per architecture and resolution. No patch.

    python scripts/clean_baseline.py --arch segformer_b0 \
        --cityscapes_root $CS --img_h 512 --img_w 1024 --n_images 20

WHY THIS RUNS FIRST: you cannot compare two models until both are healthy. If
one is outside its trained regime, every attack number downstream conflates
"attack worked" with "model was already broken". Pick the resolution where BOTH
land near published values — that is your fair comparison point, and running
each at its own native scale would confound architecture with resolution.

DATASET vs PER-IMAGE mIoU: this reports DATASET mIoU, accumulated over one
confusion matrix across all images. That is what published numbers are. A
per-image mIoU averages over only the classes present in that image, so a rare
class covering a few hundred pixels scores near zero and drags the mean down —
50 per-image is normal for a model that scores 76 on the dataset. Do not
compare the two.

WHOLE vs SLIDE INFERENCE — the other half of reproducing a published number.
DeepLab and UNet declare test_cfg=dict(mode='whole'), so a single forward IS
their published procedure. SegFormer and SETR declare mode='slide', and their
published Cityscapes numbers come from overlapping crops; a whole forward reads
several points low for them. --inference auto honours whatever the config
declares, which is what you want when checking a checkpoint against the zoo.
The default stays 'whole' so every other script's behaviour is unchanged, and
the attack path never uses slide at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from _common import add_model_args, setup_model
from _common import make_dataset
from patchreach.data.cityscapes import CityscapesSeg, class_name, upsample_to
from patchreach.metrics.miou import SegMetric
from patchreach.utils import get_device, seed_everything

SLIDE_WARNING = """[eval] A SLIDE NUMBER IS NOT THE CLEAN REFERENCE FOR AN ATTACK RUN.
       Under slide each window is an independent forward, so a patch cannot
       influence pixels outside the windows containing it. At 1024x2048 with a
       centred 128px patch that caps reach at 448/704px L/R for setr_pup, while
       segformer and deeplab reach 960 — an ARCHITECTURE-DEPENDENT ceiling on
       the quantity aggregate.py bins out to 1200px. Every attack path uses
       whole-image forward. Compare this number to the published zoo value and
       to nothing else."""


def main():
    p = add_model_args(argparse.ArgumentParser())
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--out", default=f"results/clean_baselines")
    p.add_argument("--tag", type=str, default=None)
    a = p.parse_args()
    # The tag is folded in ONCE, here. It used to be applied again where the
    # file is written, which produced 'clean_baselines_x.json_x' — tag twice,
    # and the .json buried mid-name. Strip any .json the caller passed so
    # --out foo.json --tag x lands on foo_x.json rather than foo.json_x.json.
    _stem = a.out[:-5] if a.out.endswith(".json") else a.out
    a.out = f"{_stem}_{a.tag}.json" if a.tag else f"{_stem}.json"
    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)

    # setup_model already wrapped the model for --inference, so model(img)
    # slides or not by construction; only the warning and the record need to
    # know which happened.
    mode = getattr(model, "mode", "whole")
    if mode == "slide":
        print(SLIDE_WARNING)

    ds = make_dataset(a, "val")
    loader = DataLoader(Subset(ds, list(range(min(a.n_images, len(ds))))),
                        batch_size=1, num_workers=2)

    m = SegMetric(a.num_classes, device=device)
    per_image = []
    with torch.no_grad():
        for i, (img, lbl) in enumerate(loader):
            img, lbl = img.to(device), lbl.to(device)
            pred = upsample_to(model(img), lbl.shape[-2:]).argmax(1)
            m.update(pred, lbl)
            one = SegMetric(a.num_classes, device=device)
            one.update(pred, lbl)
            per_image.append(one.compute())

    iou = m.per_class()
    dataset_miou = m.compute()

    print(f"\n{'='*66}")
    print(f" {a.arch} @ {a.img_h}x{a.img_w} ({mode}) over "
          f"{len(per_image)} val images")
    print(f"{'='*66}")
    print(f"  DATASET mIoU  : {dataset_miou:.2f}   <- compare to published")
    pim = torch.tensor(per_image)
    print(f"  per-image mIoU: {pim.mean():.2f} +/- {pim.std():.2f} "
          f"(range {pim.min():.2f}-{pim.max():.2f})")
    print(f"                  NOT comparable to published numbers")
    print("\n  per-class IoU:")
    for c in range(min(a.num_classes, 19)):
        if not torch.isnan(iou[c]):
            print(f"    {c:2d} {class_name(c):10s}: {iou[c]:6.2f}")

    rec = {"arch": a.arch, "img_h": a.img_h, "img_w": a.img_w,
           "inference": mode, "scale": getattr(a, "scale", "resize"),
           "n_images": len(per_image), "dataset_miou": dataset_miou,
           "per_image_mean": float(pim.mean()), "per_image_std": float(pim.std()),
           "backbone_channels": n_ch, "backbone_active": n_act,
           "per_class_iou": {class_name(c): (None if torch.isnan(iou[c])
                                             else float(iou[c]))
                             for c in range(min(a.num_classes, 19))}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rec = json.loads(out.read_text()) if out.exists() else []
    all_rec.append(rec)
    out.write_text(json.dumps(all_rec, indent=2))
    print(f"\n  appended -> {out}")


if __name__ == "__main__":
    main()