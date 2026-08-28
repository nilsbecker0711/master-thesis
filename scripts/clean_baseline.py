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
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from _common import add_model_args, setup_model
from patchreach.data.cityscapes import CityscapesSeg, class_name, upsample_to
from patchreach.metrics.miou import SegMetric
from patchreach.utils import get_device, seed_everything


def main():
    p = add_model_args(argparse.ArgumentParser())
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--out", default=f"results/clean_baselines")
    p.add_argument("--tag", type=str, default=None)
    a = p.parse_args()
    a.out= f'{a.out}_{a.tag}.json' if a.tag else a.out
    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
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
    print(f" {a.arch} @ {a.img_h}x{a.img_w} over {len(per_image)} val images")
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
           "n_images": len(per_image), "dataset_miou": dataset_miou,
           "per_image_mean": float(pim.mean()), "per_image_std": float(pim.std()),
           "backbone_channels": n_ch, "backbone_active": n_act,
           "per_class_iou": {class_name(c): (None if torch.isnan(iou[c])
                                             else float(iou[c]))
                             for c in range(min(a.num_classes, 19))}}
    out = Path(f"{a.out}_{a.tag}" if a.tag else a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rec = json.loads(out.read_text()) if out.exists() else []
    all_rec.append(rec)
    out.write_text(json.dumps(all_rec, indent=2))
    print(f"\n  appended -> {out}")


if __name__ == "__main__":
    main()