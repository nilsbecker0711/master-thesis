#!/usr/bin/env python
"""
Evaluate a trained patch from final.pt alone, on Cityscapes val.

Protocol: native 1024x2048 frames (multi_scale=False, flip=False, so
gen_sample applies no resize and no crop), sliding-window inference with the
crop and stride from the checkpoint's own test_cfg. Clean and patched mIoU are
accumulated in the SAME pass, so the drop is paired per frame rather than
being a difference of two independently-run numbers.

    python scripts/eval_patch.py --patch Experiments/runs/patch/<stem>/final.pt
    python scripts/eval_patch.py --patch .../final.pt --n_images 50   # quick

THE PATCH IS IN NORMALISED SPACE. final.pt records space='normalised' because
trainer.py writes the patch into an already-normalised image. This script
applies it the same way, via their own Patch.apply_patch, so the patch is
evaluated exactly as trained. If space is anything else the script refuses
rather than guessing -- a pixel-space patch would need converting first.

mIoU HERE IS THEIRS: SegmentationMetric, pooled over the dataset, 19 classes,
void (255) excluded, and the patch footprint excluded too because
apply_patch overwrites the label under the patch with ignore_label. On the
FULL val set this equals standard pooled mIoU; on a small --n_images it does
not, because classes absent from the accumulated ground truth contribute 0 to
IoU.mean() instead of being skipped. Quote only full-set numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.cityscapes import Cityscapes
from metrics.performance import SegmentationMetric
from patch.create import Patch
from pretrained_models.models import Models
from utils.utils import get_config_from_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--patch", required=True, help="path to final.pt")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--name", default=None, help="override model.name")
    p.add_argument("--n_images", type=int, default=None,
                   help="evaluate only the first N frames (debugging only)")
    p.add_argument("--out", default=None,
                   help="where to write the json. Default: beside the patch, "
                        "as eval_<n>.json or eval_val.json for the full set.")
    a = p.parse_args()

    ck = torch.load(a.patch, map_location="cpu")
    if ck.get("space") != "normalised":
        raise SystemExit(
            f"final.pt declares space={ck.get('space')!r}; this script applies "
            f"the patch into a normalised image and will not guess a "
            f"conversion. See the note in the module docstring.")

    config = get_config_from_yaml(a.config, model=a.name or ck["model_name"])
    # The patch's own size wins over the config: the tensor is what it is.
    if config.patch.size != ck["patch_size"]:
        print(f"[note] config patch.size={config.patch.size} but checkpoint is "
              f"{ck['patch_size']}; using the checkpoint's.")
        config = config._replace(
            patch=config.patch._replace(size=ck["patch_size"]))
    dev = config.experiment.device

    print(f"[patch] {tuple(ck['patch'].shape)} loc={ck['patch_loc']} "
          f"trained {ck['epochs']} epochs eps={ck['epsilon']}")
    print(f"[model] {config.model.name} slide={config.model.slide}")

    model = Models(config)
    model.get()

    ds = Cityscapes(root=config.dataset.root, list_path=config.dataset.val,
                    num_classes=config.dataset.num_classes,
                    multi_scale=False, flip=False,
                    ignore_label=config.train.ignore_label,
                    base_size=config.test.base_size,
                    crop_size=(config.test.height, config.test.width))
    dl = torch.utils.data.DataLoader(
        dataset=ds, batch_size=1, shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=config.test.pin_memory, drop_last=False)

    patch = ck["patch"].to(dev)
    apply_patch = Patch(config).apply_patch
    m_clean, m_adv = SegmentationMetric(config), SegmentationMetric(config)
    m_clean.reset()
    m_adv.reset()

    n = 0
    with torch.no_grad():
        for batch in tqdm(dl):
            image, label = batch[0].to(dev), batch[1].to(dev)

            patched_image, patched_label = apply_patch(image, label, patch)

            # Score the clean frame on the SAME label as the patched one, so
            # the two numbers differ only by the patch and the drop is exact.
            out_clean = model.predict(image, patched_label.shape)
            out_adv = model.predict(patched_image, patched_label.shape)

            m_clean.update(out_clean, patched_label)
            m_adv.update(out_adv, patched_label)

            n += 1
            if a.n_images is not None and n >= a.n_images:
                break

    pix_c, miou_c = m_clean.get()
    pix_a, miou_a = m_adv.get()
    iou_c = m_clean.get(full=True)
    iou_a = m_adv.get(full=True)

    partial = a.n_images is not None and n < len(ds)
    print(f"\nframes            : {n} of {len(ds)}"
          f"\nclean  pixAcc/mIoU: {pix_c:.4f} / {miou_c:.4f}"
          f"\npatched pixAcc/mIoU: {pix_a:.4f} / {miou_a:.4f}"
          f"\ndrop mIoU         : {miou_c - miou_a:.4f}"
          f"  ({100*(miou_c-miou_a)/max(miou_c,1e-9):.2f}% relative)")
    print("\nper-class IoU (clean -> patched), standard 19-class train-id order:")
    for c in range(config.dataset.num_classes):
        print(f"  {c:2d}: {iou_c[c]:6.4f} -> {iou_a[c]:6.4f} "
              f"({iou_a[c]-iou_c[c]:+.4f})")
    if partial:
        print("\n[warn] partial set -- absent classes drag mIoU down; do not quote.")

    # `partial` is recorded in the file, not just printed, so a debugging run
    # cannot later be mistaken for a full-set result.
    out = Path(a.out) if a.out else (
        Path(a.patch).parent
        / (f"eval_{n}.json" if partial else "eval_val.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({
            "patch": str(a.patch),
            "partial": partial,
            "frames": n,
            "frames_available": len(ds),
            "model_name": config.model.name,
            "ckpt_dir": config.model.ckpt_dir,
            "slide": config.model.slide,
            "patch_size": ck["patch_size"],
            "patch_loc": ck["patch_loc"],
            "space": ck["space"],
            "trained_epochs": ck.get("epochs"),
            "epsilon": ck.get("epsilon"),
            "train_crop": ck.get("train_crop"),
            "eval_resolution": [config.test.height, config.test.width],
            "clean": {"pixAcc": float(pix_c), "mIoU": float(miou_c)},
            "patched": {"pixAcc": float(pix_a), "mIoU": float(miou_a)},
            "drop_mIoU": float(miou_c - miou_a),
            "drop_pixAcc": float(pix_c - pix_a),
            "per_class_iou": {str(c): {"clean": float(iou_c[c]),
                                       "patched": float(iou_a[c])}
                              for c in range(config.dataset.num_classes)},
        }, fh, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
