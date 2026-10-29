#!/usr/bin/env python
"""
Evaluate a trained patch from final.pt alone, on Cityscapes val.

Protocol: native 1024x2048 frames (multi_scale=False, flip=False, so
gen_sample applies no resize and no crop), inference mode from model.slide.
Clean and patched mIoU are accumulated in the SAME pass over the same frames.

THIS REPRODUCES THEIR EVALUATION, which is split across two of their
notebooks and does NOT use one label for both numbers:

  clean    pretrained_models/CityscapeVal.ipynb -> metrics.performance
           .pixAcc_miou, val list, multi_scale=False, flip=False, no patch,
           scored against the FULL label.
  patched  Experiments/Table1and2.ipynb, same list and flags, patch written
           in at the centre and the footprint set to ignore_label.

So their clean mIoU counts the pixels the patch will later cover and their
patched mIoU does not. That is reproduced here as the headline number, and a
footprint-excluded clean baseline is reported beside it so the size of the
mismatch is visible. Everything else matches theirs by construction:

  * base_size/crop_size are DEAD on this path -- with multi_scale=False
    gen_sample never calls multi_scale_aug, the only caller of rand_crop -- so
    their train.* values and CityscapeVal's test.* values both give the native
    frame, as does this script.
  * shuffle and batch_size do not matter: SegmentationMetric pools
    total_inter/total_union, so a full pass is order- and batch-independent.
  * Patch.apply_patch's loc='center' is (w - size)//2, (h - size)//2, the same
    arithmetic their notebook hardcodes as (2048 - 200)//2, (1024 - 200)//2.

    python scripts/eval_patch.py --patch Experiments/runs/patch/<stem>/final.pt
    python scripts/eval_patch.py --patch .../final.pt --inference whole
    python scripts/eval_patch.py --patch .../final.pt --n_images 50   # quick

--inference PICKS THE MODE AND IS RECORDED EVERYWHERE. 'whole' is one forward
over the full 1024x2048 frame, which is what their GitHub does and therefore
the only setting whose numbers are comparable to their published ones. 'slide'
is the checkpoint's own published test_cfg (1024 crop, 768 stride = three
windows at this resolution), which is the thesis evaluation procedure. The
mode lands in the printout, in the json as "inference", and in the json's
FILENAME -- eval_val_whole.json vs eval_val_slide.json -- because the two are
different numbers for the same patch and must not overwrite each other.

THE PATCH TO FEED THIS IS THE ONE IN final.pt['patch'], UNCONVERTED. It is in
NORMALISED space (space='normalised'), because trainer.py optimised it by
writing it straight into an image input_transform had already normalised, and
clamp_(0,1) therefore bounded it in normalised units. Patch.apply_patch here
writes it into a normalised image exactly the same way, so the patch is
evaluated as trained. Do NOT feed it either PNG:

  final_patch.png            the TRUE APPEARANCE, pixel = normalised*std+mean,
                             a light-grey band (R 124-182, G 116-173,
                             B 104-161 in 8-bit). Read back as normalised it
                             collapses to 8-bit 127-165 -- about half the
                             contrast, a weaker patch, and nothing errors.
  final_patch_normspace.png  the right values but quantised to 1/255 steps.

Both exist for figures. If space is anything other than 'normalised' this
script refuses rather than guessing; a patchreach checkpoint (which stores
`param` in [0,1] PIXEL space, or a logit, or a GAN latent) is not loadable
here and would need rendering to pixels and then normalising first.

mIoU HERE IS THEIRS: SegmentationMetric, unmodified, pooled over the dataset,
19 classes, void (255) excluded. On the FULL val set this equals standard
pooled mIoU; on a small --n_images it does not, because classes absent from the
accumulated ground truth contribute 0 to IoU.mean() instead of being skipped.
Quote only full-set numbers.
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
    p.add_argument("--inference", choices=("config", "slide", "whole"),
                   default="config",
                   help="inference mode, overriding model.slide in the yaml. "
                        "'whole' = one forward over the full frame, their "
                        "GitHub procedure and the only mode comparable to "
                        "their published numbers. 'slide' = the checkpoint's "
                        "own test_cfg. 'config' (default) = whatever the yaml "
                        "says.")
    p.add_argument("--n_images", type=int, default=None,
                   help="evaluate only the first N frames (debugging only)")
    p.add_argument("--out", default=None,
                   help="where to write the json. Default: beside the patch, "
                        "as eval_val_<mode>.json, or eval_<n>_<mode>.json for "
                        "a partial set.")
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
    if a.inference != "config":
        config = config._replace(
            model=config.model._replace(slide=(a.inference == "slide")))
    # One name for the mode from here on, used in the printout, the json and
    # the json's filename. Derived from model.slide rather than from a.inference
    # so that the default path ('config') is labelled by what will actually run.
    mode = "slide" if config.model.slide else "whole"
    dev = config.experiment.device

    print(f"[patch] {tuple(ck['patch'].shape)} loc={ck['patch_loc']} "
          f"space={ck['space']} trained {ck['epochs']} epochs "
          f"eps={ck['epsilon']}")
    print(f"[model] {config.model.name} inference={mode}"
          f"{'' if a.inference != 'config' else ' (from yaml)'}")
    if mode == "whole":
        print("[note] whole-image inference: this is their GitHub procedure, "
              "so these numbers are the ones comparable to theirs.")
    else:
        print("[note] sliding-window inference: the checkpoint's own test_cfg, "
              "NOT their procedure -- three windows at 1024x2048, and a patch "
              "cannot influence pixels outside the windows containing it.")
    if ck["slide"] != config.model.slide:
        print(f"[note] the patch was TRAINED with slide={ck['slide']}. At the "
              f"{ck['train_crop'][0]}x{ck['train_crop'][1]} training crop that "
              f"is one window either way, so this is not a mismatch.")

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
    # THREE metrics, because their clean baseline and their patched number are
    # not scored on the same label, and that mismatch has to stay visible
    # rather than be quietly resolved one way. See the module docstring.
    #
    #   m_clean    clean logits vs the UNPATCHED label -- their baseline, the
    #              number their published drops are measured from.
    #   m_adv      patched logits vs the patched label -- their Table 2 loop.
    #   m_clean_f  clean logits vs the PATCHED label, i.e. the same pixels the
    #              patched number is scored on. NOT theirs, and strictly
    #              better: the drop is then paired, differing only by the
    #              patch rather than also by which pixels were counted.
    m_clean = SegmentationMetric(config)
    m_adv = SegmentationMetric(config)
    m_clean_f = SegmentationMetric(config)
    for m in (m_clean, m_adv, m_clean_f):
        m.reset()

    n = 0
    with torch.no_grad():
        for batch in tqdm(dl):
            image, label = batch[0].to(dev), batch[1].to(dev)

            patched_image, patched_label = apply_patch(image, label, patch)

            out_clean = model.predict(image, patched_label.shape)
            out_adv = model.predict(patched_image, patched_label.shape)

            m_clean.update(out_clean, label)              # their baseline
            m_adv.update(out_adv, patched_label)          # their patched
            m_clean_f.update(out_clean, patched_label)    # paired baseline

            n += 1
            if a.n_images is not None and n >= a.n_images:
                break

    pix_c, miou_c = m_clean.get()
    pix_a, miou_a = m_adv.get()
    pix_f, miou_f = m_clean_f.get()
    iou_c = m_clean.get(full=True)
    iou_a = m_adv.get(full=True)
    iou_f = m_clean_f.get(full=True)

    partial = a.n_images is not None and n < len(ds)
    print(f"\nframes                    : {n} of {len(ds)}"
          f"\nclean   pixAcc/mIoU       : {pix_c:.4f} / {miou_c:.4f}"
          f"   <- theirs, unpatched label"
          f"\npatched pixAcc/mIoU       : {pix_a:.4f} / {miou_a:.4f}"
          f"\ndrop mIoU (their protocol): {miou_c - miou_a:.4f}"
          f"  ({100*(miou_c-miou_a)/max(miou_c,1e-9):.2f}% relative)")
    print(f"\nclean   pixAcc/mIoU       : {pix_f:.4f} / {miou_f:.4f}"
          f"   <- footprint excluded"
          f"\ndrop mIoU (paired)        : {miou_f - miou_a:.4f}"
          f"  ({100*(miou_f-miou_a)/max(miou_f,1e-9):.2f}% relative)")
    print("\nper-class IoU (clean -> patched), standard 19-class train-id order."
          "\nclean is their protocol; the paired baseline follows in brackets:")
    for c in range(config.dataset.num_classes):
        print(f"  {c:2d}: {iou_c[c]:6.4f} -> {iou_a[c]:6.4f} "
              f"({iou_a[c]-iou_c[c]:+.4f})   [paired clean {iou_f[c]:6.4f}]")
    if partial:
        print("\n[warn] partial set -- absent classes drag mIoU down; do not quote.")

    # `partial` is recorded in the file, not just printed, so a debugging run
    # cannot later be mistaken for a full-set result.
    # The mode is IN THE FILENAME: slide and whole are different numbers for
    # the same patch, and a shared name would have the second run silently
    # overwrite the first.
    out = Path(a.out) if a.out else (
        Path(a.patch).parent
        / (f"eval_{n}_{mode}.json" if partial else f"eval_val_{mode}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({
            "patch": str(a.patch),
            "partial": partial,
            "frames": n,
            "frames_available": len(ds),
            "model_name": config.model.name,
            "ckpt_dir": config.model.ckpt_dir,
            "inference": mode,
            "inference_is_theirs": mode == "whole",
            "slide": config.model.slide,
            "slide_trained_with": ck["slide"],
            "patch_size": ck["patch_size"],
            "patch_loc": ck["patch_loc"],
            "space": ck["space"],
            "trained_epochs": ck.get("epochs"),
            "epsilon": ck.get("epsilon"),
            "train_crop": ck.get("train_crop"),
            "eval_resolution": [config.test.height, config.test.width],
            "clean_protocol": "unpatched label (theirs, CityscapeVal.ipynb)",
            "clean": {"pixAcc": float(pix_c), "mIoU": float(miou_c)},
            "patched": {"pixAcc": float(pix_a), "mIoU": float(miou_a)},
            "drop_mIoU": float(miou_c - miou_a),
            "drop_pixAcc": float(pix_c - pix_a),
            "clean_footprint_excluded": {"pixAcc": float(pix_f),
                                         "mIoU": float(miou_f)},
            "drop_mIoU_paired": float(miou_f - miou_a),
            "drop_pixAcc_paired": float(pix_f - pix_a),
            "per_class_iou": {str(c): {
                "clean": float(iou_c[c]),
                "patched": float(iou_a[c]),
                "clean_footprint_excluded": float(iou_f[c])}
                for c in range(config.dataset.num_classes)},
        }, fh, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
