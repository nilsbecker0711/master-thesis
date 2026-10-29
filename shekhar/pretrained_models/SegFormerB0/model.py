"""
SegFormer-B0 (MiT-B0), mmsegmentation checkpoint.

Same role as the other pretrained_models/* packages: name a checkpoint and
return a ready-to-use eval-mode network. The difference is that the
architecture is not defined by a model.py in this folder -- mmseg builds it
from the config that ships beside the weights -- so this file is a loader.

THE CONFIG AND WEIGHTS FILENAMES ARE THE PAIR `mim download` WRITES. Their
word order differs from each other ("segformer_mit-b0_..." for the config,
same stem for the checkpoint plus its date hash); that asymmetry is upstream's,
not a typo.

INFERENCE MODE. get_model(slide=True) wraps the segmentor in
SlidingWindowSegModel using the crop and stride from the checkpoint's own
test_cfg, which is the thesis evaluation procedure: 1024x1024 crop, 768
stride. At a 1024x1024 input that is a single window, so patch TRAINING on
1024x1024 crops is unaffected; at the 1024x2048 evaluation frame it is three
windows, which is the point.
"""
from __future__ import annotations

import os

CFG = "segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py"
WEIGHTS = ("segformer_mit-b0_8x1_1024x1024_160k_cityscapes"
           "_20211208_101857-e7f88502.pth")


def get_model(ckpt_dir, device="cuda", slide=True, num_classes=19,
              slide_checkpoint=False, amp_dtype=None):
    """(model, info). `model` maps [B,3,H,W] -> logits [B,num_classes,H,W]."""
    from patchreach.models.registry import load_segmentor
    from patchreach.models.wrapper import (SlidingWindowSegModel,
                                           WrappedSegModel)

    cfg = os.path.join(ckpt_dir, CFG)
    weights = os.path.join(ckpt_dir, WEIGHTS)
    for p in (cfg, weights):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{p} not found. Point model.ckpt_dir in configs/config.yaml "
                f"at the folder holding the mmseg config + .pth pair, or run "
                f"`mim download mmsegmentation --config "
                f"{CFG[:-3]} --dest <that folder>`.")

    model = WrappedSegModel(load_segmentor(cfg, weights)[0],
                            amp_dtype=amp_dtype).to(device)

    info = {"cfg": cfg, "weights": weights, "inference": "whole"}
    if slide:
        tc = getattr(model.model, "test_cfg", None) or {}
        if not (tc.get("crop_size") and tc.get("stride")):
            raise RuntimeError(
                f"slide requested but test_cfg declares mode={tc.get('mode')!r} "
                f"with no crop_size/stride; pass slide=False.")
        crop, stride = tuple(tc["crop_size"]), tuple(tc["stride"])
        model = SlidingWindowSegModel(model, crop, stride, num_classes,
                                      checkpoint=slide_checkpoint).to(device)
        info["inference"] = f"slide crop={crop} stride={stride}"

    model.eval()
    return model, info
