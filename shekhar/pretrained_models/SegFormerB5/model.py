"""
SegFormer-B5 (MiT-B5), mmsegmentation checkpoint.

The b5 counterpart of ../SegFormerB0/model.py -- see that file for why this is
a loader rather than an architecture definition, and for what slide=True does.

WHY B5 IS THE ONE THAT MATTERS FOR THE COMPARISON. The SegFormer row in Shekhar
et al. is HuggingFace's nvidia/segformer-b5-finetuned-cityscapes-1024-1024
(pretrained_models/models.py, HF branch), i.e. b5 -- not b0. Their recorded
clean mIoU is 0.8129 (pretrained_models/CityscapeVal.ipynb, section 7.3)
against mmseg's published 82.25 for the same model. So b5 is the apples-to-
apples arm; b0 is a smaller-model data point with no published counterpart.
"""
from __future__ import annotations

import os

CFG = "segformer_b5.py"
WEIGHTS = ("segformer_mit-b5_8x1_1024x1024_160k_cityscapes"
           "_20211206_072934-87a052ec.pth")


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
