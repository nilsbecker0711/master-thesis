#!/usr/bin/env python
"""
One-batch smoke test of the mmseg arm of this fork.

Checks, in order: config loads, the patchreach registry resolves the arch and
the checkpoint, the dataloader produces a 1024x1024 scale-augmented crop,
Patch.apply_patch writes into it, the segmentor forwards, PatchLoss backwards
into the patch, and SegmentationMetric runs. Any of those breaking is a wiring
fault, and finding it here costs seconds instead of an epoch.

    python scripts/smoke_mmseg.py
"""
from __future__ import annotations

import torch
import yaml
from addict import Dict

from dataset.cityscapes import Cityscapes
from metrics.loss import PatchLoss
from metrics.performance import SegmentationMetric
from patch.create import Patch
from pretrained_models.models import Models


def main() -> None:
    cfg = Dict(yaml.safe_load(open("configs/config.yaml")))
    # Match get_config_from_yaml, which overrides the yaml's device with this
    # check. Reading the yaml string directly would hard-fail on a CPU-only
    # box instead of falling back.
    cfg.experiment.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = cfg.experiment.device
    print(f"[device] {dev}"
          + (f" | {torch.cuda.get_device_name(0)} | "
             f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB"
             if dev == 'cuda' else ""))
    print(f"model.name = {cfg.model.name} | ckpt_dir = {cfg.model.ckpt_dir} "
          f"| slide = {cfg.model.slide}")

    model = Models(cfg)
    model.get()
    print(f"model type: {type(model.model).__name__}")

    ds = Cityscapes(root=cfg.dataset.root, list_path=cfg.dataset.train,
                    num_classes=cfg.dataset.num_classes,
                    multi_scale=cfg.train.multi_scale, flip=cfg.train.flip,
                    ignore_label=cfg.train.ignore_label,
                    base_size=cfg.train.base_size,
                    crop_size=(cfg.train.height, cfg.train.width),
                    scale_factor=cfg.train.scale_factor)
    print(f"train entries: {len(ds)}")

    img, lbl, _edge, _size, _name = ds[0]
    img = torch.from_numpy(img)[None].float().to(dev)
    lbl = torch.from_numpy(lbl)[None].long().to(dev)
    print(f"img {tuple(img.shape)} range {img.min():.3f}..{img.max():.3f} "
          f"| lbl {tuple(lbl.shape)} uniq {lbl.unique().numel()}")

    patch = torch.rand((3, cfg.patch.size, cfg.patch.size),
                       requires_grad=True, device=dev)
    patched_img, patched_lbl = Patch(cfg).apply_patch(img, lbl, patch)

    out = model.predict(patched_img, patched_lbl.shape)
    print(f"logits {tuple(out.shape)}")

    loss = PatchLoss(cfg).compute_loss(out, patched_lbl)
    loss.backward()
    assert patch.grad is not None, "no gradient reached the patch"
    print(f"loss {loss.item():.4f} | patch grad norm {patch.grad.norm():.4e}")

    met = SegmentationMetric(cfg)
    met.update(out, patched_lbl)
    pix, miou = met.get()
    print(f"pixAcc {pix:.4f} | mIoU {miou:.4f}")

    # Proof the work actually happened on the GPU rather than silently on CPU:
    # a nonzero peak allocation can only come from real device tensors.
    if dev == 'cuda':
        print(f"[gpu] logits on {out.device} | patch on {patch.device} | "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
    print("OK")


if __name__ == "__main__":
    main()
