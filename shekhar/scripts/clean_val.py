#!/usr/bin/env python
"""
Clean mIoU over the full Cityscapes val set, for the model in config.yaml.

WHY THIS AND NOT THE SMOKE TEST'S NUMBER. SegmentationMetric averages IoU over
all 19 classes, and a class absent from the accumulated ground truth
contributes 0/eps = 0 to that mean rather than being skipped. On ONE 1024x1024
crop with ~10 classes present the metric therefore cannot exceed ~10/19, which
is why smoke_mmseg.py reports ~0.39 for a model whose pixel accuracy is 0.97.
Pooled over 500 frames every class appears and the number becomes meaningful.

THE COMPARISON THIS IS FOR. pretrained_models/CityscapeVal.ipynb section 7.3
records 0.8129 for HuggingFace SegFormer-B5 at native 1024x2048, whole-frame,
using this same metric. mmseg's published figures are 82.25 (b5) and 76.54
(b0). So:

  --slide False, name mmseg_segformer_b5  -> should land near 0.81-0.82, which
      validates this fork's dataloader and metric against both their notebook
      and mmseg.
  --slide True                            -> the same model under the thesis
      evaluation procedure. The gap between the two IS the tiling effect,
      measured rather than argued.

    python scripts/clean_val.py                  # config as-is
    python scripts/clean_val.py --slide False     # whole-frame, to match them
    python scripts/clean_val.py --name mmseg_segformer_b5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from addict import Dict

from dataset.cityscapes import Cityscapes
from metrics.performance import pixAcc_miou
from pretrained_models.models import Models


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--name", default=None,
                   help="override model.name, e.g. mmseg_segformer_b5")
    p.add_argument("--slide", default=None, type=_bool,
                   help="override model.slide")
    p.add_argument("--batch_size", default=None, type=int)
    a = p.parse_args()

    cfg = Dict(yaml.safe_load(open(a.config)))
    if a.name is not None:
        cfg.model.name = a.name
    if a.slide is not None:
        cfg.model.slide = a.slide
    # Match get_config_from_yaml, which overrides the yaml's device with this
    # check rather than trusting the string.
    cfg.experiment.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = cfg.experiment.device
    print(f"model.name = {cfg.model.name} | slide = {cfg.model.slide}")
    print(f"[device] {dev}"
          + (f" | {torch.cuda.get_device_name(0)} | "
             f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB"
             if dev == 'cuda' else ""))

    model = Models(cfg)
    model.get()

    # multi_scale=False and flip=False, so gen_sample applies NO resize and no
    # crop -- crop_size is only read inside multi_scale_aug. The frames reach
    # the model at native 2048x1024, which is what their notebook evaluates at.
    ds = Cityscapes(root=cfg.dataset.root, list_path=cfg.dataset.val,
                    num_classes=cfg.dataset.num_classes,
                    multi_scale=False, flip=False,
                    ignore_label=cfg.train.ignore_label,
                    base_size=cfg.test.base_size,
                    crop_size=(cfg.test.height, cfg.test.width))
    print(f"val entries: {len(ds)}")

    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=a.batch_size or cfg.test.batch_size,
        shuffle=False, num_workers=cfg.test.num_workers,
        pin_memory=cfg.test.pin_memory, drop_last=cfg.test.drop_last)

    # pixAcc_miou dispatches on `name` and calls model(image) directly; the
    # mmseg arm in metrics/performance.py handles the bare-tensor output.
    pixAcc_miou(cfg, model=model.model, dataloader=dl, device=dev,
                name=cfg.model.name)


if __name__ == "__main__":
    main()
