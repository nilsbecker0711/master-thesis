#!/usr/bin/env python
"""
Per-iteration diagnosis of the patch training loop.

The 1-epoch run logged `Loss: nan` at EVERY sampled iteration and ended at
pixAcc 0.045. Two very different causes produce that pair:

  (a) the attack works, nothing is correctly classified any more, so
      correct_mask.sum() == 0 and loss_correct is a genuine 0/0; or
  (b) the masks never match, so correct_mask.sum() == 0 from the very first
      iteration and no attack is being measured at all.

This prints, per iteration, the quantities that tell them apart WITHOUT
touching the training code: label coverage, how many pixels the CLEAN model
gets right, how many the PATCHED model gets right, the loss, whether the
gradient is finite, and the patch's range.

Read it like this:
  * clean_correct% high (>80) at iter 0 and falling as iterations pass -> (a),
    the attack is working and the NaN is the unguarded denominator firing once
    the attack is total.
  * clean_correct% near 0 at iter 0 -> (b), the model/label/metric wiring is
    wrong and nothing downstream means anything.
  * labeled% near 0 on some iterations -> void-only crops, which is the other
    way 0/0 arises (rand_scale up to 2.1 can crop entirely into ego-vehicle
    hood).

    python scripts/diagnose_train.py --iters 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dataset.cityscapes import Cityscapes
from metrics.loss import PatchLoss
from patch.create import Patch
from pretrained_models.models import Models
from utils.utils import get_config_from_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--name", default=None)
    p.add_argument("--iters", type=int, default=25)
    a = p.parse_args()

    config = get_config_from_yaml(a.config, model=a.name)
    dev = config.experiment.device
    print(f"model={config.model.name} slide={config.model.slide} dev={dev} "
          f"patch={config.patch.size} eps={config.optimizer.init_lr}")

    model = Models(config)
    model.get()

    ds = Cityscapes(root=config.dataset.root, list_path=config.dataset.train,
                    num_classes=config.dataset.num_classes,
                    multi_scale=config.train.multi_scale, flip=config.train.flip,
                    ignore_label=config.train.ignore_label,
                    base_size=config.train.base_size,
                    crop_size=(config.train.height, config.train.width),
                    scale_factor=config.train.scale_factor, make_edge=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                     num_workers=0)

    patch = torch.rand((3, config.patch.size, config.patch.size),
                       requires_grad=True, device=dev)
    apply_patch = Patch(config).apply_patch
    criterion = PatchLoss(config)
    eps = config.optimizer.init_lr

    hdr = (f"{'it':>3} {'labeled%':>9} {'clean_ok%':>10} {'adv_ok%':>8} "
           f"{'loss':>10} {'grad':>10} {'patch[min,max]':>18}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for it, batch in enumerate(dl):
        if it >= a.iters:
            break
        image, label = batch[0].to(dev), batch[1].to(dev)
        patched_image, patched_label = apply_patch(image, label, patch)

        # Same bookkeeping PatchLoss does, so the numbers are its numbers.
        tgt = patched_label.float() + 1
        tgt[tgt >= 255] = 0
        labeled = (tgt > 0)
        n_lab = labeled.sum().item()

        with torch.no_grad():
            out_clean = model.predict(image, patched_label.shape)
            pred_c = out_clean.argmax(1).float() + 1
            n_ok_c = ((pred_c == tgt) * labeled).sum().item()

        out_adv = model.predict(patched_image, patched_label.shape)
        with torch.no_grad():
            pred_a = out_adv.argmax(1).float() + 1
            n_ok_a = ((pred_a == tgt) * labeled).sum().item()

        loss = criterion.compute_loss(out_adv, patched_label)
        if patch.grad is not None:
            patch.grad.zero_()
        loss.backward()
        g = patch.grad
        gfin = "None" if g is None else (
            f"{g.norm().item():.3e}" if torch.isfinite(g).all()
            else f"NONFINITE({torch.isnan(g).sum().item()}nan,"
                 f"{torch.isinf(g).sum().item()}inf)")

        tot = tgt.numel()
        print(f"{it:3d} {100*n_lab/tot:8.2f}% "
              f"{100*n_ok_c/max(n_lab,1):9.2f}% "
              f"{100*n_ok_a/max(n_lab,1):7.2f}% "
              f"{loss.item():10.4f} {gfin:>10} "
              f"[{patch.min().item():.3f},{patch.max().item():.3f}]")

        with torch.no_grad():
            if g is not None:
                patch += eps * g.data.sign()
                patch.clamp_(0, 1)

    print("\nclean_ok% is the model on the UNPATCHED frame -- if that is not "
          "high, the problem is wiring, not the attack.")


if __name__ == "__main__":
    main()
