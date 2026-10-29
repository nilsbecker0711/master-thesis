#!/usr/bin/env python
"""
Train one adversarial patch with THEIR procedure, on our mmseg checkpoint.

A direct CLI port of Experiments/Main.ipynb cell 2, for a single model instead
of a loop over seven. Same four calls in the same order, so the procedure is
theirs by construction:

    config      = get_config_from_yaml(..., model=<name>)
    main_logger = process_config(config)
    train_obj   = PatchTrainer(config, main_logger)
    patch, IoU  = train_obj.train()      -> pickled to log_patch_address

WHAT IS THEIRS AND STAYS THEIRS. Scale augmentation [0.5, 2.1] then a random
1024x1024 crop (dataset/base_dataset.py); 200x200 centre patch written into the
NORMALISED image (patch/create.py); patch footprint set to ignore_label so it
leaves the loss and the metric; sign-gradient ascent on raw pixels with
clamp_(0,1) and epsilon = optimizer.init_lr (trainer/trainer.py); objective is
mean CE over correctly-classified pixels, since compute_loss returns
loss_correct (metrics/loss.py). None of that is altered here.

WHAT IS OURS. Only the model: pretrained_models/SegFormerB0/model.py loads our
mmseg checkpoint, wrapped for the configured inference mode. The dataloader
needs no normalisation change -- their ImageNet mean/std on the 0-1 scale is
bit-identical to patchreach's IMG_MEAN/IMG_STD.

CONFIG IS A NAMEDTUPLE, so overrides go through _replace rather than item
assignment.

    python scripts/train_patch.py --end_epoch 1        # one pass, to time it
    python scripts/train_patch.py                      # config as-is (30)
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision.utils import save_image

from trainer.trainer import PatchTrainer
from utils.utils import get_config_from_yaml, process_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--name", default=None,
                   help="override model.name (default: whatever the yaml says)")
    p.add_argument("--end_epoch", type=int, default=None,
                   help="override train.end_epoch. One epoch is one pass over "
                        "all 2975 train images at batch_size 1.")
    p.add_argument("--tag", default="",
                   help="appended to the pickle filename")
    a = p.parse_args()

    config = get_config_from_yaml(a.config, model=a.name)
    if a.end_epoch is not None:
        config = config._replace(
            train=config.train._replace(end_epoch=a.end_epoch))

    main_logger = process_config(config)
    main_logger.info(f"[patch] size={config.patch.size} loc={config.patch.loc} "
                     f"epsilon={config.optimizer.init_lr}")
    main_logger.info(f"[train] crop={config.train.height}x{config.train.width} "
                     f"epochs={config.train.end_epoch} "
                     f"batch={config.train.batch_size}")

    t0 = time.time()
    train_obj = PatchTrainer(config, main_logger)
    save = train_obj.train()          # (patch [3,S,S], IoU [epochs,19])
    wall = time.time() - t0

    patch, iou = save
    stem = config.model.name + (f"_{a.tag}" if a.tag else "")
    run_dir = Path(config.experiment.log_patch_address) / "patch" / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # Their own output format, unchanged, so their notebooks still load it.
    with open(config.experiment.log_patch_address + stem + ".p", "wb") as fh:
        pickle.dump(save, fh)

    # final.pt -- the patch alone, evaluable without this script.
    #
    # THE `space` FIELD IS NOT DECORATION. trainer.py optimises the patch in
    # NORMALISED space: Patch.apply_patch writes it straight into an image that
    # BaseDataset.input_transform already normalised, and clamp_(0,1) therefore
    # bounds it in normalised units, not pixel units. Inverting with the mean
    # and std below, [0,1] normalised maps to roughly R [0.485,0.714],
    # G [0.456,0.680], B [0.406,0.631] in true [0,1] pixel space -- a narrow
    # light-grey band. Load this tensor into a pipeline that expects a [0,1]
    # PIXEL patch (patchreach does) and you silently evaluate a different,
    # much darker patch than the one that was trained.
    torch.save({
        "patch": patch.detach().cpu(),          # [3,S,S]
        "space": "normalised",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "patch_size": config.patch.size,
        "patch_loc": config.patch.loc,
        "model_name": config.model.name,
        "ckpt_dir": config.model.ckpt_dir,
        "slide": config.model.slide,
        "train_crop": [config.train.height, config.train.width],
        "epochs": config.train.end_epoch,
        "epsilon": config.optimizer.init_lr,
        "ignore_label": config.train.ignore_label,
        "iou_trace": iou,                       # [epochs,19]
        "wall_clock_s": wall,
        "source": "shekhar et al. procedure, mmseg checkpoint",
    }, run_dir / "final.pt")

    # TWO PNGs, because the patch is optimised in NORMALISED space and the raw
    # tensor is not what the patch looks like.
    #
    # final_patch.png          true appearance: pixel = normalised*std + mean.
    #                          This is what the patch would look like printed
    #                          or on screen -- a narrow light-grey band,
    #                          roughly R [0.485,0.714], G [0.456,0.680],
    #                          B [0.406,0.631], because clamp_(0,1) bounds the
    #                          patch in normalised units. USE THIS IN FIGURES.
    #
    # final_patch_normspace.png the raw tensor read as if it were pixels. This
    #                          is what Main.ipynb cell 3 renders
    #                          (plt.imshow on the bare patch), so it is the
    #                          space their published patch images are in. It
    #                          shows the full [0,1] range and therefore MORE
    #                          contrast than the patch actually has. Kept for
    #                          like-for-like comparison with their figures,
    #                          not as a depiction of the patch.
    p_cpu = patch.detach().cpu().float()
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    save_image((p_cpu * std_t + mean_t).clamp(0, 1), run_dir / "final_patch.png")
    save_image(p_cpu.clamp(0, 1), run_dir / "final_patch_normspace.png")

    main_logger.info(f"[done] {wall/60:.1f} min | patch {tuple(patch.shape)} "
                     f"| IoU trace {iou.shape}")
    main_logger.info(f"[save] {run_dir / 'final.pt'}")
    main_logger.info(f"[save] {run_dir / 'final_patch.png'} (true appearance) "
                     f"+ final_patch_normspace.png (their figure space)")


if __name__ == "__main__":
    main()
