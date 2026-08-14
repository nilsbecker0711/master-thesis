"""
Small utilities, vendored so the package has no project-local dependencies.

Replaces ensemble_tool.utils. Keeping them here means the repo installs and
runs standalone, which matters for the thesis: results should be reproducible
from this repo alone.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok: bool = False) -> Path:
    """`runs/exp` -> `runs/exp_1` if it exists. Never overwrites."""
    path = Path(path)
    if not path.exists() or exist_ok:
        return path
    n = 1
    while Path(f"{path}_{n}").exists():
        n += 1
    return Path(f"{path}_{n}")


def channel_probe(model, device, h: int = 512, w: int = 1024):
    """
    (n_channels, n_numerically_active) of a segmentation head.

    Some checkpoints emit an ADE20K-dimensioned 150-channel head even on
    Cityscapes, with channels 19..149 inert. Diagnostics index by channel, so
    probe rather than assume.

    Uses amax, NOT Tensor.any — multi-dim any/all only landed in torch 2.0 and
    this codebase targets 1.11.
    """
    with torch.no_grad():
        out = model(torch.zeros(1, 3, h, w, device=device))
        return out.shape[1], int((out.abs().amax(dim=(0, 2, 3)) > 1e-3).sum())
