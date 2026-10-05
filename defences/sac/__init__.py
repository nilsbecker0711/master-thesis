"""SAC — Segment and Complete (Liu et al., CVPR 2022). See defence.py."""
from __future__ import annotations

from .defence import (SAC_BREAK_ITS, SAC_DECLARED_SQUARE_SIZES,
                      SAC_INIT_L1_THRESH, SAC_SQUARE_SIZES, SAC_THRESH_DECAY,
                      SACDefence, completion_pass, shape_completion)
from .unet import UNet, load_sac_unet

__all__ = ["SACDefence", "shape_completion", "completion_pass",
           "SAC_SQUARE_SIZES", "SAC_DECLARED_SQUARE_SIZES",
           "SAC_INIT_L1_THRESH", "SAC_THRESH_DECAY", "SAC_BREAK_ITS",
           "UNet", "load_sac_unet"]
