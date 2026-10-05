"""Jedi — entropy-based patch localisation (Tarchoun et al., CVPR 2023)."""
from __future__ import annotations

from .autoencoder import (MaskAutoencoder, synthetic_mask_batch,
                          train_on_synthetic_masks)
from .defence import (CLEAN_MEAN, CLEAN_STD, MIN_BLOB_FRAC, PATCH_MEAN,
                      PATCH_STD, JediDefence, drop_small_blobs,
                      entropy_heatmap, entropy_threshold, load_calibration,
                      measure_entropy_stats, window_sizes)

__all__ = ["JediDefence", "window_sizes", "entropy_heatmap",
           "entropy_threshold", "drop_small_blobs", "load_calibration",
           "measure_entropy_stats", "PATCH_MEAN", "PATCH_STD", "CLEAN_MEAN",
           "CLEAN_STD", "MIN_BLOB_FRAC", "MaskAutoencoder",
           "train_on_synthetic_masks", "synthetic_mask_batch"]
