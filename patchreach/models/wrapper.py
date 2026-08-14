"""Gradient-preserving wrapper around an mmseg EncoderDecoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class WrappedSegModel(nn.Module):
    """
    encode_decode path: [B,C,H,W] -> logits [B,num_classes,H_out,W_out].

    NOT model(imgs, metas, return_loss=False): that routes to aug_test, which
    (a) expects List[Tensor[1,C,H,W]] and (b) runs argmax internally,
    destroying the gradients the patch optimisation depends on.

    encode_decode is defined on EncoderDecoder, so this works identically for
    InternImage, SegFormer and Swin with no per-arch branching.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _metas(B: int, H: int, W: int) -> list:
        return [{"ori_shape": (H, W, 3), "img_shape": (H, W, 3),
                 "pad_shape": (H, W, 3), "scale_factor": 1.0,
                 "flip": False, "flip_direction": None} for _ in range(B)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        return self.model.encode_decode(x, self._metas(B, H, W))
