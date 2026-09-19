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


@torch.no_grad()
def slide_logits(model, img, crop, stride, num_classes):
    r"""
    mmseg's slide_inference, reproduced. Overlapping crops, logits accumulated
    and divided by the per-pixel window count.

    WHY THIS EXISTS AND WHY forward() IS NOT IT. WrappedSegModel.forward is one
    whole-image encode_decode, which is what patch optimisation needs: a single
    differentiable pass. But SegFormer's and SETR's PUBLISHED Cityscapes numbers
    come from sliding-window inference — their configs carry
    test_cfg=dict(mode='slide', crop_size=..., stride=...), mmseg's inference()
    honours it, and encode_decode does not. A whole forward reads several points
    low against those numbers, and that is the PROCEDURE differing, not the
    checkpoint being wrong. DeepLab and UNet declare mode='whole', so for them
    the published procedure IS forward() and this function is never needed.

    NOT FOR ATTACKS. Three overlapping forwards with averaged logits smear the
    patch gradient across windows and triple the cost. The attack path stays on
    forward(); this is for clean baselines only.
    """
    B, _, H, W = img.shape
    ch, cw = crop
    sh, sw = stride
    h_grids = max(H - ch + sh - 1, 0) // sh + 1
    w_grids = max(W - cw + sw - 1, 0) // sw + 1
    preds = img.new_zeros((B, num_classes, H, W))
    count = img.new_zeros((B, 1, H, W))
    for hi in range(h_grids):
        for wi in range(w_grids):
            y1, x1 = hi * sh, wi * sw
            y2, x2 = min(y1 + ch, H), min(x1 + cw, W)
            y1, x1 = max(y2 - ch, 0), max(x2 - cw, 0)
            preds[:, :, y1:y2, x1:x2] += model(img[:, :, y1:y2, x1:x2])
            count[:, :, y1:y2, x1:x2] += 1
    return preds / count
