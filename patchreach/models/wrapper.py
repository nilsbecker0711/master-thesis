"""Gradient-preserving wrappers around an mmseg EncoderDecoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class WrappedSegModel(nn.Module):
    """
    encode_decode path: [B,C,H,W] -> logits [B,num_classes,H_out,W_out].

    NOT model(imgs, metas, return_loss=False): that routes to aug_test, which
    (a) expects List[Tensor[1,C,H,W]] and (b) runs argmax internally,
    destroying the gradients the patch optimisation depends on.

    encode_decode is defined on EncoderDecoder, so this works identically for
    InternImage, SegFormer and Swin with no per-arch branching.
    """

    mode = "whole"

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


def slide_windows(H: int, W: int, crop, stride):
    """
    Yield (y1, y2, x1, x2) per window — mmseg's slide_inference grid.

    The max()/min() clamping is mmseg's and is load-bearing: a frame SMALLER
    than the crop yields exactly one window covering the whole frame rather
    than an out-of-range slice, so a slide model still works when probed at a
    small resolution (channel_probe does exactly that).
    """
    ch, cw = crop
    sh, sw = stride
    h_grids = max(H - ch + sh - 1, 0) // sh + 1
    w_grids = max(W - cw + sw - 1, 0) // sw + 1
    for hi in range(h_grids):
        for wi in range(w_grids):
            y1, x1 = hi * sh, wi * sw
            y2, x2 = min(y1 + ch, H), min(x1 + cw, W)
            yield max(y2 - ch, 0), y2, max(x2 - cw, 0), x2


class SlidingWindowSegModel(nn.Module):
    r"""
    mmseg's slide_inference, DIFFERENTIABLE. Same contract as WrappedSegModel —
    [B,3,H,W] -> logits [B,K,H,W] — so every existing call site keeps working
    and the attack path needs no per-forward branching.

    WHY A MODULE AND NOT A FLAG. There are ~30 `model(x)` forwards across
    scripts/, patchreach/patch/ and patchreach/diagnostics/. Threading an
    inference mode through all of them would be invasive and easy to get
    half-right; wrapping the model instead means the mode is chosen once in
    setup_model and is honoured everywhere by construction.

    WHEN TO USE IT. Only to match a checkpoint's PUBLISHED inference procedure
    (SegFormer and SETR declare test_cfg mode='slide'; DeepLab and UNet declare
    'whole'). It is NOT the right mode for the reach measurement: each window is
    an independent forward, so a patch cannot influence pixels outside the
    windows containing it. At 1024x2048 with a centred 128px patch that caps
    reach at 448/704px L/R for setr_pup while segformer and deeplab reach 960 —
    an architecture-dependent ceiling on the quantity aggregate.py bins out to
    1200px. Cross-architecture comparisons stay on whole.

    MEMORY. The loss is computed on the STITCHED logits, so the graphs of all
    windows are live at backward — roughly n_windows times one window's
    activations. checkpoint=True trades that for one window's worth by
    recomputing each window during backward, at the cost of a second forward.
    """

    mode = "slide"

    def __init__(self, model: nn.Module, crop, stride, num_classes: int,
                 checkpoint: bool = False):
        super().__init__()
        self.model = model
        self.crop = tuple(int(v) for v in crop)
        self.stride = tuple(int(v) for v in stride)
        self.num_classes = int(num_classes)
        self.checkpoint = bool(checkpoint)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # Accumulated OUT OF PLACE via F.pad, as mmseg does. An in-place
        # slice-add into a tensor carrying grad history is autograd-legal but
        # fragile; this form is unambiguous and costs one full-size buffer per
        # window, which is small next to the activations.
        preds = x.new_zeros((B, self.num_classes, H, W))
        count = x.new_zeros((1, 1, H, W))          # never needs grad
        for y1, y2, x1, x2 in slide_windows(H, W, self.crop, self.stride):
            win = x[:, :, y1:y2, x1:x2]
            if self.checkpoint and torch.is_grad_enabled():
                logit = cp.checkpoint(self.model, win, use_reentrant=False)
            else:
                logit = self.model(win)
            preds = preds + F.pad(logit, (x1, W - x2, y1, H - y2))
            count[:, :, y1:y2, x1:x2] += 1
        return preds / count


@torch.no_grad()
def slide_logits(model, img, crop, stride, num_classes):
    """Inference-only slide. Kept for callers that want it without a wrapper."""
    return SlidingWindowSegModel(model, crop, stride, num_classes)(img)
