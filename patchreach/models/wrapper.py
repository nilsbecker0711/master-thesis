"""Gradient-preserving wrappers around an mmseg EncoderDecoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _bf16_kernels_ok(device: str) -> bool:
    """Does this torch build have bf16 kernels for the ops the heads use?"""
    try:
        x = torch.zeros(1, 1, 4, 4, device=device, dtype=torch.bfloat16,
                        requires_grad=True)
        y = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False).sum()
        torch.autograd.grad(y, x)
        return True
    except RuntimeError:
        return False


def low_precision_dtype() -> torch.dtype:
    """What --low_pr runs in: bf16 where this torch build can run it, else fp16 (with the gradient scaling in WrappedSegModel)."""
    if not torch.cuda.is_available():
        return torch.bfloat16
    if torch.cuda.is_bf16_supported() and _bf16_kernels_ok("cuda"):
        return torch.bfloat16
    return torch.float16


class _GradScale(torch.autograd.Function):
    """Identity forward; backward multiplies the incoming gradient by `scale`."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


class WrappedSegModel(nn.Module):
    """encode_decode path: [B,C,H,W] -> logits [B,num_classes,H_out,W_out]."""

    mode = "whole"
    GRAD_SCALE = 2.0 ** 10

    def __init__(self, model: nn.Module, amp_dtype: torch.dtype | None = None):
        super().__init__()
        self.model = model
        self.amp_dtype = amp_dtype
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
        if self.amp_dtype is None:
            return self.model.encode_decode(x, self._metas(B, H, W))
        scale = self.GRAD_SCALE if self.amp_dtype == torch.float16 else 1.0
        if scale != 1.0 and torch.is_grad_enabled():
            x = _GradScale.apply(x, 1.0 / scale)
        with torch.autocast(x.device.type, dtype=self.amp_dtype):
            out = self.model.encode_decode(x, self._metas(B, H, W))
        out = out.float()
        if scale != 1.0 and torch.is_grad_enabled():
            out = _GradScale.apply(out, scale)
        return out


def slide_windows(H: int, W: int, crop, stride):
    """Yield (y1, y2, x1, x2) per window — mmseg's slide_inference grid."""
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
    """mmseg's slide_inference, DIFFERENTIABLE."""

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
        preds = x.new_zeros((B, self.num_classes, H, W))
        count = x.new_zeros((1, 1, H, W))
        for y1, y2, x1, x2 in slide_windows(H, W, self.crop, self.stride):
            win = x[:, :, y1:y2, x1:x2]
            if self.checkpoint and torch.is_grad_enabled() and win.requires_grad:
                logit = cp.checkpoint(self.model, win, use_reentrant=True)
            else:
                logit = self.model(win)
            preds = preds + F.pad(logit, (x1, W - x2, y1, H - y2))
            count[:, :, y1:y2, x1:x2] += 1
        return preds / count


@torch.no_grad()
def slide_logits(model, img, crop, stride, num_classes):
    """Inference-only slide."""
    return SlidingWindowSegModel(model, crop, stride, num_classes)(img)
