"""Gradient-preserving wrappers around an mmseg EncoderDecoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _bf16_kernels_ok(device: str) -> bool:
    """
    Does this torch build have bf16 kernels for the ops the heads use?

    Hardware support is not enough. On the cluster's torch 1.11 an Ampere GPU
    reports is_bf16_supported(), yet the CUDA bilinear upsample (every
    decode head's resize) is dispatched for fp32/fp16 only and raises
    "upsample_bilinear2d_out_frame not implemented for 'BFloat16'". Probe
    that op, forward and backward, instead of trusting the capability flag.
    """
    try:
        x = torch.zeros(1, 1, 4, 4, device=device, dtype=torch.bfloat16,
                        requires_grad=True)
        # autograd.grad rather than a backward call: test_tsallis finds
        # optimisation loops by scanning source for one, and this is not one.
        y = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False).sum()
        torch.autograd.grad(y, x)
        return True
    except RuntimeError:
        return False


def low_precision_dtype() -> torch.dtype:
    """
    What --low_pr runs in: bf16 where this torch build can run it, else fp16
    (with the gradient scaling in WrappedSegModel). CPU autocast only supports
    bf16, so a CPU run gets bf16 regardless.
    """
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
    """
    encode_decode path: [B,C,H,W] -> logits [B,num_classes,H_out,W_out].

    NOT model(imgs, metas, return_loss=False): that routes to aug_test, which
    (a) expects List[Tensor[1,C,H,W]] and (b) runs argmax internally,
    destroying the gradients the patch optimisation depends on.

    encode_decode is defined on EncoderDecoder, so this works identically for
    InternImage, SegFormer and Swin with no per-arch branching.

    LOW PRECISION (amp_dtype set, i.e. --low_pr). The forward runs under
    autocast: weights stay fp32, matmuls/convs run and store activations in
    16 bit, softmax/LayerNorm stay fp32 (autocast's own op lists). Logits come
    back as fp32 so stitching, losses, metrics and the CSF budget are
    untouched. Living here rather than in the slide wrapper means whole and
    slide both get it, and a checkpointed window recomputes under the same
    autocast because the context is inside the checkpointed call.

    fp16 ONLY: the attack gradient is a per-pixel MEAN over ~1e6 pixels, so
    per-logit gradients sit around 1e-6 — below fp16's normal range, where
    they flush to zero and the patch silently stops moving. The gradient is
    therefore multiplied by GRAD_SCALE on entry to the 16-bit graph and divided
    back out at the input, so callers see exactly the fp32-scaled gradient and
    no optimisation loop needs to know. bf16 has fp32's exponent range and
    needs none of this.
    """

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
            x = _GradScale.apply(x, 1.0 / scale)       # undo at the input
        with torch.autocast(x.device.type, dtype=self.amp_dtype):
            out = self.model.encode_decode(x, self._metas(B, H, W))
        out = out.float()
        if scale != 1.0 and torch.is_grad_enabled():
            out = _GradScale.apply(out, scale)          # lift into fp16 range
        return out


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

    REENTRANT, and only when the window needs a gradient. torch 1.11's
    non-reentrant checkpoint (use_reentrant=False) stashes every recomputed
    activation in a closure list that is never cleared, so all windows end up
    live at once — no saving at all — and the list sits in a reference cycle
    with the recomputed graph, so it can outlive the step. A b0 slide attack
    OOMed an 80 GB card at step 2 that way. The reentrant form recomputes and
    back-propagates one window inside its own backward and frees it before
    the next. Its one requirement is an input that requires grad (else the
    output silently carries none), hence the guard; without one there is
    nothing to back-propagate and the plain forward is used. It does not
    support autograd.grad() through the window, which only Grad-CAM
    placement does — do not combine that with --slide_checkpoint.
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
            if self.checkpoint and torch.is_grad_enabled() and win.requires_grad:
                logit = cp.checkpoint(self.model, win, use_reentrant=True)
            else:
                logit = self.model(win)
            preds = preds + F.pad(logit, (x1, W - x2, y1, H - y2))
            count[:, :, y1:y2, x1:x2] += 1
        return preds / count


@torch.no_grad()
def slide_logits(model, img, crop, stride, num_classes):
    """Inference-only slide. Kept for callers that want it without a wrapper."""
    return SlidingWindowSegModel(model, crop, stride, num_classes)(img)
