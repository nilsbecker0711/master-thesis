r"""
SAC — Segment and Complete (Liu et al., CVPR 2022). arXiv:2112.04532.
Reference implementation: github.com/joellliu/SegmentAndComplete.

Two stages, and they fail differently:

  1. SEGMENT. A U-Net, self-adversarially trained, emits a per-pixel
     "is this patch" logit. Threshold at 0.5 -> `raw_mask`.
  2. COMPLETE. The raw mask is ragged, so every candidate AXIS-ALIGNED SQUARE
     within normalised Hamming distance (1 - l1_thresh) of the raw mask is
     accepted, and the final mask is their union. If nothing is accepted the
     threshold is loosened and the search repeats. This is what buys the
     paper's removal guarantee: a raw mask close enough to the true square
     completes TO that square, so the patch is fully covered even though the
     segmenter only found part of it.

SAC ZEROES, IT DOES NOT INPAINT. x_def = x01 * (1 - mask), so the removed
region becomes BLACK. That is the published method and it is kept faithfully,
but it is a confound worth stating in the write-up rather than discovering
later: a black square is itself a strong stimulus for a segmenter, so a
"recovered" mIoU is recovered despite a hard occlusion, not by restoring the
scene. drop_remote excludes the footprint, which is precisely why it is the
headline number here — inside the footprint the comparison is
patch-vs-black-square and means nothing.

THREE DEVIATIONS FROM THE PAPER, ALL DELIBERATE, ALL TO BE STATED:

  1. DOMAIN. coco_at.pth was self-adversarially trained on COCO patches for
     object DETECTION at detection resolutions. We run it on Cityscapes
     512x1024 for SEGMENTATION, on contrast-budgeted residuals it never saw.
     This is the off-the-shelf arm. Retraining their self-adversarial loop on
     our own patches is a separate, stronger arm — only worth building if the
     off-the-shelf segmenter already catches something.
  2. SQUARE SIZES. The default is [100,75,50,25] — the list the released code
     actually executes, see SAC_SQUARE_SIZES for why that is not the list its
     eval script declares. Our patch side at --patch_scale 0.25, H=512 is 128px,
     so EVERY candidate square is smaller than the patch and coverage has to be
     assembled from a union of 100px squares. It gets there: measured with 70%
     of a 128px patch found, the default list reaches coverage 1.000 at
     precision 0.923. Passing the true 128 side instead is an ORACLE setting —
     it hands the defence the patch geometry — but it is NOT an upper bound and
     must not be reported as one: the same raw mask gives coverage 1.000 at
     precision 0.840, i.e. slightly WORSE, because a union of accepted 128px
     anchors spreads further than a union of 100px ones. Report whichever you
     ran and treat them as two settings, not as a bound and a measurement.
  3. n_patch = 1. SAC's multi-patch path k-means-clusters the raw mask first.
     Patch.apply() composites exactly one footprint, so clustering would only
     add a failure mode. Multi-patch is refused rather than silently ignored.

COMPLETION IS NON-DIFFERENTIABLE. The 0.5 threshold and the Hamming
acceptance both discard gradient. SAC's reference code offers a
straight-through estimator for adaptive attacks (their `bpda=True`); nothing
here does, because every path in this repo is evaluation-only. DefendedSegModel
raises rather than let an attack train through a zero-gradient block.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ..base import PatchDefence, Purified
from .unet import load_sac_unet

#: WHAT THE RELEASED CODE ACTUALLY RUNS, which is not what it appears to.
#: SAC's eval script builds PatchDetector(square_sizes=[125,100,75,50,25])
#: (eval_object_detection_coco.py:101) and PatchDetector.__init__ stores it as
#: self.square_sizes (patch_detector.py:166) — but forward() calls
#: ShapeCompletionL1(mask, soft=...) without passing it (patch_detector.py:194),
#: so the function's own default [100,75,50,25] is what executes, for both the
#: single- and multi-patch paths. Verified against the repo 2026-10-01.
#: We default to the list that runs rather than the list that is declared; pass
#: --sac_square_sizes to override, and say which you used.
SAC_SQUARE_SIZES = (100, 75, 50, 25)
SAC_DECLARED_SQUARE_SIZES = (125, 100, 75, 50, 25)
SAC_INIT_L1_THRESH = 0.9
SAC_THRESH_DECAY = 0.7
SAC_BREAK_ITS = 15


def _integral(x: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] -> [B,1,H+1,W+1] summed-area table, C[i,j] = sum(x[:i,:j])."""
    c = x.double().cumsum(-2).cumsum(-1)
    return F.pad(c, (1, 0, 1, 0))


def _box_sums(x: torch.Tensor, size: int) -> torch.Tensor:
    r"""
    Sum of x over every size x size window, stride 1. [B,1,H-size+1,W-size+1].

    Integral image rather than avg_pool2d: pooling with a 125px kernel over a
    512x1024 frame is O(H*W*size^2) ~ 8e9 multiply-adds per size per call, and
    shape_completion() calls this up to len(square_sizes) * break_its times.
    The summed-area table is O(H*W) regardless of size. float64 because the
    table reaches ~5e5 and the result is a difference of four of its entries.
    """
    c = _integral(x)
    return (c[..., size:, size:] - c[..., :-size, size:]
            - c[..., size:, :-size] + c[..., :-size, :-size])


def _union_of_squares(accept: torch.Tensor, size: int, hw) -> torch.Tensor:
    r"""
    Union of the size x size squares whose top-left corners accept==1.

    accept is [B,1,H-size+1,W-size+1]. A pixel p is covered iff some accepted
    corner t satisfies p-size+1 <= t <= p, so the cover count is a box sum over
    accept padded by (size-1) on every side — O(H*W) again, where a max-pool
    dilation would be O(H*W*size). count >= 1 is the union, which is exactly
    the reference implementation's `ThresholdSTEGE(out_mask / 2)`.

    ANCHOR SET. The reference indexes its cumsums so that squares whose corner
    lies in the first row or column are never candidates; this takes the full
    natural anchor set [0, H-size] x [0, W-size]. The difference is one pixel of
    reach at the top-left border and it is in the defence's favour.
    """
    H, W = hw
    pad = size - 1
    ap = F.pad(accept, (pad, pad, pad, pad))
    cov = _box_sums(ap, size)
    return (cov[..., :H, :W] >= 1.0).to(accept.dtype)


def completion_pass(mask: torch.Tensor, size: int,
                    l1_thresh: float) -> Optional[torch.Tensor]:
    r"""
    One (size, l1_thresh) pass. Returns the union mask, or None if the frame is
    smaller than the square.

    A candidate square at corner (t,l) is accepted when its Hamming distance to
    `mask` is below (1 - l1_thresh) * size^2:

        hamming = (pixels inside the square that mask does NOT claim)
                + (pixels mask claims that fall OUTSIDE the square)
                = size^2 - inside + (total - inside)

    so l1_thresh near 1 is STRICT (accept only near-exact squares) and
    decaying it loosens the search. Note the direction: `thresh_decay < 1`
    makes the defence MORE willing to fire, not less.

    AN EMPTY RAW MASK CAN NEVER PRODUCE A COMPLETION. total = inside = 0 gives
    hamming = size^2, normalised 1.0, and acceptance needs < 1 - l1_thresh,
    which is strictly below 1 for any l1_thresh > 0. So a clean image on which
    the segmenter fires nowhere is returned untouched.

    A NEARLY-empty raw mask is the case to understand, and the threshold has a
    closed form. Rearranging, a square is accepted iff

        l1_thresh < (2 * inside - total) / size^2

    so for n false positives all CLUSTERED inside one candidate square
    (inside = total = n) acceptance needs l1_thresh < n / size^2. The loop's
    loosest threshold is init * decay^(break_its - 1) = 0.9 * 0.7^14 = 0.0061,
    so completion fires iff

        n > 0.0061 * min(square_sizes)^2        (n >= 4 for min size 25)

    and then the final mask is the UNION of every square that was accepted at
    once. SCATTERED false positives never fire, because no single square can
    contain them. The trigger therefore depends on the SMALLEST square in the
    list and not at all on the frame size — verified 2026-10-01 at 512x1024 and
    256x512, which erase the same 2254 absolute pixels.

    WHAT THAT COSTS, measured with the default squares: 3 clustered stray pixels
    erase nothing, 4 erase ~2250 px, and the figure then DECREASES slowly with n
    (more mask outside a square raises its hamming distance, narrowing the
    accepted set). At 512x1024 that is 0.43% of the frame; at 256x512 the same
    hole is 1.7%. Passing a smaller minimum square makes it much worse — with a
    16px square the same two stray pixels take 22% of a 64x64 frame — which is
    the real argument for not tuning --sac_square_sizes downward.

    So the clean-image cost is BIMODAL but modest per event: either the segmenter
    fires nowhere and the defence is free, or it fires on a small cluster and
    loses a ~47px-equivalent blob somewhere it hallucinated. A mean over images
    hides which of the two happened, so report the per-image distribution of
    fp_area_frac and keep the defended CLEAN column.

    One accidental safety property: a SATURATED raw mask is rejected too, so an
    untrained or broken segmenter completes to empty rather than to everything.
    The numerator (2*inside - total) goes negative once the frame is more than
    twice the area of the largest candidate square, which every resolution here
    clears comfortably. It is a GEOMETRIC guard, not an absolute one: a 100px
    square inside a 128x128 frame gives 0.36 and is accepted, so do not rely on
    it at small frame sizes.
    """
    B, _, H, W = mask.shape
    if size > H or size > W:
        return None
    inside = _box_sums(mask, size)
    total = mask.double().sum(dim=(1, 2, 3)).view(B, 1, 1, 1)
    hamming = float(size * size) - inside + (total - inside)
    accept = (hamming < (1.0 - l1_thresh) * size * size).double()
    return _union_of_squares(accept, size, (H, W)).to(mask.dtype)


def shape_completion(mask: torch.Tensor,
                     square_sizes: Sequence[int] = SAC_SQUARE_SIZES,
                     init_l1_thresh: float = SAC_INIT_L1_THRESH,
                     thresh_decay: float = SAC_THRESH_DECAY,
                     break_its: int = SAC_BREAK_ITS) -> torch.Tensor:
    r"""
    SAC's robust shape completion. [B,1,H,W] float 0/1 -> same.

    Union over `square_sizes` at the current threshold; any image still empty
    gets a loosened threshold and another try, up to `break_its` times, after
    which that image's mask stays empty.

    PER IMAGE, NOT PER BATCH. The reference implementation loops over images, so
    the decay count is per image. Batching the loop without masking would let
    one image that completed immediately stop the search for the rest of the
    batch; `still_empty` keeps the semantics identical at any batch size.
    """
    if not torch.is_floating_point(mask):
        mask = mask.float()
    B = mask.shape[0]
    out = torch.zeros_like(mask)
    still_empty = mask.flatten(1).sum(1) > 0          # nothing to find -> skip
    thresh = init_l1_thresh
    for _ in range(max(break_its, 0)):
        if not bool(still_empty.any()):
            break
        parts = []
        for s in square_sizes:
            part = completion_pass(mask, int(s), thresh)
            if part is not None:
                parts.append(part)
        if not parts:
            raise SystemExit(
                f"\nevery --sac_square_sizes entry {tuple(square_sizes)} is "
                f"larger than the {mask.shape[-2]}x{mask.shape[-1]} frame, so "
                f"shape completion can never fire.\n")
        union = (torch.stack(parts).sum(0) >= 1.0).to(mask.dtype)
        found = union.flatten(1).sum(1) > 0
        take = still_empty & found
        if bool(take.any()):
            sel = take.view(B, 1, 1, 1).to(mask.dtype)
            out = out + sel * union
            still_empty = still_empty & ~found
        thresh = thresh * thresh_decay
    return out.clamp_(0.0, 1.0)


class SACDefence(PatchDefence):
    r"""
    Segment-and-Complete as an input-space defence.

    ckpt          SAC's released patch segmenter (see load_sac_unet)
    base_filter   16 for coco_at.pth, 64 for the APRICOT-Mask checkpoint
    image_scale   resize factor before the U-Net, 1.0 = SAC's COCO setting
                  (their APRICOT setting is 0.5). The mask is resized back, so
                  this trades segmenter resolution for matching the scale the
                  segmenter was trained at.
    complete      False runs SEGMENT only, no completion. Keep it True for the
                  method as published; False is the ablation that attributes
                  recovery to the segmenter rather than to the square prior.
    """

    name = "sac"

    def __init__(self, ckpt: str, base_filter: int = 16,
                 square_sizes: Sequence[int] = SAC_SQUARE_SIZES,
                 image_scale: float = 1.0, complete: bool = True,
                 n_patch: int = 1, device="cpu", log=print):
        super().__init__()
        if n_patch != 1:
            raise SystemExit(
                "\nSAC's multi-patch path k-means-clusters the raw mask before "
                "completing each cluster. Patch.apply() composites exactly one "
                "footprint, so n_patch > 1 is refused rather than silently "
                "ignored.\n")
        self.unet = load_sac_unet(ckpt, base_filter, device=device, log=log)
        self.square_sizes = tuple(int(s) for s in square_sizes)
        self.image_scale = float(image_scale)
        self.complete = bool(complete)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x01: torch.Tensor) -> Purified:
        H, W = x01.shape[-2:]
        xin = x01
        if self.image_scale != 1.0:
            h = max(16, int(round(H * self.image_scale / 16)) * 16)
            w = max(16, int(round(W * self.image_scale / 16)) * 16)
            xin = F.interpolate(x01, size=(h, w), mode="bilinear",
                                align_corners=False)
        logits = self.unet(xin)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                                   align_corners=False)
        prob = torch.sigmoid(logits)
        raw = (prob > 0.5).to(x01.dtype)
        mask = (shape_completion(raw, self.square_sizes) if self.complete
                else raw)
        # SAC zeroes the located region. See the module docstring.
        return Purified(x01=x01 * (1.0 - mask), mask=mask, raw_mask=raw,
                        prob=prob)

    def describe(self, log=print) -> None:
        log(f"[def ] sac  squares={self.square_sizes}  "
            f"complete={self.complete}  image_scale={self.image_scale}")
