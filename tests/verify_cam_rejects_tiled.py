r"""
What SegmentationCAM ACTUALLY did on a tiled model, before it refused to.

    python tests/verify_cam_rejects_tiled.py

The EVIDENCE behind the 2026-10-17 correction in docs/experiment_log.md, kept
runnable because the claim it supports is a negative one — that the old
behaviour was silently wrong rather than loud — and a negative claim nobody can
re-run is just an assertion.

NOT A PYTEST MODULE (the name is deliberate; pytest collects test_*.py). The
REFUSAL is tested in tests/test_iap.py::test_the_cam_refuses_a_tiled_model;
this shows what the refusal is protecting against, which requires slipping past
it.

Two things matter about the setup. The segmentor's encode_decode resizes logits
back to the INPUT size, as every mmseg EncoderDecoder does — without that, the
padding arithmetic in SlidingWindowSegModel.forward is inconsistent and the run
dies with a tensor-size error, which is what an earlier, sloppier stub showed
and what the first version of this finding wrongly reported. And the tiled
module is hand-rolled rather than a SlidingWindowSegModel subclass, purely so
the isinstance guard does not fire; its forward is the same loop, window for
window.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

from patchreach.models.wrapper import WrappedSegModel, slide_windows
from patchreach.patch import segmentation_cam

K = 19
H, W = 64, 256                 # 1:4, so a 64-crop tiles several times across
CROP, STRIDE = (64, 64), (48, 48)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, stride=4, padding=1)
        self.seen = []

    def forward(self, x):
        self.seen.append(tuple(x.shape[-2:]))
        return (self.c(x),)


class Seg(nn.Module):
    """encode_decode returns logits at the INPUT resolution, as mmseg does."""
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Conv2d(8, K, 1)

    def encode_decode(self, x, metas):
        f = self.head(self.backbone(x)[0])
        return F.interpolate(f, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


class HandRolledSlide(nn.Module):
    """SlidingWindowSegModel.forward, copied, minus the class identity."""
    def __init__(self, inner):
        super().__init__()
        self.model = inner

    def forward(self, x):
        B, _, h, w = x.shape
        preds = x.new_zeros((B, K, h, w))
        count = x.new_zeros((1, 1, h, w))
        for y1, y2, x1, x2 in slide_windows(h, w, CROP, STRIDE):
            logit = self.model(x[:, :, y1:y2, x1:x2])
            preds = preds + F.pad(logit, (x1, w - x2, y1, h - y2))
            count[:, :, y1:y2, x1:x2] += 1
        return preds / count


img = torch.rand(1, 3, H, W)
label = torch.randint(0, K, (1, H, W))
wins = list(slide_windows(H, W, CROP, STRIDE))
print(f"frame {H}x{W}, crop {CROP}, stride {STRIDE} -> {len(wins)} windows")
print(f"  last window is x {wins[-1][2]}..{wins[-1][3]}  "
      f"(i.e. the right edge)")

print()
print("=" * 72)
print(" A. whole — the control")
print("=" * 72)
seg = Seg()
whole = WrappedSegModel(seg)
cam = segmentation_cam.build(whole, "ce")
seg.backbone.seen.clear()
m_whole, _ = cam(img, label)
print(f"  backbone called with shapes : {seg.backbone.seen}")
print(f"  hooked feature map shape    : {tuple(cam._pick().shape)}")
print(f"  CAM shape                   : {tuple(m_whole.shape)}")
cam.close()

print()
print("=" * 72)
print(" B. tiled — what used to happen")
print("=" * 72)
seg2 = Seg()
tiled = HandRolledSlide(WrappedSegModel(seg2))
cam2 = segmentation_cam.build(tiled, "ce", module="model.backbone")
seg2.backbone.seen.clear()
try:
    m_tiled, _ = cam2(img, label)
    print(f"  backbone called with shapes : {seg2.backbone.seen}")
    print(f"      -> called {len(seg2.backbone.seen)} times, and the hook "
          f"ASSIGNS rather than appends")
    print(f"  hooked feature map shape    : {tuple(cam2._pick().shape)}  "
          f"<- ONE WINDOW, not the frame")
    print(f"  CAM shape                   : {tuple(m_tiled.shape)}  "
          f"<- full frame anyway")
    print()
    print("  !! NO ERROR. The map is one window's features stretched over the")
    print("     whole frame. A plausible map describing the wrong region.")
    cols = m_tiled.shape[-1]
    print(f"  agreement with the whole-frame map: "
          f"{float((m_tiled - m_whole).abs().mean()):.4f} mean abs diff "
          f"(0 would mean they agree)")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
cam2.close()

print()
print("=" * 72)
print(" C. and with a TOKEN-returning backbone instead of a 4-D one")
print("=" * 72)


class TokenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, stride=4, padding=1)

    def forward(self, x):
        f = self.c(x)
        B, C, h, w = f.shape
        return (f.flatten(2).transpose(1, 2),)      # [B, N, D]


class TokenSeg(Seg):
    def __init__(self):
        super().__init__()
        self.backbone = TokenBackbone()

    def encode_decode(self, x, metas):
        t = self.backbone(x)[0]
        B, N, D = t.shape
        h = -(-x.shape[-2] // 4)
        w = -(-x.shape[-1] // 4)
        f = self.head(t.transpose(1, 2).reshape(B, D, h, w))
        return F.interpolate(f, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


tseg = TokenSeg()
ttiled = HandRolledSlide(WrappedSegModel(tseg))
cam3 = segmentation_cam.build(ttiled, "ce", module="model.backbone")
try:
    cam3(img, label)
    print("  returned a map (silently wrong, as in B)")
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:220]}")
    print("  -> token backbones RAISE, because the window's token count")
    print("     cannot be factorised against the FRAME's dimensions.")
cam3.close()
