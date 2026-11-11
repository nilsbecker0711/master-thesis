"""Segmentation-specific sensitivity / Grad-CAM."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..data.cityscapes import upsample_to


def _token_grid(n_tokens: int, H: int, W: int) -> Tuple[int, int]:
    """(h, w) for a flat token sequence, factorised against the INPUT aspect."""
    for stride in (2, 4, 8, 16, 32, 64):
        h = -(-H // stride)
        w = -(-W // stride)
        if h * w == n_tokens:
            return h, w
    raise ValueError(
        f"cannot factorise {n_tokens} tokens against a {H}x{W} input at any "
        f"stride in (2,4,8,16,32,64). Pass --cam_layer to select a different "
        f"backbone stage, or --cam_module to hook a module that emits a 4-D "
        f"feature map. Guessing a grid here would silently transpose the map.")


def as_spatial(a: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[B,K,h,w] from either a CNN feature map or a [B,N,D] token sequence."""
    if a.dim() == 4:
        return a
    if a.dim() != 3:
        raise ValueError(
            f"feature tensor has shape {tuple(a.shape)}; expected [B,K,h,w] "
            f"(CNN) or [B,N,D] (tokens)")

    B, N, D = a.shape
    try:
        h, w = _token_grid(N, H, W)
        return a.transpose(1, 2).reshape(B, D, h, w)
    except ValueError:
        h, w = _token_grid(N - 1, H, W)
        return a[:, 1:].transpose(1, 2).reshape(B, D, h, w)


class SegmentationCAM:
    """M_i in [0,1]^{HxW}, one map per image, DETACHED."""

    def __init__(self, model, adv_loss, layer: int = -1,
                 module: str = "backbone", target: str = "pred",
                 eps: float = 1e-8):
        self.model = model
        self.adv_loss = adv_loss
        self.layer = layer
        self.target = target
        self.eps = eps
        self.n_degenerate = 0

        if target not in ("gt", "pred"):
            raise ValueError(f"cam_target must be 'gt' or 'pred', got {target!r}")

        from ..models.wrapper import SlidingWindowSegModel
        if isinstance(model, SlidingWindowSegModel):
            raise ValueError(
                "SegmentationCAM cannot run on a SlidingWindowSegModel: the "
                "backbone is called once per window, so there is no "
                "frame-shaped feature map to weight. Build the CAM on the "
                "un-tiled model (the WrappedSegModel at .model) if a "
                "whole-frame map is acceptable alongside tiled inference — "
                "state it as a deviation, the map and the attack then see "
                "different model behaviour — or use --inference whole. For "
                "SegFormer/Swin the two coincide at 1024x1024 and below, "
                "where the crop clamps to a single window.")

        self.module = self._resolve(model, module)
        self._feats: Optional[Sequence[torch.Tensor]] = None
        self._handle = self.module.register_forward_hook(self._hook)

    @staticmethod
    def _resolve(model, dotted: str):
        """Walk a dotted path from the mmseg segmentor INSIDE WrappedSegModel."""
        obj = getattr(model, "model", model)
        for part in dotted.split("."):
            if not hasattr(obj, part):
                avail = [n for n, _ in obj.named_children()]
                raise AttributeError(
                    f"--cam_module {dotted!r}: {type(obj).__name__} has no "
                    f"attribute {part!r}. Available children: {avail}")
            obj = getattr(obj, part)
        return obj

    def _hook(self, _module, _inp, out):
        self._feats = out if isinstance(out, (list, tuple)) else [out]

    def _pick(self) -> torch.Tensor:
        if self._feats is None:
            raise RuntimeError(
                "the CAM hook never fired — the module passed via --cam_module "
                "is not on the encode_decode forward path")
        feats = self._feats
        if not (-len(feats) <= self.layer < len(feats)):
            raise IndexError(
                f"--cam_layer {self.layer} out of range: the hooked module "
                f"returned {len(feats)} feature map(s) with shapes "
                f"{[tuple(f.shape) for f in feats]}")
        return feats[self.layer]

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __call__(self, imgs: torch.Tensor, labels: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = imgs.shape

        with torch.enable_grad():
            x = imgs.detach().clone().requires_grad_(True)
            self._feats = None
            logits = self.model(x)
            A = self._pick()
            if not A.requires_grad:
                raise RuntimeError(
                    "the hooked feature map does not require grad. The model "
                    "was probably called inside torch.no_grad() by something "
                    "wrapping this call, or the hooked module is detached from "
                    "the graph.")

            lg = upsample_to(logits, labels.shape[-2:])
            if self.target == "gt":
                y = labels
            else:
                y = lg.argmax(1).detach()

            score = -self.adv_loss(lg, y, None, None)
            grads = torch.autograd.grad(score, A, retain_graph=False)[0]

        A = as_spatial(A.detach(), H, W)
        G = as_spatial(grads.detach(), H, W)

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(H, W), mode="bilinear",
                            align_corners=False)

        flat = cam.reshape(B, -1)
        lo = flat.min(dim=1).values.view(B, 1, 1, 1)
        hi = flat.max(dim=1).values.view(B, 1, 1, 1)
        span = hi - lo

        degenerate = span <= self.eps
        self.n_degenerate += int(degenerate.sum())
        cam = torch.where(degenerate,
                          torch.full_like(cam, 0.5),
                          (cam - lo) / span.clamp(min=self.eps))
        return cam.detach(), logits.detach()


def build(model, loss_fn: str, target_class: int = 8, layer: int = -1,
          module: str = "backbone", target: str = "pred",
          attack_loss=None, tsallis=None) -> SegmentationCAM:
    """Convenience constructor."""
    if attack_loss is None:
        from ..losses import adversarial
        attack_loss = adversarial.build(loss_fn, target_class,
                                        **(tsallis or {}))
    return SegmentationCAM(model, attack_loss, layer=layer, module=module,
                           target=target)
