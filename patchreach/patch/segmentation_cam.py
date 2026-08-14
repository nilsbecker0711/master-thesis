r"""
Segmentation-specific sensitivity / Grad-CAM.

WHY NOT A CLASSIFICATION GRAD-CAM
---------------------------------
Classification Grad-CAM differentiates a single logit y_c. A segmentation model
emits F(x) in R^{CxHxW} — there is no single scalar to differentiate, and the
per-class-channel variant ("sum the target channel") answers a different
question than the one this project asks.

The scalar used here is the EXISTING attack objective, taken in the direction
that STRENGTHENS the attack:

    S_seg = - L_existing(F(x), y)

THE SIGN IS THE WHOLE POINT. Every objective in losses/adversarial.py is
defined so that MINIMISING it strengthens the attack:

    ce      -> -CE(logits, y)          (minimise => CE rises => wrong)
    cospgd  -> -cos.detach() * CE      (same)
    ipatch  -> +CE(logits, target)     (minimise => driven to target)

Grad-CAM's construction assumes a score to be INCREASED. Feeding L directly
would produce the exact inverse map: high where the attack is already winning,
low where there is headroom. Negating once, uniformly, fixes all three.

FEATURE LAYER
-------------
models/wrapper.py exposes only encode_decode, so the internal features are
reached with a forward hook on the backbone rather than by re-implementing the
forward pass. mmseg backbones return a TUPLE of multi-scale maps; `layer`
selects one (-1 = deepest/coarsest, the standard Grad-CAM choice — largest
receptive field, most semantic).

Do not assume a CNN-style [B,K,h,w]. SegFormer's MiT backbone already reshapes
its tokens back to 2-D before returning, so it lands in the 4-D path. A
backbone that returns [B,N,D] tokens is reshaped by FACTORISING N against the
INPUT ASPECT RATIO — sqrt(N) is wrong at 512x1024 (a 2:1 grid is not square)
and would silently produce a transposed, meaningless map.

GRADIENT SEPARATION
-------------------
WrappedSegModel sets requires_grad_(False) on every parameter. With a non-grad
input PyTorch builds NO graph at all, so autograd.grad(S, A) raises. The pass
therefore runs under enable_grad() on a requires_grad_(True) CLONE of the
input. The model is never unfrozen — only activations carry grad — and the
returned map is detached before it can reach the generator.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..data.cityscapes import upsample_to


# ═════════════════════════════════════════════════════════════════════════════
#  Feature-map normalisation
# ═════════════════════════════════════════════════════════════════════════════

def _token_grid(n_tokens: int, H: int, W: int) -> Tuple[int, int]:
    """
    (h, w) for a flat token sequence, factorised against the INPUT aspect.

    sqrt(N) is only correct for square inputs. At 512x1024 the grid is 2:1, so
    a square reshape transposes the map and every downstream conclusion is
    wrong. Candidate strides cover every patch size in common use.
    """
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
    """
    [B,K,h,w] from either a CNN feature map or a [B,N,D] token sequence.

    A leading class token is detected by N-1 factorising when N does not.
    """
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
        h, w = _token_grid(N - 1, H, W)          # drop a leading cls token
        return a[:, 1:].transpose(1, 2).reshape(B, D, h, w)


# ═════════════════════════════════════════════════════════════════════════════
#  The CAM
# ═════════════════════════════════════════════════════════════════════════════

class SegmentationCAM:
    r"""
    M_i in [0,1]^{HxW}, one map per image, DETACHED.

        w_k   = GAP( dS_seg / dA_k )
        M_raw = ReLU( sum_k w_k A_k )
        M     = per-sample min-max normalise( upsample(M_raw) )

    Per-SAMPLE normalisation, not batch-wide: a batch-wide min-max makes every
    map depend on which other images happen to share the batch, which is not
    reproducible at inference where batch composition differs.

    Usage:
        cam = SegmentationCAM(model, adv_loss)
        M, clean_logits = cam(imgs, labels)      # M [B,1,H,W], both detached

    `clean_logits` is returned so the caller does not pay for a second clean
    forward pass — the CAM pass already computed it.
    """

    def __init__(self, model, adv_loss, layer: int = -1,
                 module: str = "backbone", target: str = "pred",
                 eps: float = 1e-8):
        self.model = model
        self.adv_loss = adv_loss
        self.layer = layer
        self.target = target
        self.eps = eps
        self.n_degenerate = 0          # maps that came out constant

        if target not in ("gt", "pred"):
            raise ValueError(f"cam_target must be 'gt' or 'pred', got {target!r}")

        self.module = self._resolve(model, module)
        self._feats: Optional[Sequence[torch.Tensor]] = None
        self._handle = self.module.register_forward_hook(self._hook)

    # ── plumbing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve(model, dotted: str):
        """
        Walk a dotted path from the mmseg segmentor INSIDE WrappedSegModel.

        The wrapper holds the segmentor at `.model`, so 'backbone' resolves to
        model.model.backbone. Raising with the available children beats an
        AttributeError three frames deep.
        """
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

    # ── the map ──────────────────────────────────────────────────────────────

    def __call__(self, imgs: torch.Tensor, labels: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = imgs.shape

        # enable_grad because callers legitimately sit inside no_grad (eval,
        # export). requires_grad_ on the INPUT because every model parameter is
        # frozen — without a grad-requiring leaf, no graph is built and
        # autograd.grad() below raises "element 0 does not require grad".
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
                # Label-free attacker: the model's own argmax as pseudo-labels.
                # Same information --placement semantic already assumes.
                y = lg.argmax(1).detach()

            # SIGN: every objective in adversarial.py is minimised to attack.
            # Grad-CAM wants a score to INCREASE, so negate exactly once.
            score = -self.adv_loss(lg, y, None, None)
            grads = torch.autograd.grad(score, A, retain_graph=False)[0]

        A = as_spatial(A.detach(), H, W)
        G = as_spatial(grads.detach(), H, W)

        weights = G.mean(dim=(2, 3), keepdim=True)              # [B,K,1,1]
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))    # [B,1,h,w]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear",
                            align_corners=False)

        flat = cam.reshape(B, -1)
        lo = flat.min(dim=1).values.view(B, 1, 1, 1)
        hi = flat.max(dim=1).values.view(B, 1, 1, 1)
        span = hi - lo

        # A fully-suppressed map (ReLU killed everything) would normalise to
        # all-zeros, and an argmax over a constant map returns index 0 — i.e.
        # the top-left corner, silently. Emit a constant 0.5 instead and let
        # find_max_response_placement() fall back to centre explicitly.
        degenerate = span <= self.eps
        self.n_degenerate += int(degenerate.sum())
        cam = torch.where(degenerate,
                          torch.full_like(cam, 0.5),
                          (cam - lo) / span.clamp(min=self.eps))
        return cam.detach(), logits.detach()


def build(model, loss_fn: str, target_class: int = 8, layer: int = -1,
          module: str = "backbone", target: str = "pred",
          attack_loss=None) -> SegmentationCAM:
    """
    Convenience constructor.

    `attack_loss` lets the caller pass the SAME callable the training loop
    optimises (ablation F: "different segmentation targets for the sensitivity
    map"). Otherwise one is built from `loss_fn` via losses.adversarial.build,
    so the CAM and the attack share one implementation.
    """
    if attack_loss is None:
        from ..losses import adversarial
        attack_loss = adversarial.build(loss_fn, target_class)
    return SegmentationCAM(model, attack_loss, layer=layer, module=module,
                           target=target)
