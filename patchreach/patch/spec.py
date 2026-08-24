r"""
PatchConfig + Patch — the single object that owns everything about the patch.

WHY THIS EXISTS
---------------
The previous codebase carried REACH_MASK, LAP_REF, LAP_EDGES, SHAPE_MASK and
PLACEMENT as module-level globals consumed inside apply_patch(). That produced
a class of ordering bugs that are invisible until results look wrong — most
recently, placement resolving to centre because apply_patch() ran before the
clean forward pass that semantic placement depends on. Bundling the state onto
one object makes the dependency explicit and the ordering checkable.

PARAMETERISATION — UNIFIED ON SIGMOID
-------------------------------------
The old single-image script treated patch_param as PIXELS with an in-loop
clamp to [0,1]; the training script treated it as LOGITS through a sigmoid.
That split cost two bugs (LAP init, raw_ganinit seeding) and forced every init
path to be written twice.

Here it is always sigmoid:  patch = sigmoid(param).

  * No clamping step, so no dead gradients at the [0,1] boundary. A hard clamp
    zeroes the gradient for any pixel sitting on the bound, which is exactly
    where an adversarial patch wants to live.
  * Every init goes through lap.logit_seed(), one code path.
  * gan mode is unaffected: the latent is not a pixel tensor and still uses a
    hard clip, which is correct there (BigGAN was trained on z~N(0,1); |z|>3
    gives out-of-distribution artefacts).

Checkpoints record their parameterisation, so old ones stay loadable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import csf as csf_mod
from . import lap as lap_mod
from . import placement as placement_mod
from . import shape as shape_mod

# 'conditional' is not selectable from any CLI — add_patch_args() deliberately
# does not list it. It exists so ONE generated patch can be frozen into a Patch
# (conditional_generator.as_patch) and pushed through the existing diagnostic
# suite unchanged. Behaviourally it is identical to 'raw': sigmoid rendering,
# no regularisers, no reference. Only the label differs, so diagnostics.txt
# reports the run's real provenance instead of claiming patch_mode=raw.
MODES = ("raw", "gan", "raw_ganinit", "lap", "conditional", "csf",
         "universal_csf")


@dataclass
class PatchConfig:
    """Everything that defines a patch, independent of the model or data."""
    mode: str = "raw"
    size: int = 128                       # parameter resolution
    scale: float = 0.25                   # side as a fraction of image HEIGHT

    # ── shape (shape.py) ─────────────────────────────────────────────────────
    shape: str = "square"                 # square | alpha | chroma | auto
    reference_fit: str = "auto"     # auto | crop | pad | stretch
    shape_bg: str = "white"
    shape_thresh: float = 0.15

    # ── placement (placement.py) ─────────────────────────────────────────────
    placement: str = "center"             # center | fixed | semantic | gradcam
    placement_class: int = 0              # 0=road: omnipresent in Cityscapes
    placement_xy: Tuple[float, float] = (0.5, 0.5)
    # Border keep-out for gradcam only. 0 leaves every other policy untouched.
    placement_margin: int = 0

    # ── reference image: shared by lap AND shape ─────────────────────────────
    # In the old code the silhouette was derived from --lap_reference, so a
    # shaped patch was only possible in lap mode. One field for both fixes it.
    reference: Optional[str] = None
    init_reference: Optional[torch.Tensor] = None
    # ── LAP (lap.py) ─────────────────────────────────────────────────────────
    lap_alpha: float = 0.0                # 0 = stage 1 (transition patch)
    lap_beta: float = 0.0
    lap_gamma: float = 0.0
    lap_edge_thresh: float = 0.1
    lap_freeze_edges: bool = False

    # ── BigGAN ───────────────────────────────────────────────────────────────
    gan_init: str = "random"              # random | dog
    cls_biggan: int = 259
    latent_clip: float = 2.0
    logit_clip: float = 6.0        # pixel modes; 0 disables

    # ── perceptual constraint (csf.py), mode='csf' only ──────────────────────
    csf_threshold: float = 0.25
    csf_model: str = "barten"
    csf_beta: float = 3.0
    csf_min_cycles: float = 2.0
    csf_pixel_size_cm: float = 0.0114
    csf_viewing_distance_cm: float = 50.0

    # ── universal_csf only ───────────────────────────────────────────────────
    # ONE residual shared across the whole dataset, added onto whatever content
    # it lands on. mode='csf' optimises a fresh residual per image and takes
    # its base from that image; this is the non-adaptive control for it, and
    # the ONLY difference is that delta is shared and the reference luminance
    # is fixed. Everything else -- the 1/CSF envelope, the Minkowski pooling,
    # tau -- is the same machinery.
    csf_display_peak_cd_m2: float = 100.0
    csf_lref: float = 0.0          # 0 = legacy mu=0.5; >0 = that Y as L_ref
    csf_composite: str = "clip"    # clip | fit

    init_from: Optional[str] = None       # checkpoint to seed from (LAP stage 2)

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "lap" and not self.reference and not self.init_from:
            raise ValueError("mode='lap' needs `reference` (or `init_from`)")
        if self.mode == "universal_csf":
            if self.csf_composite not in ("clip", "fit"):
                raise ValueError("csf_composite must be clip|fit, got "
                                 f"{self.csf_composite!r}")
            if self.shape != "square":
                # The residual's spectrum is computed on the full square. A
                # silhouette would make the pasted region a different signal
                # from the one tau was measured on, and the bound would not
                # describe what the observer sees.
                raise ValueError("mode='universal_csf' supports --shape square "
                                 "only; the spectral bound is defined on the "
                                 "full footprint")
        if self.shape != "square" and not self.reference:
            raise ValueError(f"shape={self.shape!r} needs `reference` — the "
                             "silhouette is derived from it")
        return self


class Patch:
    """
    Owns the optimised parameter and every derived artefact.

    Usage:
        patch = Patch(cfg, device, mean_t, std_t, generator=G)
        patch.resolve_placement(H, W, clean_pred)     # before first apply()
        rendered = patch.render()
        patched, footprint = patch.apply(imgs)
    """

    def __init__(self, cfg: PatchConfig, device, mean_t, std_t,
                 generator=None, init_reference =None):
        self.cfg = cfg.validate()
        self.device = device
        self._mean, self._std = mean_t, std_t
        self.G = generator

        self.reference: Optional[torch.Tensor] = None   # [3,S,S] in [0,1]
        self.edges: Optional[torch.Tensor] = None       # [S,S] bool
        self.shape_mask: Optional[torch.Tensor] = None  # [S,S] bool
        self.placement: Optional[Tuple[int, int]] = None
        self.init_reference = init_reference
        self._load_reference()
        # init_reference overrides a file reference. It is how mode='csf' gets
        # the IMAGE CONTENT it will sit on: the perceptual guarantee is about a
        # residual, so the base has to be what the patch actually covers.
        if init_reference is not None:
            self.reference = init_reference.to(self.device).clamp(0, 1)
        self._csf_values = self._csf_budget = None
        self._frac_clipped = 0.0
        self._frac_at_bound = 0.0
        if self.cfg.mode == "universal_csf":
            c = self.cfg
            geom = csf_mod.ViewingGeometry(c.csf_pixel_size_cm,
                                           c.csf_viewing_distance_cm,
                                           c.csf_display_peak_cd_m2)
            self._geometry = geom
            if c.csf_lref > 0:
                self._csf_values, budget = csf_mod.patch_budget_at_luminance(
                    c.size, c.csf_lref, geom, c.csf_model, 1.0,
                    c.csf_min_cycles, units="srgb", device=self.device)
                self._contrast_scale = csf_mod.calibrated_contrast_scale(
                    c.csf_lref)
            else:
                self._csf_values, budget = csf_mod.patch_budget(
                    c.size, geom, c.csf_model, 1.0, c.csf_min_cycles,
                    device=self.device)
                self._contrast_scale = csf_mod.CONTRAST_SCALE
            # Normalise ONCE. The envelope depends only on the configuration,
            # never on the parameter, so recomputing it per step would be pure
            # cost -- and a per-step normalisation would silently make tau a
            # function of the current iterate.
            self._csf_budget = csf_mod.normalise_budget_to_tau(
                budget, self._csf_values, c.csf_threshold, c.csf_beta,
                self._contrast_scale)
        if self.cfg.mode == "csf":
            self._csf_values, self._csf_budget = csf_mod.patch_budget(
                c_size := self.cfg.size,
                csf_mod.ViewingGeometry(self.cfg.csf_pixel_size_cm,
                                        self.cfg.csf_viewing_distance_cm),
                self.cfg.csf_model, self.cfg.csf_threshold,
                self.cfg.csf_min_cycles, device=self.device)
        self.param = self._init_param()

    # ── construction ─────────────────────────────────────────────────────────

    def _load_reference(self):
        c = self.cfg
        
        if c.reference is None:
            return
        rgb, alpha = shape_mod.load_reference_rgba(
            c.reference, c.size, self.device, c.reference_fit)
        self.reference = rgb
        self.edges = lap_mod.edge_mask(rgb, c.lap_edge_thresh)
        if c.shape != "square":
            self.shape_mask = shape_mod.derive_shape_mask(
                rgb, alpha, c.shape, c.shape_bg, c.shape_thresh)

    def _init_param(self) -> torch.Tensor:
        c = self.cfg

        if c.init_from:                                     # LAP stage 2, resume
            ck = torch.load(c.init_from, map_location="cpu")
            return ck["param"].to(self.device).clone().requires_grad_(True)

        if c.mode == "gan":
            self._require_generator()
            # 'dog': small-norm central latent so BigGAN emits a clean, typical
            # sample from step 0. The CLASS is fixed by set_classes(cls_biggan)
            # regardless — this only controls how typical the START is.
            s = 0.1 if c.gan_init == "dog" else 0.5
            return (torch.randn(self.G.dim_z, device=self.device) * s
                    ).requires_grad_(True)

        if c.mode == "raw_ganinit":
            self._require_generator()
            return lap_mod.logit_seed(self._biggan_sample()
                                      ).clone().requires_grad_(True)

        if c.mode == "lap":
            return lap_mod.logit_seed(self.reference).clone().requires_grad_(True)

        if c.mode == "universal_csf":
            # Deliberately LARGE. The projection is invariant to scaling raw
            # up, so a big init starts with essentially every bin at its bound
            # and pooled visibility at exactly tau -- which is what makes a
            # universal run at tau comparable to an overfit run at tau. A small
            # init starts far below the budget and the run has to climb there
            # first, wasting steps and under-reporting tau throughout.
            return (torch.randn(3, c.size, c.size, device=self.device) * 10.0
                    ).requires_grad_(True)

        if c.mode == "csf":
            # NOT zeros. csf_residual normalises to exactly tau, so a zero raw
            # tensor is 0/0 with an undefined gradient. The reparameterisation
            # is scale-invariant, so only the SHAPE of this init matters and
            # its magnitude is irrelevant.
            return torch.randn(3, c.size, c.size,
                               device=self.device).requires_grad_(True)

        # raw: zeros -> sigmoid -> uniform 0.5 grey
        return torch.zeros(3, c.size, c.size,
                           device=self.device).requires_grad_(True)

    def _require_generator(self):
        if self.G is None:
            raise ValueError(f"mode={self.cfg.mode!r} needs a BigGAN generator")

    @torch.no_grad()
    def _biggan_sample(self) -> torch.Tensor:
        z = torch.randn(self.G.dim_z, device=self.device) * 0.3   # central
        img = ((self.G(z.unsqueeze(0)) + 1.0) * 0.5).clamp(0, 1)
        return F.interpolate(img, size=(self.cfg.size, self.cfg.size),
                             mode="bilinear", align_corners=False).squeeze(0)

    # ── rendering ────────────────────────────────────────────────────────────

    def residual(self) -> torch.Tensor:
        """
        universal_csf: param -> the shared residual delta [1,3,S,S].

        In sRGB CODE units, the space the composite happens in. Not a patch --
        it is added to content rather than replacing it, so it is signed and
        has no [0,1] range of its own.
        """
        if self.cfg.mode != "universal_csf":
            raise ValueError(f"residual() is universal_csf only, got "
                             f"{self.cfg.mode!r}")
        d, at_bound = csf_mod.universal_residual(
            self.param.unsqueeze(0), self._csf_budget, self._csf_values,
            self.cfg.csf_threshold, self.cfg.csf_beta, self._contrast_scale)
        self._frac_at_bound = float(at_bound)
        return d

    def render(self) -> torch.Tensor:
        """param -> [3,S,S] in [0,1], differentiable."""
        c = self.cfg
        if c.mode == "universal_csf":
            # FOR VIEWING ONLY -- the residual on mid grey, so a saved PNG
            # shows its structure. apply() never calls this; it composites the
            # residual onto real content. Saving `delta` raw would be a
            # near-black image at any usable tau.
            return (0.5 + self.residual()[0]).clamp(0.0, 1.0)
        if c.mode == "gan":
            return ((self.G(self.param.unsqueeze(0)) + 1.0) * 0.5).squeeze(0)
        if c.mode == "csf":
            # ADDITIVE in linear pixel space, deliberately not through the
            # sigmoid: the spectral budget describes a residual on the base,
            # and sigmoid(logit(r)+d) != r+d would distort exactly the spectrum
            # the budget was computed for.
            base = (self.reference if self.reference is not None
                    else torch.full((3, c.size, c.size), 0.5,
                                    device=self.device))
            d = csf_mod.csf_residual(self.param.unsqueeze(0), self._csf_budget,
                                     self._csf_values, c.csf_threshold,
                                     c.csf_beta, mask=self.shape_mask)
            d = csf_mod.fit_to_range(d, base.unsqueeze(0),
                                     mask=self.shape_mask)
            return (base + d[0]).clamp(0.0, 1.0)

        p = torch.sigmoid(self.param)
        if c.mode == "lap" and c.lap_freeze_edges and self.edges is not None:
            # Pin the reference's strong edges so the outline survives — the
            # Grad term of Tan et al. Eq 5. Interior pixels stay free.
            p = torch.where(self.edges.unsqueeze(0), self.reference, p)
        return p

    # ── placement ────────────────────────────────────────────────────────────

    def resolve_placement(self, H: int, W: int,
                          clean_pred: Optional[torch.Tensor] = None,
                          score_map: Optional[torch.Tensor] = None):
        """
        Must be called before the first apply() when placement != 'center'.

        Semantic placement reads the CLEAN PREDICTION and gradcam placement
        reads the SENSITIVITY MAP, so the caller has to run the clean forward
        pass (and the CAM) first. Ordering:
            clean forward -> resolve_placement -> apply

        score_map : [H,W] float, required for placement='gradcam'. Missing, it
        raises rather than centring silently — see placement.resolve().
        """
        p = int(H * self.cfg.scale)
        self.placement = placement_mod.resolve(
            self.cfg.placement, H, W, p, clean_pred,
            self.cfg.placement_class, self.cfg.placement_xy,
            score_map=score_map, margin=self.cfg.placement_margin)
        return self.placement
   
 
    # ── compositing — the ONE patch applicator ───────────────────────────────

    def apply(self, imgs: torch.Tensor):
        """
        Composite onto a batch. Returns (patched [B,3,H,W], footprint [B,H,W]).

        GRADIENT SAFETY: built with F.pad, never in-place slice assignment.
        In PyTorch 1.11 an in-place slice into a no-grad tensor creates no
        backward node, so param.grad silently stays None and the patch never
        learns. F.pad is a native autograd op, so the chain
            param -> render -> interpolate -> normalise -> pad -> mul/add
        is differentiable end to end.

        FOOTPRINT follows the SILHOUETTE, not the bounding box. That mask drives
        loss exclusion and the remote-mIoU denominator, so with a cutout only
        genuinely occluded pixels are excluded — stricter and more honest than
        excluding the whole square.
        """
        B, _, H, W = imgs.shape
        p = int(H * self.cfg.scale)

        if self.cfg.mode == "universal_csf":
            return self._apply_residual(imgs, p)

        rendered = F.interpolate(self.render().unsqueeze(0), size=(p, p),
                                 mode="bilinear", align_corners=False)
        normed = (rendered - self._mean) / self._std

        top, left = (self.placement if self.placement is not None
                     else ((H - p) // 2, (W - p) // 2))
        top = max(0, min(int(top), H - p))
        left = max(0, min(int(left), W - p))
        pads = (left, W - p - left, top, H - p - top)

        full = F.pad(normed, pads).expand(B, -1, -1, -1)

        if self.shape_mask is None:
            sm = torch.ones(1, 1, p, p, device=imgs.device)
        else:
            sm = F.interpolate(
                self.shape_mask.float().view(1, 1, *self.shape_mask.shape),
                size=(p, p), mode="nearest")
        mask = F.pad(sm, pads)                                  # [1,1,H,W]

        patched = imgs * (1.0 - mask) + full * mask
        footprint = (mask[0, 0] > 0.5).unsqueeze(0).expand(B, -1, -1)
        return patched, footprint

    def _apply_residual(self, imgs: torch.Tensor, p: int):
        r"""
        x'_i = x_i, with the footprint replaced by clip(x_i[footprint] + delta).

        THE ONE STRUCTURAL DIFFERENCE FROM EVERY OTHER MODE. apply() above
        renders a single patch and expands it across the batch, so every image
        receives identical pixels. Here every image receives the SAME RESIDUAL
        on DIFFERENT CONTENT, which is the whole point: delta is universal, the
        content it sits on is not.

        The composite runs in [0,1] sRGB code space and is re-normalised
        afterwards, because that is the space the spectral budget is defined
        in. Doing it in normalised space would apply a per-channel scale to the
        residual and quietly change its spectrum by a factor of std.

        CLIPPING IS TRACKED, NOT ASSUMED AWAY. Where the underlying content is
        near 0 or 1 the clamp truncates the residual, so the realised
        perturbation is not the projected one and the tau guarantee is violated
        in the permissive direction. `frac_clipped` is the fraction of
        footprint pixels where that happened, and it is logged every step.
        --csf_composite fit uses fit_to_range instead, which rescales rather
        than truncates and therefore preserves the spectrum -- at the cost of a
        per-image scale, which is a per-image adaptation a universal patch is
        not supposed to have. clip is the default for that reason.

        GRADIENT: clamp has zero gradient on the saturated side, so a heavily
        clipped run also loses the gradient on those pixels. That is a real
        cost of the honest composite and another reason to watch frac_clipped.
        """
        B, _, H, W = imgs.shape
        top, left = (self.placement if self.placement is not None
                     else ((H - p) // 2, (W - p) // 2))
        top = max(0, min(int(top), H - p))
        left = max(0, min(int(left), W - p))

        delta = self.residual()
        if delta.shape[-1] != p:
            # Resampling a residual RESAMPLES ITS SPECTRUM, so the budget the
            # bins were projected onto no longer describes the pasted signal.
            # Refuse rather than silently invalidate tau.
            raise ValueError(
                f"universal_csf needs the parameter grid to equal the pasted "
                f"footprint: size={self.cfg.size} but int(H*scale)={p}. "
                f"Set --patch_size {p} (or --patch_scale {self.cfg.size/H:g}).")

        img01 = (imgs * self._std + self._mean).clamp(0.0, 1.0)
        win = img01[:, :, top:top + p, left:left + p]

        if self.cfg.csf_composite == "fit":
            d = csf_mod.fit_to_range(delta.expand(B, -1, -1, -1), win)
        else:
            d = delta
        raw_win = win + d
        new_win = raw_win.clamp(0.0, 1.0)
        with torch.no_grad():
            self._frac_clipped = float(
                ((raw_win < 0.0) | (raw_win > 1.0)).float().mean())

        normed = (new_win - self._mean) / self._std
        pads = (left, W - p - left, top, H - p - top)
        full = F.pad(normed, pads)
        mask = F.pad(torch.ones(1, 1, p, p, device=imgs.device), pads)

        patched = imgs * (1.0 - mask) + full * mask
        footprint = (mask[0, 0] > 0.5).unsqueeze(0).expand(B, -1, -1)
        return patched, footprint

    def set_reference_from_image(self, imgs: torch.Tensor, mean_t, std_t):
        """
        Take the base from the image region the patch will cover.

        Must be called AFTER resolve_placement(). This is what makes mode='csf'
        an invisible-residual attack rather than a visible textured square: the
        patch starts as an exact copy of what it replaces, and only the
        CSF-bounded residual is added on top.
        """
        B, _, H, W = imgs.shape
        p = int(H * self.cfg.scale)
        top, left = (self.placement if self.placement is not None
                     else ((H - p) // 2, (W - p) // 2))
        top = max(0, min(int(top), H - p))
        left = max(0, min(int(left), W - p))
        img01 = (imgs[:1] * std_t + mean_t).clamp(0, 1)
        crop = img01[:, :, top:top + p, left:left + p]
        self.reference = F.interpolate(crop, size=(self.cfg.size, self.cfg.size),
                                       mode="bilinear", align_corners=False
                                       )[0].detach()
        return self.reference

    # ── optimisation helpers ─────────────────────────────────────────────────

    def project(self):
        """
        Post-step projection.

        gan   : BigGAN was trained with z~N(0,1); |z|>3 gives out-of-
                distribution artefacts, so the latent is hard-clipped.

        pixel : the logit is clipped too. The sigmoid BOUNDS the pixel value
                but NOT the parameter — nothing stops |param| running to 20+,
                and there sigmoid'(x) < 1e-8, so the gradient vanishes and Adam
                can never bring the pixel back. Observed failure: a single-image
                run reached a 17.5-point remote drop by step 660, then collapsed
                at step 700 to an inert patch whose loss equalled its untrained
                value, and stayed there for 300 more steps.

                +/-6 keeps the full useful pixel range (sigmoid(6) = 0.9975)
                while leaving sigmoid' ~ 2.5e-3 — small, but not zero, so a
                saturated pixel can still recover. Set logit_clip=0 to disable.
        """
        with torch.no_grad():
            if self.cfg.mode == "gan":
                self.param.data.clamp_(-self.cfg.latent_clip,
                                       self.cfg.latent_clip)
            elif self.cfg.mode == "csf":
                pass          # scale-invariant reparameterisation; nothing to clip
            elif self.cfg.mode == "universal_csf":
                # The projection happens inside residual(), on the SPECTRUM,
                # every forward pass -- so the constraint is already applied to
                # what the loss sees rather than repaired afterwards. Clipping
                # the raw parameter here would do nothing: the projection is
                # invariant to scaling it up.
                pass
            elif self.cfg.logit_clip > 0:
                self.param.data.clamp_(-self.cfg.logit_clip,
                                       self.cfg.logit_clip)

    def active_mask(self) -> Optional[torch.Tensor]:
        r"""
        The support of q = M(p) — the pixels that are actually optimised.

        Tan et al. Eq 5:  M(p) = p - Grad(p, theta) - Bg(p)

        and then, explicitly: "M(p) is the first step in the training process to
        manipulate the inputs, so all patches in the following formulas are
        masked by M() and we set q = M(p)."

        So the paper has TWO tensors: the raw parameter p (a full rectangle) and
        the masked patch q. Eqs 7-9 and the pasted patch are all q. Two things
        are removed:

          Grad(p, theta)  the reference's strong edges — FROZEN, so the outline
                          survives optimisation (handled in render()).
          Bg(p)           everything outside the object outline — NEVER PASTED,
                          because their cartoons sit on black backgrounds and
                          the patch is the object, not its bounding box.

        The previous implementation dropped Bg entirely and masked L_rat by the
        edges alone, which is p minus Grad — not q. Two consequences:

          1. Background parameters were pulled toward the reference background
             by a term that can never affect the attack. Wasted optimisation.
          2. With a ~55% silhouette, ~45% of the reported L_rat magnitude came
             from irrelevant pixels, so the alpha suggested by magnitude_report
             was systematically too small.

        Returns None when neither removal applies (square patch, edges not
        frozen), in which case q == p and the losses run over the full
        rectangle — which is the correct behaviour for that configuration.
        """
        c = self.cfg
        m = None
        if self.shape_mask is not None:                 # Bg(p)
            m = self.shape_mask
        if c.lap_freeze_edges and self.edges is not None:   # Grad(p, theta)
            m = (~self.edges) if m is None else (m & (~self.edges))
        return m

    def regularisers(self) -> dict:
        """
        LAP regularisation terms, raw and weighted, all scored on q = M(p).

        Terms with weight 0 are skipped, so stage 1 costs nothing. Returns
        zeros for non-LAP modes.
        """
        z = torch.zeros((), device=self.device)
        out = {"rat": z, "tv": z, "nps": z, "total": z}
        c = self.cfg
        if c.mode != "lap" or self.reference is None:
            return out

        rendered = self.render()
        active = self.active_mask()
        if c.lap_alpha > 0:
            out["rat"] = lap_mod.rationality_loss(rendered, self.reference,
                                                  mask=active)
        if c.lap_beta > 0:
            out["tv"] = lap_mod.tv_loss(rendered, mask=active)
        if c.lap_gamma > 0:
            out["nps"] = lap_mod.nps_loss(rendered, mask=active)
        out["total"] = (c.lap_alpha * out["rat"] + c.lap_beta * out["tv"]
                        + c.lap_gamma * out["nps"])
        return out

    @torch.no_grad()
    def stats(self) -> dict:
        """Per-step logging. gan tracks the latent; pixel modes track spread."""
        if self.cfg.mode == "gan":
            return {"latent_absmax": self.param.abs().max().item()}
        if self.cfg.mode == "universal_csf":
            d = self.residual()
            return {"visibility": float(csf_mod.visibility_index(
                        d, self._csf_values, beta=self.cfg.csf_beta,
                        contrast_scale=self._contrast_scale)),
                    # climbing toward 1 means every bin sits on its bound and
                    # the run is only re-phasing, not reallocating
                    "frac_at_bound": self._frac_at_bound,
                    # non-zero means the tau guarantee is being violated in the
                    # permissive direction on real content
                    "frac_clipped": self._frac_clipped,
                    "resid_rms": float(d.pow(2).mean().sqrt()),
                    "resid_absmax": float(d.abs().max())}

        if self.cfg.mode == "csf":
            rendered = self.render()
            base = (self.reference if self.reference is not None
                    else torch.full_like(rendered, 0.5))
            d = rendered - base
            act = self.shape_mask if self.shape_mask is not None else None
            dd = d[:, act] if act is not None else d
            # BOTH conventions. tau is enforced under the mu=0.5 assumption,
            # but Cityscapes windows measure mu ~ 0.25, so the locally-measured
            # visibility is roughly TWICE the nominal one. Reporting only the
            # first would hide a systematic 2x under-statement.
            vis_local = float(csf_mod.visibility_index(
                csf_mod._masked(d.unsqueeze(0), self.shape_mask),
                self._csf_values, beta=self.cfg.csf_beta,
                contrast_scale=csf_mod.local_contrast_scale(
                    base.unsqueeze(0), self.shape_mask)))
            return {"visibility": float(csf_mod.realised_visibility(
                        rendered.unsqueeze(0), base.unsqueeze(0),
                        self._csf_values, self.cfg.csf_beta,
                        mask=self.shape_mask)),
                    "visibility_local": vis_local,
                    "resid_rms": float(dd.pow(2).mean().sqrt()),
                    "resid_absmax": float(dd.abs().max())}
        px = torch.sigmoid(self.param)
        lim = self.cfg.logit_clip if self.cfg.logit_clip > 0 else 12.0
        return {"pixel_std": px.std().item(),
                # fraction sitting at the logit bound — the early warning for
                # the saturation collapse described in project()
                "frac_at_clip": (self.param.abs() >= lim * 0.99
                                 ).float().mean().item(),
                "logit_absmax": self.param.abs().max().item()}

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"param": self.param.detach().cpu(),
                    "config": asdict(self.cfg),
                    "parameterisation": ("latent" if self.cfg.mode == "gan"
                                         else "sigmoid"),
                    "placement": self.placement}, path)

    @classmethod
    def load(cls, path, device, mean_t, std_t, generator=None):
        ck = torch.load(path, map_location="cpu")
        cfg = PatchConfig(**ck["config"])
        cfg.init_from = None                     # param comes from the file
        obj = cls(cfg, device, mean_t, std_t, generator)
        obj.param = ck["param"].to(device).clone().requires_grad_(True)
        obj.placement = ck.get("placement")
        return obj

    def describe(self, H: int, W: int, log=print):
        c = self.cfg
        p = int(H * c.scale)
        n = c.size ** 2 * 3 if c.mode != "gan" else (
            self.G.dim_z if self.G else "?")
        log(f"[patch] mode      : {c.mode}  ({n} free parameters)")
        log(f"[patch] size      : {c.size}px param -> {p}px rendered "
            f"(scale {c.scale})")
        if self.shape_mask is not None:
            frac = self.shape_mask.float().mean().item()
            log(f"[patch] silhouette: {int(self.shape_mask.sum()):,} px "
                f"({100*frac:.1f}% of the bounding box) — adversarial surface "
                f"is {100*frac:.0f}% of the square baseline")
        log(f"[patch] placement : {c.placement}"
            + (f" on class {c.placement_class}" if c.placement == "semantic"
               else "")
            + (f" (margin {c.placement_margin}px)"
               if c.placement == "gradcam" and c.placement_margin else ""))
        if self.placement is not None:
            top, left = self.placement
            ctop, cleft = (H - p) // 2, (W - p) // 2
            d = ((top - ctop) ** 2 + (left - cleft) ** 2) ** 0.5
            log(f"[patch] top-left  : ({top}, {left})  {d:.0f}px from centre")
        if c.mode == "csf":
            st = self.stats()
            log(f"[patch] CSF       : {c.csf_model} tau={c.csf_threshold:g} "
                f"requested -> {st['visibility']:.4f} realised   "
                f"(residual rms {st['resid_rms']:.5f}, "
                f"max {st['resid_absmax']:.4f})")
            frac = st["visibility"] / max(c.csf_threshold, 1e-12)
            if frac < 0.5:
                # The failure this exists to prevent: a five-epoch run whose
                # residual had been scaled to EXACTLY ZERO, so the patch was
                # the unmodified reference image and nothing was learned. It
                # was invisible in the logs because only the final metrics
                # were read. Say it once, loudly, before training starts.
                log(f"[patch] WARNING   : the residual was shrunk to "
                    f"{100*frac:.1f}% of the requested budget — "
                    f"fit_to_range found no headroom.")
                log("          The base has saturated pixels where the "
                    "residual cannot fit without clipping.")
                log("          A cutout PNG under --shape square keeps its "
                    "TRANSPARENT PADDING as pure black,")
                log("          which has zero headroom and drives this to 0. "
                    "Use --shape alpha, or a")
                log("          reference without large saturated regions, or "
                    "lower --csf_threshold.")

        if c.mode == "lap":
            log(f"[patch] LAP       : alpha={c.lap_alpha:g} beta={c.lap_beta:g} "
                f"gamma={c.lap_gamma:g}  "
                f"{'stage 2' if c.lap_alpha > 0 else 'stage 1 (transition)'}")
            act = self.active_mask()
            if act is None:
                log(f"[patch] q = M(p)  : FULL RECTANGLE — no Bg (square shape) "
                    f"and no Grad (edges not frozen). Every parameter is free, "
                    f"so this is `raw` with a reference-image init.")
            else:
                log(f"[patch] q = M(p)  : {int(act.sum()):,} / {act.numel():,} px "
                    f"({100*act.float().mean():.1f}%) optimised & regularised"
                    + (f"; {int(self.edges.sum()):,} edge px frozen"
                       if c.lap_freeze_edges and self.edges is not None else ""))