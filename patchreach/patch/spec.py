"""PatchConfig + Patch — the single object that owns everything about the patch."""
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

MODES = ("raw", "gan", "raw_ganinit", "lap", "conditional", "csf",
         "universal_csf")


@dataclass
class PatchConfig:
    """Everything that defines a patch, independent of the model or data."""
    mode: str = "raw"
    size: int = 128
    scale: float = 0.25
    scale_ref: str = "height"

    shape: str = "square"
    reference_fit: str = "auto"
    shape_bg: str = "white"
    shape_thresh: float = 0.15

    placement: str = "center"
    placement_class: int = 0
    placement_xy: Tuple[float, float] = (0.5, 0.5)
    placement_margin: int = 0

    reference: Optional[str] = None
    init_reference: Optional[torch.Tensor] = None
    lap_alpha: float = 0.0
    lap_beta: float = 0.0
    lap_gamma: float = 0.0
    lap_edge_thresh: float = 0.1
    lap_freeze_edges: bool = False

    gan_init: str = "random"
    cls_biggan: int = 259
    latent_clip: float = 2.0
    logit_clip: float = 6.0

    pixel_param: str = "sigmoid"

    raw_init: str = "grey"

    csf_threshold: float = 0.25
    csf_model: str = "barten"
    csf_beta: float = 3.0
    csf_min_cycles: float = 2.0
    csf_pixel_size_cm: float = 0.0114
    csf_viewing_distance_cm: float = 50.0

    csf_max_cpp: float = 0.0

    csf_display_peak_cd_m2: float = 100.0
    csf_lref: float = 0.0
    csf_composite: str = "clip"

    csf_enforce: str = "nominal"

    csf_param: str = "pgd"

    tsallis_q: float = 0.0
    tsallis_schedule: str = "const"
    tsallis_q_start: float = -2.0
    tsallis_q_end: float = 1.0

    init_from: Optional[str] = None

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.scale_ref not in placement_mod.SCALE_REFS:
            raise ValueError(f"scale_ref must be one of "
                             f"{placement_mod.SCALE_REFS}, got {self.scale_ref!r}")
        if self.pixel_param not in ("sigmoid", "direct"):
            raise ValueError("pixel_param must be sigmoid|direct, got "
                             f"{self.pixel_param!r}")
        if self.raw_init not in ("grey", "random"):
            raise ValueError("raw_init must be grey|random, got "
                             f"{self.raw_init!r}")
        if self.raw_init == "random" and self.mode not in ("raw",):
            raise ValueError("raw_init='random' supports --patch_mode raw "
                             f"only, got mode={self.mode!r}. csf and "
                             "universal_csf already start from a random "
                             "parameter projected onto the CSF envelope.")
        if self.pixel_param == "direct" and self.mode != "raw":
            raise ValueError("pixel_param='direct' supports "
                             "--patch_mode raw only, got "
                             f"mode={self.mode!r}")
        if self.mode == "lap" and not self.reference and not self.init_from:
            raise ValueError("mode='lap' needs `reference` (or `init_from`)")
        if self.mode in ("csf", "universal_csf"):
            if self.csf_enforce not in ("nominal", "realised"):
                raise ValueError("csf_enforce must be nominal|realised, got "
                                 f"{self.csf_enforce!r}")
            if not 0.0 <= self.csf_max_cpp <= 0.7072:
                raise ValueError(
                    "csf_max_cpp must be in [0, 0.7072] cycles/pixel (0 = no "
                    f"high cut), got {self.csf_max_cpp!r}")
            if 0.0 < self.csf_max_cpp <= self.csf_min_cycles / self.size:
                raise ValueError(
                    f"csf_max_cpp={self.csf_max_cpp:g} is at or below the low "
                    f"cut csf_min_cycles/size = "
                    f"{self.csf_min_cycles / self.size:g}: the band is empty.")
        if self.mode == "csf":
            if self.csf_param not in ("pgd", "squash"):
                raise ValueError("csf_param must be pgd|squash, got "
                                 f"{self.csf_param!r}")
            if self.csf_param == "pgd" and self.shape != "square":
                raise ValueError(
                    "csf_param='pgd' supports --shape square only; the "
                    "spectral bound is defined on the full footprint and a "
                    "silhouette moves energy between bins, overshooting tau. "
                    "Use --csf_param squash for shaped patches.")
        if self.mode == "universal_csf":
            if self.csf_composite not in ("clip", "fit"):
                raise ValueError("csf_composite must be clip|fit, got "
                                 f"{self.csf_composite!r}")
            if self.shape != "square":
                raise ValueError("mode='universal_csf' supports --shape square "
                                 "only; the spectral bound is defined on the "
                                 "full footprint")
        if self.shape != "square" and not self.reference:
            raise ValueError(f"shape={self.shape!r} needs `reference` — the "
                             "silhouette is derived from it")
        from ..losses.tsallis import validate_q
        validate_q(self.tsallis_q, self.tsallis_schedule,
                   self.tsallis_q_start, self.tsallis_q_end)
        return self


class Patch:
    """Owns the optimised parameter and every derived artefact."""

    def __init__(self, cfg: PatchConfig, device, mean_t, std_t,
                 generator=None, init_reference =None):
        self.cfg = cfg.validate()
        self.device = device
        self._mean, self._std = mean_t, std_t
        self.G = generator

        self.reference: Optional[torch.Tensor] = None
        self.edges: Optional[torch.Tensor] = None
        self.shape_mask: Optional[torch.Tensor] = None
        self.placement: Optional[Tuple[int, int]] = None
        self.init_reference = init_reference
        self._load_reference()
        if init_reference is not None:
            self.reference = init_reference.to(self.device).clamp(0, 1)
        self._csf_values = self._csf_budget = None
        self._frac_clipped = 0.0
        self._frac_at_bound = 0.0
        self._realised_vis = self._realised_vis_max = 0.0
        if self.cfg.mode == "universal_csf":
            c = self.cfg
            geom = csf_mod.ViewingGeometry(c.csf_pixel_size_cm,
                                           c.csf_viewing_distance_cm,
                                           c.csf_display_peak_cd_m2)
            self._geometry = geom
            if c.csf_lref > 0:
                self._csf_values, budget = csf_mod.patch_budget_at_luminance(
                    c.size, c.csf_lref, geom, c.csf_model, 1.0,
                    c.csf_min_cycles, units="srgb", device=self.device,
                    max_cpp=c.csf_max_cpp)
                self._contrast_scale = csf_mod.calibrated_contrast_scale(
                    c.csf_lref)
            else:
                self._csf_values, budget = csf_mod.patch_budget(
                    c.size, geom, c.csf_model, 1.0, c.csf_min_cycles,
                    device=self.device, max_cpp=c.csf_max_cpp)
                self._contrast_scale = csf_mod.CONTRAST_SCALE
            self._csf_budget = csf_mod.normalise_budget_to_tau(
                budget, self._csf_values, c.csf_threshold, c.csf_beta,
                self._contrast_scale)
        if self.cfg.mode == "csf":
            self._contrast_scale = csf_mod.CONTRAST_SCALE
            tau_here = (1.0 if self.cfg.csf_param == "pgd"
                        else self.cfg.csf_threshold)
            self._csf_values, budget = csf_mod.patch_budget(
                self.cfg.size,
                csf_mod.ViewingGeometry(self.cfg.csf_pixel_size_cm,
                                        self.cfg.csf_viewing_distance_cm),
                self.cfg.csf_model, tau_here,
                self.cfg.csf_min_cycles, device=self.device,
                max_cpp=self.cfg.csf_max_cpp)
            self._csf_budget = (
                csf_mod.normalise_budget_to_tau(
                    budget, self._csf_values, self.cfg.csf_threshold,
                    self.cfg.csf_beta, self._contrast_scale)
                if self.cfg.csf_param == "pgd" else budget)
        self.param = self._init_param()
        if self.cfg.mode == "universal_csf" or (
                self.cfg.mode == "csf" and self.cfg.csf_param == "pgd"):
            self._project_spectrum()

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

        if c.init_from:
            ck = torch.load(c.init_from, map_location="cpu")
            return ck["param"].to(self.device).clone().requires_grad_(True)

        if c.mode == "gan":
            self._require_generator()
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
            return (torch.randn(3, c.size, c.size, device=self.device) * 10.0
                    ).requires_grad_(True)

        if c.mode == "csf":
            if c.csf_param == "pgd":
                return (torch.randn(3, c.size, c.size, device=self.device)
                        * 10.0).requires_grad_(True)
            return torch.randn(3, c.size, c.size,
                               device=self.device).requires_grad_(True)

        pixels = (torch.rand(3, c.size, c.size, device=self.device)
                  if c.raw_init == "random"
                  else torch.full((3, c.size, c.size), 0.5,
                                  device=self.device))
        if c.pixel_param == "direct":
            return pixels.clone().requires_grad_(True)
        return lap_mod.logit_seed(pixels).clone().requires_grad_(True)

    def _require_generator(self):
        if self.G is None:
            raise ValueError(f"mode={self.cfg.mode!r} needs a BigGAN generator")

    @torch.no_grad()
    def _biggan_sample(self) -> torch.Tensor:
        z = torch.randn(self.G.dim_z, device=self.device) * 0.3
        img = ((self.G(z.unsqueeze(0)) + 1.0) * 0.5).clamp(0, 1)
        return F.interpolate(img, size=(self.cfg.size, self.cfg.size),
                             mode="bilinear", align_corners=False).squeeze(0)

    def residual(self) -> torch.Tensor:
        """universal_csf: param -> the shared residual delta [1,3,S,S]."""
        if self.cfg.mode != "universal_csf":
            raise ValueError(f"residual() is universal_csf only, got "
                             f"{self.cfg.mode!r}")
        return self.param.unsqueeze(0)

    def render(self) -> torch.Tensor:
        """param -> [3,S,S] in [0,1], differentiable."""
        c = self.cfg
        if c.mode == "universal_csf":
            return (0.5 + self.residual()[0]).clamp(0.0, 1.0)
        if c.mode == "gan":
            return ((self.G(self.param.unsqueeze(0)) + 1.0) * 0.5).squeeze(0)
        if c.mode == "csf":
            base = (self.reference if self.reference is not None
                    else torch.full((3, c.size, c.size), 0.5,
                                    device=self.device))
            if c.csf_param == "pgd":
                d = self.param.unsqueeze(0)
            else:
                d = csf_mod.csf_residual(
                    self.param.unsqueeze(0), self._csf_budget,
                    self._csf_values, c.csf_threshold,
                    c.csf_beta, mask=self.shape_mask)
            d = csf_mod.fit_to_range(d, base.unsqueeze(0),
                                     mask=self.shape_mask)
            if c.csf_enforce == "realised":
                d = csf_mod.fit_to_visibility(
                    d, base.unsqueeze(0), self._csf_values, c.csf_threshold,
                    c.csf_beta, mask=self.shape_mask)
            return (base + d[0]).clamp(0.0, 1.0)

        if c.pixel_param == "direct":
            p = self.param
        else:
            p = torch.sigmoid(self.param)
        if c.mode == "lap" and c.lap_freeze_edges and self.edges is not None:
            p = torch.where(self.edges.unsqueeze(0), self.reference, p)
        return p

    def side(self, H: int, W: int) -> int:
        """Pasted side in px at an H x W input."""
        return placement_mod.footprint_side(H, W, self.cfg.scale,
                                            self.cfg.scale_ref)

    def resolve_placement(self, H: int, W: int,
                          clean_pred: Optional[torch.Tensor] = None,
                          score_map: Optional[torch.Tensor] = None):
        """Must be called before the first apply() when placement != 'center'."""
        p = self.side(H, W)
        self.placement = placement_mod.resolve(
            self.cfg.placement, H, W, p, clean_pred,
            self.cfg.placement_class, self.cfg.placement_xy,
            score_map=score_map, margin=self.cfg.placement_margin)
        return self.placement
   
    def apply(self, imgs: torch.Tensor):
        """Composite onto a batch."""
        B, _, H, W = imgs.shape
        p = self.side(H, W)

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
        mask = F.pad(sm, pads)

        patched = imgs * (1.0 - mask) + full * mask
        footprint = (mask[0, 0] > 0.5).unsqueeze(0).expand(B, -1, -1)
        return patched, footprint

    def _apply_residual(self, imgs: torch.Tensor, p: int):
        """x'_i = x_i, with the footprint replaced by clip(x_i[footprint] + delta)."""
        B, _, H, W = imgs.shape
        top, left = (self.placement if self.placement is not None
                     else ((H - p) // 2, (W - p) // 2))
        top = max(0, min(int(top), H - p))
        left = max(0, min(int(left), W - p))

        delta = self.residual()
        if delta.shape[-1] != p:
            ref_len = H if self.cfg.scale_ref == "height" else (H * W) ** 0.5
            raise ValueError(
                f"universal_csf needs the parameter grid to equal the pasted "
                f"footprint: size={self.cfg.size} but the {self.cfg.scale_ref} "
                f"rule gives p={p} at {H}x{W}. Set --patch_size {p} "
                f"(or --patch_scale {self.cfg.size / ref_len:g}).")

        img01 = (imgs * self._std + self._mean).clamp(0.0, 1.0)
        win = img01[:, :, top:top + p, left:left + p]

        if self.cfg.csf_composite == "fit":
            d = csf_mod.fit_to_range(delta.expand(B, -1, -1, -1), win)
        else:
            d = delta
        if self.cfg.csf_enforce == "realised":
            d = csf_mod.fit_to_visibility(
                d, win, self._csf_values, self.cfg.csf_threshold,
                self.cfg.csf_beta, contrast_scale=self._contrast_scale,
                shared_scale=True)
        raw_win = win + d
        new_win = raw_win.clamp(0.0, 1.0)
        with torch.no_grad():
            self._frac_clipped = float(
                ((raw_win < 0.0) | (raw_win > 1.0)).float().mean())
            rv = csf_mod.visibility_index(
                new_win - win, self._csf_values, beta=self.cfg.csf_beta,
                contrast_scale=self._contrast_scale)
            self._realised_vis = float(rv.mean())
            self._realised_vis_max = float(rv.max())

        normed = (new_win - self._mean) / self._std
        pads = (left, W - p - left, top, H - p - top)
        full = F.pad(normed, pads)
        mask = F.pad(torch.ones(1, 1, p, p, device=imgs.device), pads)

        patched = imgs * (1.0 - mask) + full * mask
        footprint = (mask[0, 0] > 0.5).unsqueeze(0).expand(B, -1, -1)
        return patched, footprint

    def set_reference_from_image(self, imgs: torch.Tensor, mean_t, std_t):
        """Take the base from the image region the patch will cover."""
        B, _, H, W = imgs.shape
        p = self.side(H, W)
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

    def project(self):
        """Post-step projection."""
        with torch.no_grad():
            if self.cfg.mode == "gan":
                self.param.data.clamp_(-self.cfg.latent_clip,
                                       self.cfg.latent_clip)
            elif self.cfg.mode == "csf":
                if self.cfg.csf_param == "pgd":
                    self._project_spectrum()
            elif self.cfg.mode == "universal_csf":
                self._project_spectrum()
            elif self.cfg.pixel_param == "direct":
                self.param.data.clamp_(0.0, 1.0)
            elif self.cfg.logit_clip > 0:
                self.param.data.clamp_(-self.cfg.logit_clip,
                                       self.cfg.logit_clip)

    @torch.no_grad()
    def _project_spectrum(self, eps: float = 1e-12):
        """Clamp every rfft bin of `param` to its budget, leaving phase free."""
        spec = torch.fft.rfft2(self.param.data, norm="forward")
        mag = spec.abs()
        scale = (self._csf_budget / mag.clamp(min=eps)).clamp(max=1.0)
        live = (self._csf_budget > 0).expand_as(scale)
        self._frac_at_bound = float(
            ((scale < 1.0) & live).float().sum()
            / live.float().sum().clamp(min=1.0))
        self.param.data = torch.fft.irfft2(
            spec * scale, s=self.param.shape[-2:], norm="forward")

    @torch.no_grad()
    def _spectral_occupancy(self, eps: float = 1e-12) -> Tuple[float, float]:
        """How much of the per-bin budget the current residual actually spends."""
        spec = torch.fft.rfft2(self.param.detach(), norm="forward")
        mag = spec.abs()
        if self.cfg.csf_param == "pgd":
            ratio = (mag / self._csf_budget.clamp(min=eps)).clamp(max=1.0)
        else:
            ratio = mag / (1.0 + mag.pow(2)).sqrt()
        live = (self._csf_budget > 0).expand_as(ratio)
        n = live.sum().clamp(min=1)
        return (float(((ratio >= 0.99) & live).sum() / n),
                float((ratio * live).sum() / n))

    def active_mask(self) -> Optional[torch.Tensor]:
        """The support of q = M(p) — the pixels that are actually optimised."""
        c = self.cfg
        m = None
        if self.shape_mask is not None:
            m = self.shape_mask
        if c.lap_freeze_edges and self.edges is not None:
            m = (~self.edges) if m is None else (m & (~self.edges))
        return m

    def regularisers(self) -> dict:
        """LAP regularisation terms, raw and weighted, all scored on q = M(p)."""
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
                    "frac_at_bound": self._frac_at_bound,
                    "frac_clipped": self._frac_clipped,
                    "realised_visibility": self._realised_vis,
                    "realised_visibility_max": self._realised_vis_max,
                    "resid_rms": float(d.pow(2).mean().sqrt()),
                    "resid_absmax": float(d.abs().max())}

        if self.cfg.mode == "csf":
            rendered = self.render()
            base = (self.reference if self.reference is not None
                    else torch.full_like(rendered, 0.5))
            d = rendered - base
            act = self.shape_mask if self.shape_mask is not None else None
            dd = d[:, act] if act is not None else d
            vis_local = float(csf_mod.visibility_index(
                csf_mod._masked(d.unsqueeze(0), self.shape_mask),
                self._csf_values, beta=self.cfg.csf_beta,
                contrast_scale=csf_mod.local_contrast_scale(
                    base.unsqueeze(0), self.shape_mask)))
            frac_at_bound, spend_mean = self._spectral_occupancy()
            return {"visibility": float(csf_mod.realised_visibility(
                        rendered.unsqueeze(0), base.unsqueeze(0),
                        self._csf_values, self.cfg.csf_beta,
                        mask=self.shape_mask)),
                    "visibility_local": vis_local,
                    "resid_rms": float(dd.pow(2).mean().sqrt()),
                    "resid_absmax": float(dd.abs().max()),
                    "frac_at_bound": frac_at_bound,
                    "spend_mean": spend_mean}
        if self.cfg.pixel_param == "direct":
            px = self.param
            return {"pixel_std": px.std().item(),
                    "frac_at_clip": ((px <= 0.0) | (px >= 1.0)
                                     ).float().mean().item(),
                    "param_absmax": px.abs().max().item()}
        px = torch.sigmoid(self.param)
        lim = self.cfg.logit_clip if self.cfg.logit_clip > 0 else 12.0
        return {"pixel_std": px.std().item(),
                "frac_at_clip": (self.param.abs() >= lim * 0.99
                                 ).float().mean().item(),
                "logit_absmax": self.param.abs().max().item()}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"param": self.param.detach().cpu(),
                    "config": asdict(self.cfg),
                    "parameterisation": ("latent" if self.cfg.mode == "gan"
                                         else "pixel" if
                                         self.cfg.pixel_param == "direct"
                                         else "sigmoid"),
                    "placement": self.placement}, path)

    @classmethod
    def load(cls, path, device, mean_t, std_t, generator=None):
        ck = torch.load(path, map_location="cpu")
        saved = dict(ck["config"])
        if saved.get("mode") == "csf" and "csf_param" not in saved:
            saved["csf_param"] = "squash"
        saved.setdefault("scale_ref", "height")
        cfg = PatchConfig(**saved)
        cfg.init_from = None
        obj = cls(cfg, device, mean_t, std_t, generator)
        obj.param = ck["param"].to(device).clone().requires_grad_(True)
        obj.placement = ck.get("placement")
        return obj

    def param_from_png(self, path) -> float:
        """Re-parameterise this patch from the PIXELS OF A PNG — the rendered patch read back off disk, as `best_patch.png` stores it."""
        from PIL import Image
        import numpy as np

        if self.cfg.mode != "raw":
            raise ValueError(
                f"param_from_png needs mode='raw', got {self.cfg.mode!r}. "
                "A PNG only round-trips when it is the whole patch; for "
                "'csf'/'universal_csf' it is not, and for 'lap'/'gan'/"
                "'raw_ganinit' this path is simply not covered yet.")

        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        px = torch.from_numpy(arr / 255.0).permute(2, 0, 1).to(self.device)
        if px.shape[-2:] != (self.cfg.size, self.cfg.size):
            raise ValueError(
                f"{Path(path).name} is {tuple(px.shape[-2:])}, but the "
                f"checkpoint's parameter is {self.cfg.size}x{self.cfg.size}. "
                "optimise.py writes the PNG at the parameter resolution, so a "
                "mismatch means this PNG belongs to a different run — "
                "resizing it here would silently change the patch.")

        with torch.no_grad():
            before = self.render().detach()
        self.cfg.pixel_param = "direct"
        self.param = px.clone().requires_grad_(True)
        with torch.no_grad():
            delta = float((self.render().detach() - before).abs().max())
        return delta

    @classmethod
    def from_png(cls, path, device, mean_t, std_t, img_h: int, scale=None):
        """Build a patch from A PNG ALONE, with no checkpoint."""
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        px = torch.from_numpy(arr / 255.0).permute(2, 0, 1)
        h, w = px.shape[-2:]
        if h != w:
            raise ValueError(
                f"{Path(path).name} is {w}x{h}, not square. This path builds "
                "a square patch pasted at the centre; a cutout needs its "
                "silhouette, which an RGB render does not carry, and a "
                "figure-style PNG (axes, two panels) is not a patch at all.")

        cfg = PatchConfig(mode="raw", size=h,
                          scale=(h / img_h) if scale is None else float(scale),
                          scale_ref="height", pixel_param="direct",
                          placement="center")
        obj = cls(cfg, device, mean_t, std_t)
        obj.param = px.to(device).clone().requires_grad_(True)
        return obj

    def describe(self, H: int, W: int, log=print):
        c = self.cfg
        p = self.side(H, W)
        n = c.size ** 2 * 3 if c.mode != "gan" else (
            self.G.dim_z if self.G else "?")
        log(f"[patch] mode      : {c.mode}  ({n} free parameters)")
        log(f"[patch] size      : {c.size}px param -> {p}px rendered "
            f"(scale {c.scale} of {c.scale_ref}; "
            f"{100 * p * p / (H * W):.2f}% of the {H}x{W} frame)")
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
        if c.mode in ("csf", "universal_csf") and self._csf_budget is not None:
            live = self._csf_budget > 0
            f = csf_mod.radial_frequency_cpp(c.size, c.size,
                                             self._csf_budget.device)
            lo, hi = float(f[live].min()), float(f[live].max())
            log(f"[patch] band      : {lo:.4f}-{hi:.4f} c/px "
                f"({int(live.sum()):,} live bins"
                + (f", high cut {c.csf_max_cpp:g}" if c.csf_max_cpp > 0
                   else ", no high cut") + ")")
        if c.mode == "csf":
            st = self.stats()
            rel = "==" if c.csf_param == "squash" else "<="
            log(f"[patch] CSF       : {c.csf_model} tau={c.csf_threshold:g} "
                f"requested -> {st['visibility']:.4f} realised   "
                f"(residual rms {st['resid_rms']:.5f}, "
                f"max {st['resid_absmax']:.4f})")
            log(f"[patch] CSF param : {c.csf_param}  "
                f"(tau {rel} requested; bins at bound "
                f"{100*st['frac_at_bound']:.1f}%, mean spend "
                f"{st['spend_mean']:.3f})")
            if c.csf_param == "squash":
                log("          NOTE: under the squash the magnitude spectrum "
                    "stops being a free variable from")
                log("          EITHER end — saturation (spend -> 1, the "
                    "squash flattens) or scale-invariance")
                log("          (spend -> 0, the rescale to exactly tau makes "
                    "||param|| irrelevant). Watch")
                log("          spend_mean to see which. --csf_param pgd "
                    "removes both.")
            frac = st["visibility"] / max(c.csf_threshold, 1e-12)
            if frac < 0.5:
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
