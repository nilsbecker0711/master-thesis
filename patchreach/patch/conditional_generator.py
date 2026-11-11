"""Image-conditioned adversarial patch generator."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.cityscapes import upsample_to
from . import csf as csf_mod
from . import lap as lap_mod
from . import placement as placement_mod

COND_MODES = ("image", "image+ref", "image+ref+cam")
RESIDUAL_MODES = ("logit", "clip", "none", "csf")
PLACEMENT_MODES = ("center", "gradcam", "semantic", "fixed")
REFERENCE_MODES = ("center", "window")


@dataclass
class GeneratorConfig:
    """ARCHITECTURE ONLY — everything needed to rebuild G_theta from a checkpoint."""
    size: int = 128
    base_ch: int = 32
    depth: int = 3
    cond: str = "image+ref+cam"
    residual: str = "logit"
    residual_scale: float = 1.0
    noise_dim: int = 0

    csf_threshold: float = 0.25
    csf_min_cycles: float = 2.0
    csf_model: str = "barten"
    csf_beta: float = 3.0
    csf_pixel_size_cm: float = 0.0114
    csf_viewing_distance_cm: float = 50.0

    def validate(self) -> "GeneratorConfig":
        if self.cond not in COND_MODES:
            raise ValueError(f"cond must be one of {COND_MODES}, got {self.cond!r}")
        if self.residual not in RESIDUAL_MODES:
            raise ValueError(f"residual must be one of {RESIDUAL_MODES}, "
                             f"got {self.residual!r}")
        if self.size % (2 ** self.depth) != 0:
            raise ValueError(
                f"size={self.size} must be divisible by 2**depth="
                f"{2 ** self.depth}; the U-Net skips need matching resolutions")
        return self

    @property
    def geometry(self) -> "csf_mod.ViewingGeometry":
        return csf_mod.ViewingGeometry(self.csf_pixel_size_cm,
                                       self.csf_viewing_distance_cm)

    @property
    def in_channels(self) -> int:
        ch = 3
        if "ref" in self.cond:
            ch += 3
        if "cam" in self.cond:
            ch += 2
        return ch + self.noise_dim

    def describe(self, log=print):
        log(f"[gen ] cond      : {self.cond}  ({self.in_channels} input channels)")
        log(f"[gen ] residual  : {self.residual}  (scale {self.residual_scale:g})")
        log(f"[gen ] output    : {self.size}x{self.size} in [0,1]")
        log(f"[gen ] noise     : "
            + ("off — deterministic" if self.noise_dim == 0
               else f"{self.noise_dim} channels"))
        if self.residual == "csf":
            log(f"[gen ] perceptual: {self.csf_model} CSF, tau = "
                f"{self.csf_threshold:g} JND, Minkowski beta = {self.csf_beta:g}, "
                f"min {self.csf_min_cycles:g} cycles/patch")
            self.geometry.describe(log)


def _norm(ch: int) -> nn.Module:
    """GroupNorm, NOT BatchNorm."""
    return nn.GroupNorm(math.gcd(8, ch), ch)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _norm(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), _norm(cout), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.f(x)


class ConditionalPatchGenerator(nn.Module):
    """U-Net over the conditioning stack, emitting a RESIDUAL Delta at [3,S,S]."""

    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg.validate()

        chans = [cfg.base_ch * (2 ** i) for i in range(cfg.depth + 1)]
        self.stem = _Block(cfg.in_channels, chans[0])

        self.down = nn.ModuleList(
            [_Block(chans[i], chans[i + 1]) for i in range(cfg.depth)])
        self.up = nn.ModuleList(
            [_Block(chans[i + 1] + chans[i], chans[i])
             for i in reversed(range(cfg.depth))])

        self.head = nn.Conv2d(chans[0], 3, 3, padding=1)
        if cfg.residual == "csf":
            nn.init.normal_(self.head.weight, std=0.05)
            nn.init.zeros_(self.head.bias)
            cmap, budget = csf_mod.patch_budget(
                cfg.size, cfg.geometry, cfg.csf_model, cfg.csf_threshold,
                cfg.csf_min_cycles)
            self.register_buffer("csf_values", cmap)
            self.register_buffer("csf_budget", budget)
        else:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _stack(self, image, reference, cam_global, cam_local, noise):
        cfg = self.cfg
        parts = [image]
        if "ref" in cfg.cond:
            if reference is None:
                raise ValueError(f"cond={cfg.cond!r} needs `reference`")
            parts.append(reference)
        if "cam" in cfg.cond:
            if cam_global is None or cam_local is None:
                raise ValueError(f"cond={cfg.cond!r} needs `cam_global` and "
                                 f"`cam_local`")
            parts += [cam_global, cam_local]
        if cfg.noise_dim:
            if noise is None:
                raise ValueError(f"noise_dim={cfg.noise_dim} needs `noise`")
            parts.append(noise)
        return torch.cat(parts, dim=1)

    def sample_noise(self, B: int, device, deterministic: bool = False):
        """z_i ~ N(0,1) during training; ZEROS (the prior mean) at evaluation."""
        if not self.cfg.noise_dim:
            return None
        shape = (B, self.cfg.noise_dim, self.cfg.size, self.cfg.size)
        if deterministic:
            return torch.zeros(shape, device=device)
        return torch.randn(shape, device=device)

    def forward(self, image, reference=None, cam_global=None, cam_local=None,
                noise=None) -> torch.Tensor:
        x = self.stem(self._stack(image, reference, cam_global, cam_local, noise))

        skips = []
        for blk in self.down:
            skips.append(x)
            x = blk(F.avg_pool2d(x, 2))

        for blk, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            x = blk(torch.cat([x, skip], dim=1))

        delta = self.head(x) * self.cfg.residual_scale
        return self.compose(delta, reference)

    def compose(self, delta: torch.Tensor,
                reference: Optional[torch.Tensor]) -> torch.Tensor:
        """Delta -> p in [0,1]."""
        mode = self.cfg.residual
        if mode == "none":
            return torch.sigmoid(delta)
        if reference is None:
            raise ValueError(f"residual={mode!r} needs `reference` as its base")
        if mode == "clip":
            return (reference + torch.tanh(delta)).clamp(0.0, 1.0)
        if mode == "csf":
            d = csf_mod.csf_residual(delta, self.csf_budget, self.csf_values,
                                     self.cfg.csf_threshold, self.cfg.csf_beta)
            d = csf_mod.fit_to_range(d, reference)
            return (reference + d).clamp(0.0, 1.0)
        return torch.sigmoid(lap_mod.logit_seed(reference) + delta)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def denormalise_batch(imgs: torch.Tensor, mean_t, std_t) -> torch.Tensor:
    """[B,3,H,W] normalised -> [B,3,H,W] in [0,1]."""
    return (imgs * std_t + mean_t).clamp(0.0, 1.0)


def patch_side(H: int, scale: float, W: Optional[int] = None,
               ref: str = "height") -> int:
    """Delegates to placement.footprint_side — the SAME rule Patch.apply() uses, now in one place so the two cannot drift."""
    if ref != "height" and W is None:
        raise ValueError(f"scale_ref={ref!r} needs W")
    return placement_mod.footprint_side(H, W if W is not None else H,
                                        scale, ref)


def center_crop_reference(imgs01: torch.Tensor, p: int, size: int
                          ) -> torch.Tensor:
    """r_i = Resize( CenterCrop(x_i) )"""
    _, _, H, W = imgs01.shape
    top, left = (H - p) // 2, (W - p) // 2
    crop = imgs01[:, :, top:top + p, left:left + p]
    return F.interpolate(crop, size=(size, size), mode="bilinear",
                         align_corners=False).clamp(0.0, 1.0).detach()


def window_reference(imgs01: torch.Tensor, placements: Sequence[Tuple[int, int]],
                     p: int, size: int) -> torch.Tensor:
    """r_i = Resize( Crop_{t_i, l_i}(x_i) ) — the content the patch REPLACES"""
    outs = [F.interpolate(imgs01[i:i + 1, :, t:t + p, l:l + p],
                          size=(size, size), mode="bilinear",
                          align_corners=False)
            for i, (t, l) in enumerate(placements)]
    return torch.cat(outs, dim=0).clamp(0.0, 1.0).detach()


def crop_windows(maps: torch.Tensor, placements: Sequence[Tuple[int, int]],
                 p: int, size: int) -> torch.Tensor:
    """Per-image crop of [B,C,H,W] at each (top,left), resized to size x size."""
    outs = [F.interpolate(maps[i:i + 1, :, t:t + p, l:l + p], size=(size, size),
                          mode="bilinear", align_corners=False)
            for i, (t, l) in enumerate(placements)]
    return torch.cat(outs, dim=0).detach()


def resize_to_patch(maps: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(maps, size=(size, size), mode="bilinear",
                         align_corners=False).detach()


@torch.no_grad()
def resolve_batch_placement(policy: str, H: int, W: int, p: int,
                            cam: Optional[torch.Tensor] = None,
                            clean_pred: Optional[torch.Tensor] = None,
                            cls: int = 0,
                            xy: Tuple[float, float] = (0.5, 0.5),
                            margin: int = 0
                            ) -> List[Tuple[int, int]]:
    """One (top, left) PER IMAGE."""
    if policy not in PLACEMENT_MODES:
        raise ValueError(f"placement must be one of {PLACEMENT_MODES}, "
                         f"got {policy!r}")

    if policy == "gradcam":
        if cam is None:
            raise ValueError("placement='gradcam' needs the sensitivity map")
        return [placement_mod.find_max_response_placement(
            cam[i, 0], p, centre_if_constant=True, margin=margin)
            for i in range(cam.shape[0])]

    if policy == "semantic":
        if clean_pred is None:
            raise ValueError("placement='semantic' needs the clean prediction")
        return [placement_mod.find_semantic_placement(clean_pred[i], cls, p)
                for i in range(clean_pred.shape[0])]

    B = cam.shape[0] if cam is not None else (
        clean_pred.shape[0] if clean_pred is not None else 1)
    return [placement_mod.resolve(policy, H, W, p, None, cls, xy)] * B


def composite_batch(imgs: torch.Tensor, patches: torch.Tensor,
                    placements: Sequence[Tuple[int, int]], p: int,
                    mean_t, std_t) -> Tuple[torch.Tensor, torch.Tensor]:
    """x_i^adv = Apply(x_i, p_i)"""
    B, _, H, W = imgs.shape
    if patches.shape[0] != B or len(placements) != B:
        raise ValueError(
            f"batch mismatch: imgs {B}, patches {patches.shape[0]}, "
            f"placements {len(placements)} — this is a per-image attack, every "
            f"image needs its own patch and its own placement")

    fulls, masks = [], []
    ones = torch.ones(1, 1, p, p, device=imgs.device, dtype=imgs.dtype)
    for i, (top, left) in enumerate(placements):
        rendered = F.interpolate(patches[i:i + 1], size=(p, p), mode="bilinear",
                                 align_corners=False)
        normed = (rendered - mean_t) / std_t
        t = max(0, min(int(top), H - p))
        l = max(0, min(int(left), W - p))
        pads = (l, W - p - l, t, H - p - t)
        fulls.append(F.pad(normed, pads))
        masks.append(F.pad(ones, pads))

    full = torch.cat(fulls, dim=0)
    mask = torch.cat(masks, dim=0)
    patched = imgs * (1.0 - mask) + full * mask
    return patched, (mask[:, 0] > 0.5)


class ConditionalAttack:
    """One batch, end to end"""

    def __init__(self, model, cam, generator, mean_t, std_t,
                 scale: float, size: int, placement: str = "gradcam",
                 placement_class: int = 0,
                 placement_xy: Tuple[float, float] = (0.5, 0.5),
                 method: str = "generator", reference: str = "center",
                 placement_margin: int = 0):
        self.model = model
        self.cam = cam
        self.generator = generator
        self.mean_t, self.std_t = mean_t, std_t
        self.scale, self.size = scale, size
        self.placement = placement
        self.placement_class = placement_class
        self.placement_xy = placement_xy
        self.placement_margin = placement_margin
        if method not in ("generator", "reference"):
            raise ValueError(f"method must be 'generator' or 'reference', "
                             f"got {method!r}")
        if reference not in REFERENCE_MODES:
            raise ValueError(f"reference must be one of {REFERENCE_MODES}, "
                             f"got {reference!r}")
        self.method = method
        self.reference = reference

    def __call__(self, imgs: torch.Tensor, labels: torch.Tensor,
                 deterministic_noise: bool = False) -> dict:
        B, _, H, W = imgs.shape
        p = patch_side(H, self.scale)

        cam, clean_logits = self.cam(imgs, labels)
        imgs01 = denormalise_batch(imgs, self.mean_t, self.std_t).detach()

        clean_pred = (upsample_to(clean_logits, (H, W)).argmax(1)
                      if self.placement == "semantic" else None)
        places = resolve_batch_placement(
            self.placement, H, W, p, cam=cam, clean_pred=clean_pred,
            cls=self.placement_class, xy=self.placement_xy,
            margin=self.placement_margin)

        refs = (window_reference(imgs01, places, p, self.size)
                if self.reference == "window"
                else center_crop_reference(imgs01, p, self.size))

        if self.method == "reference":
            patches = refs
        else:
            noise = self.generator.sample_noise(B, imgs.device,
                                                deterministic_noise)
            patches = self.generator(
                image=resize_to_patch(imgs01, self.size),
                reference=refs,
                cam_global=resize_to_patch(cam, self.size),
                cam_local=crop_windows(cam, places, p, self.size),
                noise=noise)

        patched, footprint = composite_batch(imgs, patches, places, p,
                                             self.mean_t, self.std_t)
        return {"patches": patches, "references": refs, "cam": cam,
                "placements": places, "patched": patched,
                "footprint": footprint, "clean_logits": clean_logits,
                "patch_side": p}


def lap_terms(patches: torch.Tensor, references: torch.Tensor,
              alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0
              ) -> dict:
    """L_LAP(p_i, r_i) = alpha*L_rat + beta*L_tv + gamma*L_nps"""
    z = torch.zeros((), device=patches.device)
    out = {"rat": z, "tv": z, "nps": z, "total": z}
    B = patches.shape[0]

    if alpha > 0:
        out["rat"] = sum(lap_mod.rationality_loss(patches[i], references[i])
                         for i in range(B)) / B
    if beta > 0:
        out["tv"] = sum(lap_mod.tv_loss(patches[i]) for i in range(B)) / B
    if gamma > 0:
        out["nps"] = sum(lap_mod.nps_loss(patches[i]) for i in range(B)) / B

    out["total"] = alpha * out["rat"] + beta * out["tv"] + gamma * out["nps"]
    return out


def build_lpips(device, net: str = "alex", log=print):
    """LPIPS(r_i, p_i), or None when the package is unavailable."""
    try:
        import lpips as _lpips
    except ImportError:
        log("[lpips] package not installed — perceptual metric will be "
            "SKIPPED. `pip install lpips` to enable it. Training is "
            "unaffected: LPIPS is never an objective.")
        return None
    return _lpips.LPIPS(net=net).to(device).eval()


@torch.no_grad()
def lpips_distances(metric, patches: torch.Tensor, references: torch.Tensor
                    ) -> List[float]:
    """Per-image LPIPS, returned as a LIST so the caller can report the DISTRIBUTION rather than only a mean."""
    if metric is None:
        return []
    d = metric(patches * 2.0 - 1.0, references * 2.0 - 1.0)
    return d.reshape(-1).cpu().tolist()


def as_patch(patch01: torch.Tensor, placement: Tuple[int, int], scale: float,
             device, mean_t, std_t, reference: Optional[torch.Tensor] = None):
    """Freeze ONE generated patch into a real `Patch` object."""
    from .spec import Patch, PatchConfig

    S = int(patch01.shape[-1])
    obj = Patch(PatchConfig(mode="conditional", size=S, scale=scale),
                device, mean_t, std_t)
    obj.param = lap_mod.logit_seed(patch01.detach()).clone()
    obj.placement = (int(placement[0]), int(placement[1]))
    if reference is not None:
        obj.reference = reference.detach()
    return obj


CHECKPOINT_FORMAT = "conditional_generator_v1"


def save_checkpoint(path, generator: ConditionalPatchGenerator, optimizer,
                    epoch: int, train_config: dict, patch_geometry: dict,
                    extra: Optional[dict] = None):
    """theta, the optimiser state, the epoch, and BOTH configurations."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "generator_state_dict": generator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "train_config": train_config,
        "generator_config": asdict(generator.cfg),
        "patch_geometry": patch_geometry,
        **(extra or {}),
    }, path)


def load_checkpoint(path, device):
    """(generator in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location="cpu")
    fmt = ck.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a conditional-generator checkpoint (format="
            f"{fmt!r}). A Patch checkpoint from train.py/overfit.py stores a "
            f"single 'param' tensor and must be loaded with Patch.load().")
    gen = ConditionalPatchGenerator(GeneratorConfig(**ck["generator_config"]))
    gen.load_state_dict(ck["generator_state_dict"])
    return gen.to(device).eval(), ck
