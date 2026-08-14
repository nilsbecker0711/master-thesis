r"""
Image-conditioned adversarial patch generator.

    p_i = G_theta(C_i, r_i, M_i, z_i)

    C_i  the target image (global scene context)
    r_i  a fixed-size CENTRE CROP of x_i, resized to the patch resolution
    M_i  the segmentation sensitivity map from segmentation_cam.py, DETACHED
    z_i  optional noise, OFF by default (deterministic generation)

    theta* = argmin_theta  E_{(x,y)~D} [ L_attack(f(x (+) G_theta(x,r,M)), y)
                                         + lambda_LAP * L_LAP(p_i, r_i) ]

A DIFFERENT THREAT MODEL FROM EVERY EXISTING MODE
-------------------------------------------------
raw / lap / gan / raw_ganinit all optimise ONE patch tensor. Here the patch is
not a parameter at all: theta is shared across the whole dataset and the patch
is a FUNCTION of the image. At test time an unseen image gets its own patch
from one forward pass — there is NO gradient-based optimisation at test time.
That is what separates this from the universal patch, and it is the claim the
evaluation has to support.

WHY NOT Patch.apply()
---------------------
Patch.apply() renders ONE [3,S,S] and broadcasts it across the batch under ONE
self.placement. Per-image patches with per-image placement cannot be expressed
through it. composite_batch() below therefore MIRRORS its semantics exactly —
same interpolate -> normalise -> F.pad order, same top/left clamping, same
`> 0.5` footprint rule, same reason for using F.pad rather than slice
assignment (in-place slicing into a no-grad tensor creates no backward node,
so the gradient silently vanishes). tests/test_conditional_generator.py
asserts the two agree numerically in the degenerate shared-patch case, so this
mirror cannot drift from the original unnoticed.

WHY THE SIGMOID RESIDUAL IS THE DEFAULT
---------------------------------------
spec.py unified every pixel mode on `patch = sigmoid(param)` specifically
because a hard clamp to [0,1] zeroes the gradient for any pixel sitting on the
bound — which is exactly where an adversarial patch wants to live. The same
argument applies to the residual head, so:

    logit (default)  p = sigmoid( logit_seed(r) + Delta )
    clip             p = clamp( r + Delta, 0, 1 )       <- the literal formula
    none             p = sigmoid( Delta )               <- ignores r as a base

All three give p == r (or 0.5 grey for `none`, matching `raw` mode's init) at
step 0, because the final conv is ZERO-INITIALISED. The untrained generator is
therefore EXACTLY baseline A, which makes the baseline comparison exact rather
than approximate.

NORMALISATION CONVENTION
------------------------
The dataloader emits IMAGENET-NORMALISED tensors. The generator works entirely
in [0,1] RGB — references, patches and LAP losses all live there, matching
Patch.render(). composite_batch() applies (x - mean)/std at the last moment,
exactly where Patch.apply() does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.cityscapes import upsample_to
from . import lap as lap_mod
from . import placement as placement_mod

COND_MODES = ("image", "image+ref", "image+ref+cam")
RESIDUAL_MODES = ("logit", "clip", "none")
PLACEMENT_MODES = ("center", "gradcam", "semantic", "fixed")


# ═════════════════════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratorConfig:
    """
    ARCHITECTURE ONLY — everything needed to rebuild G_theta from a checkpoint.

    Placement policy, patch scale and LAP weights are TRAINING configuration,
    not architecture, and are stored separately in the checkpoint so an
    ablation can re-evaluate one generator under several placement policies
    without implying the weights changed.
    """
    size: int = 128                  # output resolution S (p_i is [3,S,S])
    base_ch: int = 32
    depth: int = 3                   # downsampling levels
    cond: str = "image+ref+cam"      # ablation E
    residual: str = "logit"          # logit | clip | none
    residual_scale: float = 1.0
    noise_dim: int = 0               # 0 = deterministic

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
    def in_channels(self) -> int:
        ch = 3                                    # C_i, always present
        if "ref" in self.cond:
            ch += 3                               # r_i
        if "cam" in self.cond:
            ch += 2                               # M_i global + M_i at the window
        return ch + self.noise_dim

    def describe(self, log=print):
        log(f"[gen ] cond      : {self.cond}  ({self.in_channels} input channels)")
        log(f"[gen ] residual  : {self.residual}  (scale {self.residual_scale:g})")
        log(f"[gen ] output    : {self.size}x{self.size} in [0,1]")
        log(f"[gen ] noise     : "
            + ("off — deterministic" if self.noise_dim == 0
               else f"{self.noise_dim} channels"))


# ═════════════════════════════════════════════════════════════════════════════
#  Architecture
# ═════════════════════════════════════════════════════════════════════════════

def _norm(ch: int) -> nn.Module:
    """
    GroupNorm, NOT BatchNorm.

    Two reasons that both bite in this setup: batch sizes are small (4 by
    default, 1 at export time), and BatchNorm behaves differently in train()
    and eval(). The generator is trained in train() and run in eval() at test
    time, so a BatchNorm would make the reported test-time patch differ from
    the one the attack loss actually optimised. gcd keeps the group count legal
    for any channel width.
    """
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
    r"""
    U-Net over the conditioning stack, emitting a RESIDUAL Delta at [3,S,S].

    Shared across the entire dataset: ONE theta, no per-image parameter tensor.
    The output is a function of the inputs only, so an unseen image produces its
    patch in a single forward pass.

    forward(image, reference, cam_global, cam_local, noise) -> p in [0,1]^{B,3,S,S}

    All conditioning inputs are [B,*,S,S] in [0,1] and are expected to arrive
    DETACHED — none of them is optimised. `reference` is required whenever
    residual != 'none' because it is the base of the residual, even under
    cond='image' where it is not fed in as an input channel (ablation E: the
    generator is denied the reference as INFORMATION while the residual base
    stays fixed, so the comparison isolates conditioning rather than confounding
    it with a change of parameterisation).
    """

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
        # ZERO-INIT: Delta == 0 at step 0, so p_i == r_i exactly and the
        # untrained generator IS baseline A. Any drift from the reference is
        # then attributable to training, not to initialisation noise.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ── conditioning ─────────────────────────────────────────────────────────

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
        """
        z_i ~ N(0,1) during training; ZEROS (the prior mean) at evaluation.

        Deterministic evaluation matters here: the whole claim is "one frozen
        generator, one patch per image". A resampled z would make the reported
        test-time patch irreproducible.
        """
        if not self.cfg.noise_dim:
            return None
        shape = (B, self.cfg.noise_dim, self.cfg.size, self.cfg.size)
        if deterministic:
            return torch.zeros(shape, device=device)
        return torch.randn(shape, device=device)

    # ── forward ──────────────────────────────────────────────────────────────

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
        """Delta -> p in [0,1]. Split out so tests can drive it directly."""
        mode = self.cfg.residual
        if mode == "none":
            return torch.sigmoid(delta)
        if reference is None:
            raise ValueError(f"residual={mode!r} needs `reference` as its base")
        if mode == "clip":
            return (reference + torch.tanh(delta)).clamp(0.0, 1.0)
        return torch.sigmoid(lap_mod.logit_seed(reference) + delta)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
#  Conditioning inputs
# ═════════════════════════════════════════════════════════════════════════════

def denormalise_batch(imgs: torch.Tensor, mean_t, std_t) -> torch.Tensor:
    """
    [B,3,H,W] normalised -> [B,3,H,W] in [0,1].

    The batched counterpart of data.cityscapes.denormalise, which only handles
    a single [1,3,H,W] and moves the result to CPU. mean_t/std_t are the
    [1,3,1,1] tensors from norm_tensors(), so this broadcasts as-is.
    """
    return (imgs * std_t + mean_t).clamp(0.0, 1.0)


def patch_side(H: int, scale: float) -> int:
    """p = int(H * scale) — the SAME rule Patch.apply() uses. Do not change."""
    return int(H * scale)


def center_crop_reference(imgs01: torch.Tensor, p: int, size: int
                          ) -> torch.Tensor:
    r"""
        r_i = Resize( CenterCrop(x_i) )

    Fixed-size spatial centre crop of side p, resized to the generator's output
    resolution, kept in [0,1] RGB. NOT optimised and NOT searched for — §4 of
    the specification is explicit that the first implementation must not
    dynamically locate an object. Detached, so no gradient can reach the image.

    Note the crop is taken at the image CENTRE while the patch may be PLACED
    elsewhere (gradcam placement). The reference is therefore a visual anchor,
    not a prediction of what the patch will cover. That is a deliberate
    property of this first implementation and it is what cam_local exists to
    partially compensate for.
    """
    _, _, H, W = imgs01.shape
    top, left = (H - p) // 2, (W - p) // 2
    crop = imgs01[:, :, top:top + p, left:left + p]
    return F.interpolate(crop, size=(size, size), mode="bilinear",
                         align_corners=False).clamp(0.0, 1.0).detach()


def crop_windows(maps: torch.Tensor, placements: Sequence[Tuple[int, int]],
                 p: int, size: int) -> torch.Tensor:
    """
    Per-image crop of [B,C,H,W] at each (top,left), resized to size x size.

    Used for M_i restricted to the window the patch will actually occupy — the
    only part of the sensitivity map that is spatially aligned with the output.
    """
    outs = [F.interpolate(maps[i:i + 1, :, t:t + p, l:l + p], size=(size, size),
                          mode="bilinear", align_corners=False)
            for i, (t, l) in enumerate(placements)]
    return torch.cat(outs, dim=0).detach()


def resize_to_patch(maps: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(maps, size=(size, size), mode="bilinear",
                         align_corners=False).detach()


# ═════════════════════════════════════════════════════════════════════════════
#  Placement
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def resolve_batch_placement(policy: str, H: int, W: int, p: int,
                            cam: Optional[torch.Tensor] = None,
                            clean_pred: Optional[torch.Tensor] = None,
                            cls: int = 0,
                            xy: Tuple[float, float] = (0.5, 0.5)
                            ) -> List[Tuple[int, int]]:
    r"""
    One (top, left) PER IMAGE.

    center   : image centre. Identical to placement.resolve('center', ...).
    gradcam  : argmax_{(u,v)} MeanPool( M_i[u:u+p, v:v+p] ) — the patch-sized
               window of highest average segmentation sensitivity.
    semantic : the EXISTING policy, per image, on the clean prediction.
    fixed    : the EXISTING policy — a normalised (y,x) centre.

    Existing modes are untouched: this function is only ever reached from the
    conditional-generator entry points, and center/fixed/semantic delegate to
    placement.py rather than reimplementing it.
    """
    if policy not in PLACEMENT_MODES:
        raise ValueError(f"placement must be one of {PLACEMENT_MODES}, "
                         f"got {policy!r}")

    if policy == "gradcam":
        if cam is None:
            raise ValueError("placement='gradcam' needs the sensitivity map")
        return [placement_mod.find_max_response_placement(
            cam[i, 0], p, centre_if_constant=True)
            for i in range(cam.shape[0])]

    if policy == "semantic":
        if clean_pred is None:
            raise ValueError("placement='semantic' needs the clean prediction")
        return [placement_mod.find_semantic_placement(clean_pred[i], cls, p)
                for i in range(clean_pred.shape[0])]

    B = cam.shape[0] if cam is not None else (
        clean_pred.shape[0] if clean_pred is not None else 1)
    return [placement_mod.resolve(policy, H, W, p, None, cls, xy)] * B


# ═════════════════════════════════════════════════════════════════════════════
#  Compositing
# ═════════════════════════════════════════════════════════════════════════════

def composite_batch(imgs: torch.Tensor, patches: torch.Tensor,
                    placements: Sequence[Tuple[int, int]], p: int,
                    mean_t, std_t) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
        x_i^adv = Apply(x_i, p_i)

    Returns (patched [B,3,H,W] normalised, footprint [B,H,W] bool).

    SEMANTICS ARE Patch.apply()'s, PER IMAGE. Same order (interpolate to p x p
    -> (x-mean)/std -> F.pad), same clamping of top/left into [0, H-p] /
    [0, W-p], same `> 0.5` footprint rule.

    GRADIENT SAFETY: built with F.pad, never in-place slice assignment — see
    the note in Patch.apply(). The per-image loop plus torch.cat keeps that
    property; F.pad cannot take per-sample padding, and the alternative
    (scattering into a preallocated canvas) is exactly the slice-assignment
    pattern that silently breaks the backward graph. The loop costs nothing
    measurable: the frozen segmentation forward pass dominates by orders of
    magnitude at these batch sizes.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═════════════════════════════════════════════════════════════════════════════

class ConditionalAttack:
    r"""
    One batch, end to end:

        x_i -> M_i (detached) -> r_i -> placement -> p_i = G(x_i,r_i,M_i)
            -> x_i^adv = Apply(x_i, p_i)

    Shared by training, evaluation and export so the three cannot drift. That
    matters more than usual here: the central claim is that TEST-TIME
    behaviour is a pure forward pass of the same function trained at
    train-time, and two separate implementations of the pipeline would make
    that unverifiable.

    `method`:
        generator  p_i = G_theta(...)            — the proposed attack
        reference  p_i = r_i                     — BASELINE A, no generator at
                                                   all. Runs through identical
                                                   placement and compositing,
                                                   so the comparison isolates
                                                   the generator.
    """

    def __init__(self, model, cam, generator, mean_t, std_t,
                 scale: float, size: int, placement: str = "gradcam",
                 placement_class: int = 0,
                 placement_xy: Tuple[float, float] = (0.5, 0.5),
                 method: str = "generator"):
        self.model = model
        self.cam = cam
        self.generator = generator
        self.mean_t, self.std_t = mean_t, std_t
        self.scale, self.size = scale, size
        self.placement = placement
        self.placement_class = placement_class
        self.placement_xy = placement_xy
        if method not in ("generator", "reference"):
            raise ValueError(f"method must be 'generator' or 'reference', "
                             f"got {method!r}")
        self.method = method

    def __call__(self, imgs: torch.Tensor, labels: torch.Tensor,
                 deterministic_noise: bool = False) -> dict:
        B, _, H, W = imgs.shape
        p = patch_side(H, self.scale)

        # 1-2. clean prediction + sensitivity. Both come out of ONE forward
        #      pass, and both are DETACHED — M_i is a conditioning signal, and
        #      no gradient may run back through the frozen segmentation model
        #      into the generator by this route.
        cam, clean_logits = self.cam(imgs, labels)

        # 3. reference: centre crop in [0,1], never optimised
        imgs01 = denormalise_batch(imgs, self.mean_t, self.std_t).detach()
        refs = center_crop_reference(imgs01, p, self.size)

        # 5. placement (before generation: cam_local is cropped at the window)
        # upsample BEFORE argmax — the head emits below input resolution and
        # placement is measured in image pixels. Same order as everywhere else
        # in the repo (upsample_to(...).argmax(1)).
        clean_pred = (upsample_to(clean_logits, (H, W)).argmax(1)
                      if self.placement == "semantic" else None)
        places = resolve_batch_placement(
            self.placement, H, W, p, cam=cam, clean_pred=clean_pred,
            cls=self.placement_class, xy=self.placement_xy)

        # 4. generate
        if self.method == "reference":
            patches = refs                       # BASELINE A
        else:
            noise = self.generator.sample_noise(B, imgs.device,
                                                deterministic_noise)
            patches = self.generator(
                image=resize_to_patch(imgs01, self.size),
                reference=refs,
                cam_global=resize_to_patch(cam, self.size),
                cam_local=crop_windows(cam, places, p, self.size),
                noise=noise)

        # 6. composite
        patched, footprint = composite_batch(imgs, patches, places, p,
                                             self.mean_t, self.std_t)
        return {"patches": patches, "references": refs, "cam": cam,
                "placements": places, "patched": patched,
                "footprint": footprint, "clean_logits": clean_logits,
                "patch_side": p}


# ═════════════════════════════════════════════════════════════════════════════
#  LAP constraint, per image
# ═════════════════════════════════════════════════════════════════════════════

def lap_terms(patches: torch.Tensor, references: torch.Tensor,
              alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0
              ) -> dict:
    r"""
        L_LAP(p_i, r_i) = alpha*L_rat + beta*L_tv + gamma*L_nps

    scored per image against its OWN reference and averaged over the batch.

    Reuses lap.py's implementations unchanged. It cannot go through
    Patch.regularisers(), which is hardcoded to mode=='lap' and to ONE static
    self.reference — here the reference is image-specific.

    active_mask() has no analogue: shape masking derives a silhouette from a
    reference FILE (shape.py) and there is none here, so q == p over the full
    square. That is the same situation Patch.describe() reports as "q = M(p) :
    FULL RECTANGLE".

    SCALE WARNING (lap.py's, unchanged): L_rat and L_tv are SUMS over the
    patch, order 1e2-1e4, while the attack losses are per-pixel MEANS of order
    1e-2 to 20. Weights default to 0; read magnitude_report() before setting
    them or the constraint swamps the attack by orders of magnitude.

    Zero-weighted terms are skipped, so an unconstrained run costs nothing.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
#  LPIPS — EVALUATION ONLY
# ═════════════════════════════════════════════════════════════════════════════

def build_lpips(device, net: str = "alex", log=print):
    r"""
    LPIPS(r_i, p_i), or None when the package is unavailable.

    NEVER A TRAINING LOSS. Training is L_attack + lambda*L_LAP only. Keeping
    LPIPS out of the objective is what makes it an INDEPENDENT perceptual
    measurement — optimising against it would make a low reported LPIPS a
    restatement of the objective rather than evidence about the patch.

    lpips is not in pyproject dependencies (the cluster env is pinned around
    torch 1.11 + mmcv 1.x), so the import is lazy and a miss degrades to
    "metric not reported" instead of killing a finished training run.
    """
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
    """
    Per-image LPIPS, returned as a LIST so the caller can report the
    DISTRIBUTION rather than only a mean. LPIPS expects [-1,1].
    """
    if metric is None:
        return []
    d = metric(patches * 2.0 - 1.0, references * 2.0 - 1.0)
    return d.reshape(-1).cpu().tolist()


# ═════════════════════════════════════════════════════════════════════════════
#  Diagnostics bridge
# ═════════════════════════════════════════════════════════════════════════════

def as_patch(patch01: torch.Tensor, placement: Tuple[int, int], scale: float,
             device, mean_t, std_t, reference: Optional[torch.Tensor] = None):
    r"""
    Freeze ONE generated patch into a real `Patch` object.

    WHY THIS EXISTS
    ---------------
    For a single image the generator's output IS a patch plus a placement —
    exactly what `Patch` already models. Freezing it into one lets the ENTIRE
    existing diagnostic suite (report.run, report.panels_for_images, the ERF
    probe, reach curves, confusion/flow tables, margin and entropy figures) run
    on the conditional attack with no parallel implementation and no changes to
    diagnostics code.

    The alternative — a duck-typed adapter — would have to re-expose .param,
    .apply(), .render(), .cfg, .placement and .shape_mask, and would silently
    drift from Patch's semantics. This returns the real thing instead.

    `param` is seeded through logit_seed so sigmoid(param) == patch01 exactly,
    which matters for the ERF probe: geometric.receptive_field() overwrites
    param with noise and restores it afterwards, so the probe measures the
    architecture's reach AT THE PLACEMENT THE GENERATOR CHOSE. That is the
    correct control for this attack and is directly comparable with the probe
    run for every other patch mode.

    NOT for training or checkpointing — it is a diagnostic snapshot of one
    image's result. The generator is the model; this is one of its outputs.
    """
    from .spec import Patch, PatchConfig

    S = int(patch01.shape[-1])
    obj = Patch(PatchConfig(mode="conditional", size=S, scale=scale),
                device, mean_t, std_t)
    obj.param = lap_mod.logit_seed(patch01.detach()).clone()
    obj.placement = (int(placement[0]), int(placement[1]))
    if reference is not None:
        obj.reference = reference.detach()
    return obj


# ═════════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═════════════════════════════════════════════════════════════════════════════

CHECKPOINT_FORMAT = "conditional_generator_v1"


def save_checkpoint(path, generator: ConditionalPatchGenerator, optimizer,
                    epoch: int, train_config: dict, patch_geometry: dict,
                    extra: Optional[dict] = None):
    """
    theta, the optimiser state, the epoch, and BOTH configurations.

    Deliberately NOT a rendered patch: a single image's patch is an OUTPUT of
    this model, not the model. Saving one would make the artefact
    indistinguishable from a universal-patch checkpoint and would lose the only
    thing that generalises.
    """
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
    """(generator in eval mode, checkpoint dict). Architecture from the file."""
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
