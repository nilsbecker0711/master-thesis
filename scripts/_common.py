"""Shared CLI plumbing. Keeps the entry points thin and consistent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from patchreach.data.cityscapes import CityscapesSeg, norm_tensors
from patchreach.models.registry import REGISTRY, resolve, load_segmentor
from patchreach.models.wrapper import WrappedSegModel
from patchreach.utils import get_device, seed_everything, channel_probe

# Ten fixed val images. EVERY number in the thesis is reported on these, so
# scene-to-scene variance (which is large for contestability) is measured
# rather than accidental.
FIXED10 = [430, 46, 441, 331, 45, 162, 195, 424, 91, 353]


def add_model_args(p):
    p.add_argument("--arch", default="internimage_t",
                   help=f"one of {sorted(REGISTRY)}")
    p.add_argument("--cfg_path", default=None,
                   help="mmseg config (.py). Overrides the registry entry. "
                        "Must MATCH the checkpoint variant — a b0 config will "
                        "not build b5.")
    p.add_argument("--weights", default=None, help="checkpoint (.pth)")
    p.add_argument("--cityscapes_root", default=None,
                   help="Cityscapes root. Required, but may come from --config\n"
                        "where a script supports one — validated in setup_model().")
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--seed", type=int, default=42)
    return p


def add_patch_args(p):
    p.add_argument("--patch_mode", default="raw",
                   choices=["raw", "gan", "raw_ganinit", "lap", "csf",
                            "universal_csf"],
                   help="csf: base + a residual whose spectrum is bounded by "
                        "the human contrast sensitivity function. Pair with "
                        "--from_image in overfit.py so the base is the region "
                        "the patch covers; otherwise the base is grey and the "
                        "patch is a visible square with an invisible texture. "
                        "universal_csf: ONE residual shared across the whole "
                        "dataset, added onto whatever content it lands on — "
                        "the non-adaptive control for csf. train.py only; it "
                        "is meaningless on a single image, where it IS csf.")
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--patch_scale", type=float, default=0.25)
    p.add_argument("--logit_clip", type=float, default=6.0,
                   help="bound on |param| for pixel modes. Stops the\n"
                        "sigmoid saturating into a dead-gradient state. 0 disables.")
    p.add_argument("--shape", default="square",
                   choices=["square", "alpha", "chroma", "auto"],
                   help="silhouette source — the Bg() term of Tan et al. Eq 5. "
                        "'alpha' needs an RGBA reference; 'auto' keys the "
                        "background from the corner medians.")
    p.add_argument("--reference_fit", default="auto",
                   choices=["auto", "crop", "pad", "stretch"],
                   help="how to square a non-square reference. "
                        "auto: crop to the alpha bbox then pad "
                        "(maximises object coverage), else "
                        "centre-crop. stretch DISTORTS geometry.")
    p.add_argument("--shape_bg", default="white", choices=["white", "black"],
                   help="background colour to key out for --shape chroma")
    p.add_argument("--shape_thresh", type=float, default=0.15,
                   help="chroma keying distance threshold")
    p.add_argument("--placement", default="center",
                   choices=["center", "fixed", "semantic", "gradcam"],
                   help="where the patch goes. 'gradcam' is the argmax over "
                        "MeanPool(M_i) — the per-image sensitivity hotspot, "
                        "the SAME policy --gen_placement gradcam uses on the "
                        "conditional generator. It is only coherent for "
                        "PER-IMAGE attacks (overfit.py, overfit_population.py): "
                        "a universal patch is shared across images while the "
                        "map is not, so train.py would be placing one tensor "
                        "by a different image's sensitivity every batch.")
    p.add_argument("--placement_class", type=int, default=0,
                   help="trainId for --placement semantic. 0=road is "
                        "omnipresent in Cityscapes; 1=sidewalk is absent from "
                        "many frames and would silently fall back to centre.")
    p.add_argument("--placement_xy", type=float, nargs=2, default=[0.5, 0.5],
                   help="normalised (y, x) CENTRE for --placement fixed. "
                        "(0.75, 0.5) is the road surface straight ahead.")
    p.add_argument("--placement_margin", type=int, default=0,
                   help="keep a --placement gradcam window this many px from "
                        "every image border. DEFAULT 0 reproduces the "
                        "unmargined argmax and leaves center/fixed/semantic "
                        "untouched. The sensitivity map's hottest ridge is the "
                        "near-field road boundary along the bottom of a "
                        "dashcam frame, so the argmax tends to pin the patch "
                        "flush against the edge, where ~half its receptive "
                        "field lies outside the image and can never be "
                        "influenced. Try p/2 (64 at the default geometry).")
    p.add_argument("--reference", default=None)
    p.add_argument("--lap_alpha", type=float, default=0.0)
    p.add_argument("--lap_beta", type=float, default=0.0)
    p.add_argument("--lap_gamma", type=float, default=0.0)
    p.add_argument("--lap_freeze_edges", action="store_true",
                   help="hard-pin the reference's strong edges (Grad term of "
                        "Eq 5) so the outline survives optimisation")
    p.add_argument("--lap_edge_thresh", type=float, default=0.1,
                   help="theta for the edge mask. Tan et al. use 0.1, tuned "
                        "for FLAT CARTOONS. Photographs have far denser "
                        "gradients — at 0.1 roughly half a photo reference "
                        "gets frozen — so 0.15-0.25 is usually right. Check "
                        "with score_reference.py --sweep_edge; aim for "
                        "free/sil of 0.5-0.7.")
    p.add_argument("--init_from", default=None)
    add_cam_args(p)
    add_csf_args(p)
    return p


def add_cam_args(p):
    r"""
    Sensitivity-map arguments — shared by add_patch_args and
    add_generator_args, one definition.

    These used to live only on the generator, because gradcam placement lived
    only on the generator. --placement gradcam brings the same policy to the
    per-image patch modes, and it needs the same map, built the same way. A
    second copy of these flags is exactly the drift that put three bugs into
    train.py/overfit.py before add_patch_args existed.

    Guarded like add_csf_args so a parser that ever takes BOTH groups does not
    raise on the duplicate option strings.
    """
    if any(a.option_strings and "--cam_objective" in a.option_strings
           for a in p._actions):
        return p
    g = p.add_argument_group("sensitivity map (Grad-CAM)")
    g.add_argument("--cam_objective", default="attack",
                   choices=["attack", "ce", "cospgd", "ipatch_cospgd"],
                   help="ABLATION F. Scalar S_seg differentiated for the CAM. "
                        "'attack' reuses --loss_fn so the map and the attack "
                        "share one objective. S_seg = -L, because every loss "
                        "in adversarial.py is MINIMISED to attack while "
                        "Grad-CAM needs a score to increase.")
    g.add_argument("--cam_target", default="pred", choices=["pred", "gt"],
                   help="labels for S_seg. 'pred' uses the model's own argmax, "
                        "keeping the LABEL-FREE threat model that --placement "
                        "semantic already assumes. 'gt' is a strictly stronger "
                        "attacker and must be declared as such.")
    g.add_argument("--cam_layer", type=int, default=-1,
                   help="which feature map from the hooked module. -1 = "
                        "deepest/coarsest, the standard Grad-CAM choice.")
    g.add_argument("--cam_module", default="backbone",
                   help="dotted path inside the mmseg segmentor to hook")
    return p


def add_csf_args(p):
    """Shared by add_patch_args and add_generator_args — one definition."""
    if any(a.option_strings and "--csf_threshold" in a.option_strings
           for a in p._actions):
        return p
    g = p.add_argument_group("perceptual constraint (csf)")
    g.add_argument("--csf_threshold", type=float, default=0.25)
    g.add_argument("--csf_model", default="barten", choices=["barten", "sso"])
    g.add_argument("--csf_beta", type=float, default=3.0)
    g.add_argument("--csf_min_cycles", type=float, default=2.0)
    g.add_argument("--csf_pixel_size_cm", type=float, default=0.0114)
    g.add_argument("--csf_viewing_distance_cm", type=float, default=50.0)
    g.add_argument("--csf_display_peak_cd_m2", type=float, default=100.0,
                   help="display peak white, for the luminance-aware path "
                        "only. Sets the absolute luminance Barten's pupil "
                        "formula needs; 100 is the sRGB reference white.")
    g.add_argument("--csf_lref", type=float, default=0.0,
                   help="reference LINEAR luminance Y for the CSF budget. "
                        "DEFAULT 0 keeps the legacy mu=0.5 convention, so a "
                        "tau here means what it means in every run recorded so "
                        "far. >0 switches to the calibrated budget, which is "
                        "~1.9x tighter at Nyquist — measured, not estimated. "
                        "0.097 is the Cityscapes train median at centre.")
    g.add_argument("--csf_composite", default="clip", choices=["clip", "fit"],
                   help="universal_csf only. clip: x+delta clamped to [0,1], "
                        "and frac_clipped reports where that bit. fit: "
                        "fit_to_range rescales instead, preserving the "
                        "spectrum but introducing a PER-IMAGE scale — which is "
                        "an adaptation a universal patch should not have. "
                        "clip is the honest default; fit is the control.")
    return p


def add_generator_args(p):
    r"""
    Image-conditioned generator (--mode conditional_generator).

    A SEPARATE argument group from add_patch_args on purpose. The generator is
    a different threat model, not another patch_mode: raw/lap/gan/raw_ganinit
    all optimise ONE patch tensor, while here theta is shared across the
    dataset and the patch is a FUNCTION of the image. Folding it into
    --patch_mode would have let it inherit reference/shape/init_from semantics
    that do not apply, and would have put a second meaning on Patch.param.
    """
    g = p.add_argument_group("conditional generator")

    # ── geometry ─────────────────────────────────────────────────────────────
    g.add_argument("--patch_size", type=int, default=128,
                   help="S — the generator's output resolution")
    g.add_argument("--patch_scale", type=float, default=0.25,
                   help="patch side as a fraction of image HEIGHT; "
                        "p = int(H*scale), the same rule Patch.apply() uses")

    # ── architecture (saved as generator_config) ─────────────────────────────
    g.add_argument("--gen_base_ch", type=int, default=32)
    g.add_argument("--gen_depth", type=int, default=3,
                   help="U-Net downsampling levels; patch_size must be "
                        "divisible by 2**depth")
    g.add_argument("--gen_cond", default="image+ref+cam",
                   choices=["image", "image+ref", "image+ref+cam"],
                   help="ABLATION E. What G_theta is conditioned on. The "
                        "residual base stays r_i in every setting, so this "
                        "isolates conditioning from parameterisation.")
    g.add_argument("--gen_residual", default="logit",
                   choices=["logit", "clip", "none", "csf"],
                   help="logit: p=sigmoid(logit_seed(r)+D) — matches spec.py's "
                        "sigmoid parameterisation, no dead gradients at 0/1. "
                        "clip: p=clamp(r+tanh(D),0,1). none: p=sigmoid(D), "
                        "ignoring r as a base.")
    g.add_argument("--gen_residual_scale", type=float, default=1.0)
    g.add_argument("--gen_noise_dim", type=int, default=0,
                   help="z_i channels. 0 = deterministic generation (default). "
                        "Evaluation always uses the prior mean (zeros) so the "
                        "reported test-time patch is reproducible.")

    # ── perceptual constraint (--gen_residual csf) ───────────────────────────
    add_csf_args(p)

    # ── sensitivity map ──────────────────────────────────────────────────────
    add_cam_args(p)

    # ── placement ────────────────────────────────────────────────────────────
    g.add_argument("--gen_placement", default="gradcam",
                   choices=["center", "gradcam", "semantic", "fixed"],
                   help="ABLATION A vs B. gradcam = argmax over MeanPool(M_i). "
                        "This flag is READ ONLY by the conditional-generator "
                        "scripts; --placement for the existing patch modes is "
                        "untouched.")
    g.add_argument("--gen_placement_class", type=int, default=0)
    g.add_argument("--gen_placement_xy", type=float, nargs=2,
                   default=[0.5, 0.5])
    g.add_argument("--gen_placement_margin", type=int, default=0,
                   help="keep the gradcam window this many px from every image "
                        "border. DEFAULT 0 reproduces the unmargined argmax. "
                        "The sensitivity map's hottest ridge is the near-field "
                        "road boundary along the bottom of a dashcam frame, so "
                        "the argmax tends to pin the patch flush against the "
                        "edge, where ~half its receptive field lies outside "
                        "the image. Try p/2 (64 at the default geometry).")

    # ── reference source ─────────────────────────────────────────────────────
    g.add_argument("--gen_reference", default="center",
                   choices=["center", "window"],
                   help="where r_i is sampled. 'center' is the centre crop. "
                        "'window' samples the content the patch REPLACES, "
                        "removing the perspective mismatch between a "
                        "mid-distance reference and a near-field destination. "
                        "With 'window' baseline A becomes a literal no-op, so "
                        "every point of degradation is the generator's.")

    # ── LAP constraint ───────────────────────────────────────────────────────
    g.add_argument("--gen_lap_alpha", type=float, default=0.0,
                   help="ABLATION C vs D. L_rat weight. DEFAULT 0: L_rat and "
                        "L_tv are SUMS (1e2-1e4) while the attack losses are "
                        "per-pixel MEANS (1e-2 to 20). Read the step-1 "
                        "magnitude report before setting this.")
    g.add_argument("--gen_lap_beta", type=float, default=0.0)
    g.add_argument("--gen_lap_gamma", type=float, default=0.0)
    return p


def build_generator_config(a):
    from patchreach.patch.conditional_generator import GeneratorConfig
    return GeneratorConfig(
        size=a.patch_size, base_ch=a.gen_base_ch, depth=a.gen_depth,
        cond=a.gen_cond, residual=a.gen_residual,
        residual_scale=a.gen_residual_scale,
        noise_dim=a.gen_noise_dim,
        csf_threshold=a.csf_threshold, csf_model=a.csf_model,
        csf_beta=a.csf_beta, csf_min_cycles=a.csf_min_cycles,
        csf_pixel_size_cm=a.csf_pixel_size_cm,
        csf_viewing_distance_cm=a.csf_viewing_distance_cm).validate()


def setup_model(a):
    """(model, n_channels, n_active, spec). Prints the identity checks."""
    if not getattr(a, "cityscapes_root", None):
        raise SystemExit("--cityscapes_root is required (or set it in --config)")
    cfg, weights, spec = resolve(a.arch, a.cfg_path, a.weights)
    print(f"\n[arch] {a.arch} — {spec.family}")
    print(f"[cfg ] {cfg}")
    print(f"[ckpt] {weights}")
    device = get_device()
    model = WrappedSegModel(load_segmentor(cfg, weights)[0]).to(device)
    n_ch, n_act = channel_probe(model, device, min(a.img_h, 512),
                                min(a.img_w, 1024))
    print(f"[head] {n_ch} channels ({n_act} numerically active)")
    return model, n_ch, n_act, spec


def image_indices(arg, n_val):
    """'fixed10' | 'all' | comma/space separated indices."""
    if arg in (None, "fixed10"):
        return FIXED10
    if arg == "all":
        return list(range(n_val))
    return [int(x) for x in str(arg).replace(",", " ").split()]


def build_patch(a, device, mean_t, std_t, generator=None,
                init_reference=None):
    from patchreach.patch.spec import PatchConfig, Patch
    cfg = PatchConfig(
        mode=a.patch_mode, size=a.patch_size, scale=a.patch_scale,
        shape=a.shape, placement=a.placement,
        placement_class=a.placement_class, reference=a.reference,
        placement_xy=tuple(a.placement_xy),
        placement_margin=getattr(a, "placement_margin", 0),
        reference_fit=a.reference_fit, logit_clip=a.logit_clip, shape_bg=a.shape_bg, shape_thresh=a.shape_thresh,
        lap_alpha=a.lap_alpha, lap_beta=a.lap_beta, lap_gamma=a.lap_gamma,
        lap_edge_thresh=a.lap_edge_thresh,
        lap_freeze_edges=a.lap_freeze_edges, init_from=a.init_from,
        csf_threshold=a.csf_threshold, csf_model=a.csf_model,
        csf_beta=a.csf_beta, csf_min_cycles=a.csf_min_cycles,
        csf_pixel_size_cm=a.csf_pixel_size_cm,
        csf_viewing_distance_cm=a.csf_viewing_distance_cm,
        csf_display_peak_cd_m2=getattr(a, "csf_display_peak_cd_m2", 100.0),
        csf_lref=getattr(a, "csf_lref", 0.0),
        csf_composite=getattr(a, "csf_composite", "clip"))
    return Patch(cfg, device, mean_t, std_t, generator=generator,
                 init_reference=init_reference)