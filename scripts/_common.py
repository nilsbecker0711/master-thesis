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
                   choices=["raw", "gan", "raw_ganinit", "lap"])
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
                   choices=["center", "fixed", "semantic"])
    p.add_argument("--placement_class", type=int, default=0,
                   help="trainId for --placement semantic. 0=road is "
                        "omnipresent in Cityscapes; 1=sidewalk is absent from "
                        "many frames and would silently fall back to centre.")
    p.add_argument("--placement_xy", type=float, nargs=2, default=[0.5, 0.5],
                   help="normalised (y, x) CENTRE for --placement fixed. "
                        "(0.75, 0.5) is the road surface straight ahead.")
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
    return p


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


def build_patch(a, device, mean_t, std_t, generator=None):
    from patchreach.patch.spec import PatchConfig, Patch
    cfg = PatchConfig(
        mode=a.patch_mode, size=a.patch_size, scale=a.patch_scale,
        shape=a.shape, placement=a.placement,
        placement_class=a.placement_class, reference=a.reference,
        placement_xy=tuple(a.placement_xy),
        reference_fit=a.reference_fit, logit_clip=a.logit_clip, shape_bg=a.shape_bg, shape_thresh=a.shape_thresh,
        lap_alpha=a.lap_alpha, lap_beta=a.lap_beta, lap_gamma=a.lap_gamma,
        lap_edge_thresh=a.lap_edge_thresh,
        lap_freeze_edges=a.lap_freeze_edges, init_from=a.init_from)
    return Patch(cfg, device, mean_t, std_t, generator=generator)