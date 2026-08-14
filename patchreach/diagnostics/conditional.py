r"""
Visualisation for the image-conditioned generator.

The question a number cannot answer: does the generator learn CONTEXTUAL
REALISM, or does it just paint a conspicuous adversarial texture over the
reference? Only the side-by-side shows that, and it has to be looked at
regularly rather than once at the end — a run whose drop_remote climbs while
the patch degenerates into noise is a result about the attack loss, not about
the hypothesis.

Panels per image:
    a_clean            original x_i
    b_reference        r_i = Resize(CenterCrop(x_i))
    c_sensitivity      M_i, with the chosen placement window drawn on it
    d_patch            p_i = G_theta(...)
    e_patched          x_i^adv
    f_clean_prediction argmax f(x_i)
    g_adv_prediction   argmax f(x_i^adv)
    panel.png          all of the above in one strip

Existing figure machinery is reused wherever it applies (label_to_colour,
blend, denormalise). save_panels() in report.py is NOT reused directly: it
takes a `patch` object and calls patch.render(), which assumes one patch for
the batch — the assumption this attack family exists to break.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch

from ..data.cityscapes import blend, denormalise, label_to_colour, upsample_to


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def heatmap_overlay(img01: torch.Tensor, cam: torch.Tensor,
                    alpha: float = 0.5) -> torch.Tensor:
    """
    [3,H,W] photo blended with M_i under `inferno`.

    Uses the same blend() the segmentation overlays use, so the sensitivity
    panel is visually comparable with the prediction panels rather than being
    on its own arbitrary scale.
    """
    from matplotlib import cm as mcm
    m = cam.detach().float().clamp(0, 1).cpu().numpy()
    rgb = torch.from_numpy(mcm.inferno(m)[..., :3]).permute(2, 0, 1).float()
    return blend(img01.cpu(), rgb, alpha)


def save_conditional_panels(img, label, patched, patch01, reference01, cam,
                            clean_logits, adv_logits, placement, p,
                            mean_t, std_t, out_dir: Path,
                            target_class: Optional[int] = None,
                            title: str = ""):
    """
    One image's full story. Every tensor is a SINGLE-image slice:
        img/patched [1,3,H,W] normalised   patch01/reference01 [3,S,S] in [0,1]
        cam [H,W] in [0,1]                 label [1,H,W]
    """
    plt = _plt()
    import matplotlib.patches as mpatches
    from torchvision.utils import save_image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = label.shape[-2:]
    pc = upsample_to(clean_logits, hw).argmax(1)[0]
    pa = upsample_to(adv_logits, hw).argmax(1)[0]

    clean_photo = denormalise(img, mean_t, std_t)
    adv_photo = denormalise(patched, mean_t, std_t)
    clean_ov = blend(clean_photo, label_to_colour(pc))
    adv_ov = blend(adv_photo, label_to_colour(pa))
    heat = heatmap_overlay(clean_photo, cam)

    save_image(clean_photo, out_dir / "a_clean.png")
    save_image(reference01.detach().cpu(), out_dir / "b_reference.png")
    save_image(heat, out_dir / "c_sensitivity.png")
    save_image(patch01.detach().clamp(0, 1).cpu(), out_dir / "d_patch.png")
    save_image(adv_photo, out_dir / "e_patched.png")
    save_image(clean_ov, out_dir / "f_clean_prediction.png")
    save_image(adv_ov, out_dir / "g_adv_prediction.png")

    # ── combined strip ───────────────────────────────────────────────────────
    top, left = placement
    fig = plt.figure(figsize=(26, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.12, wspace=0.05)

    wide = [(clean_ov, "(f) clean prediction"), (heat, "(c) sensitivity M_i"),
            (adv_photo, "(e) patched"), (adv_ov, "(g) adv. prediction")]
    for i, (t, ttl) in enumerate(wide):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(t.permute(1, 2, 0).numpy())
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
        if i in (1, 2):
            # Draw the footprint so placement is auditable by eye. A gradcam
            # placement that always lands in the same corner is a bug, and it
            # is invisible in the numbers.
            ax.add_patch(mpatches.Rectangle((left, top), p, p, fill=False,
                                            edgecolor="cyan", lw=1.8))

    small = [(clean_photo, "(a) clean"),
             (reference01.detach().cpu(), "(b) reference r_i"),
             (patch01.detach().clamp(0, 1).cpu(), "(d) generated patch p_i")]
    for i, (t, ttl) in enumerate(small):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(t.permute(1, 2, 0).numpy())
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    d = (patch01.detach().cpu() - reference01.detach().cpu())
    ax.text(0.0, 0.95,
            f"{title}\n\n"
            f"placement (top,left) = ({top}, {left})\n"
            f"patch side p = {p} px\n"
            f"|p_i - r_i|  mean {d.abs().mean():.4f}  max {d.abs().max():.4f}\n"
            f"L2(p_i, r_i) = {d.pow(2).sum().sqrt():.2f}\n\n"
            f"p_i is generated by G_theta from (x_i, r_i, M_i).\n"
            f"No test-time optimisation of p_i.",
            fontsize=10, va="top", family="monospace")

    fig.savefig(out_dir / "panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_dir


def lpips_histogram(values: Sequence[float], out_path, title="LPIPS(r_i, p_i)"):
    """
    The DISTRIBUTION, not just the mean.

    A mean hides the case this experiment most needs to detect: most patches
    barely moving while a minority blow up into high-distortion textures. Those
    are different phenomena and they average to the same number.
    """
    if not values:
        return None
    import numpy as np
    plt = _plt()
    v = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(v, bins=min(40, max(8, len(v) // 3)), color="#4c72b0", alpha=0.85)
    ax.axvline(float(v.mean()), color="red", ls="--", lw=1.4,
               label=f"mean {v.mean():.4f}")
    ax.axvline(float(np.median(v)), color="orange", ls=":", lw=1.4,
               label=f"median {np.median(v):.4f}")
    ax.set_xlabel("LPIPS distance to the centre-crop reference")
    ax.set_ylabel("images")
    ax.set_title(f"{title}\nevaluation metric only — never a training objective")
    ax.legend(fontsize=8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)),
            "p75": float(np.percentile(v, 75)),
            "n": int(v.size)}
