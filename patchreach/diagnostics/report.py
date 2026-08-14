r"""
Diagnostic orchestrator + every figure the thesis needs.

Dispatches on loss_fn: targeted runs get the semantic suite (margin, target
probability, contestability), untargeted runs get the confusion suite. The
geometric suite runs for BOTH, because it is the shared control.

Every figure is written as a STANDALONE png with its own title and legend, so
any single one drops into the thesis without cropping a composite. A combined
`panel.png` and `overview.png` exist for quick inspection, not for publication.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data.cityscapes import (PALETTE, blend, class_name, denormalise,
                               label_to_colour, upsample_to)
from ..metrics.curves import (targeted_reach, untargeted_reach, print_curve,
                              collapse_point)
from ..metrics.miou import SegMetric
from . import geometric, semantic, untargeted


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _plt().close(fig)


def _legend_handles(classes, target_class=None):
    """Cityscapes colour key, restricted to classes actually in the image."""
    import matplotlib.patches as mpatches
    return [mpatches.Patch(facecolor=PALETTE[c].tolist(), edgecolor="none",
                           label=class_name(c)
                           + (" *" if c == target_class else ""))
            for c in classes if c < 19]


def _add_legend(ax, handles, ncol=3, loc="lower left"):
    ax.legend(handles=handles, loc=loc, fontsize=7, framealpha=0.85, ncol=ncol,
              handlelength=1.2, handleheight=0.9, borderpad=0.5,
              labelspacing=0.3, columnspacing=0.8)


# ═════════════════════════════════════════════════════════════════════════════
#  Panels
# ═════════════════════════════════════════════════════════════════════════════

def save_panels(img, label, patched, clean_logits, adv_logits, footprint,
                patch, mean_t, std_t, out_dir: Path, target_class=None):
    """
    a_clean, b_clean_prediction, c_patched, d_adv_prediction, e_change_map,
    f_ground_truth, patch, panel.png

    PANEL (b) IS THE CLEAN PREDICTION, NOT THE GROUND TRUTH. The attack's effect
    is (clean pred -> adv pred). Scoring the panel against GT folds the model's
    own errors into what reads as attack damage — which matters when per-image
    clean mIoU can sit near 50. GT is still written as (f) for reference.

    The change map colours flipped pixels by DISTANCE from the patch, which
    makes perspective effects legible: in a dashcam frame the bottom of the
    image is NEARBY road, so a flood that appears to span the scene is often
    entirely within a few hundred pixels of the patch.
    """
    plt = _plt()
    from matplotlib import cm as mcm
    import matplotlib.patches as mpatches
    from torchvision.utils import save_image

    hw = label.shape[-2:]
    lc = upsample_to(clean_logits, hw)
    la = upsample_to(adv_logits, hw)
    pc, pa = lc.argmax(1)[0], la.argmax(1)[0]

    clean_photo = denormalise(img, mean_t, std_t)
    patched_photo = denormalise(patched, mean_t, std_t)
    clean_ov = blend(clean_photo, label_to_colour(pc))
    adv_ov = blend(patched_photo, label_to_colour(pa))
    gt_ov = blend(clean_photo, label_to_colour(label[0]))

    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(clean_photo, out_dir / "a_clean.png")
    save_image(patched_photo, out_dir / "c_patched.png")
    save_image(patch.render().cpu(), out_dir / "patch.png")

    # Legend covers GT + BOTH predictions, so (b) and (d) share one colour key
    # and any class the attack introduced is included.
    cls = sorted(set(label[0][label[0] != 255].unique().tolist())
                 | set(pc.cpu().unique().tolist())
                 | set(pa.cpu().unique().tolist()))
    handles = _legend_handles(cls, target_class)

    for tensor, name, title in [
            (clean_ov, "b_clean_prediction", "(b) clean prediction"),
            (adv_ov, "d_adv_prediction", "(d) adversarial prediction"),
            (gt_ov, "f_ground_truth", "(f) ground truth")]:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(tensor.permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        _add_legend(ax, handles, ncol=4)
        _save(fig, out_dir / f"{name}.png")

    # ── change map, coloured by distance ─────────────────────────────────────
    from ..losses.reach import centroid, distance_map
    H, W = hw
    dist = distance_map(H, W, *centroid(footprint), pc.device)
    remote = (label[0] != 255) & (~footprint[0])
    changed = (pc != pa) & remote
    rgb = np.zeros((H, W, 3))
    rgb[remote.cpu().numpy()] = [0.85, 0.85, 0.85]
    rgb[footprint[0].cpu().numpy()] = [0.25, 0.25, 0.25]
    max_d = dist[remote].max().item() if remote.any() else 1.0
    ch = changed.cpu().numpy()
    if ch.any():
        nd = np.clip(dist.cpu().numpy()[ch] / max(max_d, 1), 0, 1)
        rgb[ch] = mcm.plasma(1.0 - nd)[:, :3]
    flip_pct = 100.0 * changed.sum().item() / max(int(remote.sum()), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(f"(e) prediction changes — {flip_pct:.1f}% of remote pixels\n"
                 "colour = distance from patch, grey = unchanged, "
                 "dark = footprint", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, max_d))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, label="distance (px)")
    _save(fig, out_dir / "e_change_map.png")

    # ── combined, shared legend strip underneath ─────────────────────────────
    fig = plt.figure(figsize=(30, 8))
    gs = fig.add_gridspec(2, 5, height_ratios=[6, 1], hspace=0.05, wspace=0.04)
    for i, (t, ttl) in enumerate([
            (clean_photo, "(a) clean"), (clean_ov, "(b) clean prediction"),
            (patched_photo, "(c) patched"), (adv_ov, "(d) adv. prediction")]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(t.permute(1, 2, 0).numpy())
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(rgb)
    ax.set_title(f"(e) change map — {flip_pct:.1f}%", fontsize=10)
    ax.axis("off")
    axl = fig.add_subplot(gs[1, :])
    axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=min(len(handles), 12),
               fontsize=8, frameon=False)
    _save(fig, out_dir / "panel.png")
    return flip_pct


# ═════════════════════════════════════════════════════════════════════════════
#  Figures
# ═════════════════════════════════════════════════════════════════════════════

def _iou_figure(clean_iou, adv_iou, present, out_path, target_class=None,
                title=None):
    """Per-class IoU, clean vs patched. Bars in the class's own palette colour."""
    plt = _plt()
    names = [class_name(c) for c in present]
    c_vals = [clean_iou[c].item() for c in present]
    a_vals = [adv_iou[c].item() for c in present]
    x = np.arange(len(present))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(present) * 0.8), 4.5))
    ax.bar(x - w / 2, c_vals, w, label="clean", color="#4c72b0", alpha=0.9)
    ax.bar(x + w / 2, a_vals, w, label="patched", color="#dd8452", alpha=0.9)
    for i, (c, a) in enumerate(zip(c_vals, a_vals)):
        if c - a > 2:
            ax.annotate(f"-{c-a:.0f}", (x[i] + w / 2, a), fontsize=7,
                        ha="center", va="bottom", color="red")
    if target_class is not None and target_class in present:
        ax.axvline(present.index(target_class), color="red", ls="--", lw=1.2,
                   alpha=0.5, label="target class")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-class IoU: clean vs patched"
                 + (f" — {title}" if title else ""))
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, out_path)


def per_class_iou_figure(clean_logits, adv_logits, label, K, out_path,
                         target_class=None, title=None, log=None):
    """
    Per-class IoU chart, clean vs patched, plus the numbers as a dict.

    Public wrapper over _iou_figure that does the metric computation too, so a
    caller holding only logits does not have to build SegMetrics itself.

    Returns {class_name: {"clean": .., "adv": .., "drop": ..}} — aggregate mIoU
    hides WHICH classes the attack destroyed, and that per-class breakdown is
    what reveals structured single-channel collapse (e.g. one class losing ~68
    IoU while everything else moves by <5).
    """
    import torch
    hw = label.shape[-2:]
    mc = SegMetric(K, device=label.device)
    ma = SegMetric(K, device=label.device)
    mc.update(upsample_to(clean_logits, hw).argmax(1), label)
    ma.update(upsample_to(adv_logits, hw).argmax(1), label)
    ciou, aiou = mc.per_class(), ma.per_class()

    present = [c for c in sorted(label[label != 255].unique().tolist()) if c < K]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _iou_figure(ciou, aiou, present, out_path, target_class, title=title)

    out = {class_name(c): {"clean": float(ciou[c]), "adv": float(aiou[c]),
                           "drop": float(ciou[c] - aiou[c])} for c in present}
    if log is not None:
        log("\n[iou] per-class, clean -> patched:")
        for c in present:
            log(f"    {c:2d} {class_name(c):10s}: {ciou[c]:6.2f} -> "
                f"{aiou[c]:6.2f}  ({aiou[c]-ciou[c]:+.1f})")
    return out


def _reach_figure(curves, out_path, title="Attack reach"):
    """curves: list of (label, distances, rates)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lbl, d, r in curves:
        ax.plot(d, r, "-o", label=lbl, markersize=4)
    ax.axhline(5, color="gray", ls=":", lw=0.8, label="5% threshold")
    ax.set_xlabel("distance from patch centre (px)")
    ax.set_ylabel("% of ring pixels")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{title}\nplateau height = the contestable fraction of the "
                 "scene, not raw attack power")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, out_path)


def _margin_figure(mc, ma, delta, out_path):
    plt = _plt()
    vmax = max(np.percentile(mc.numpy(), 99), np.percentile(ma.numpy(), 99), 1e-6)
    vlim = max(np.percentile(np.abs(delta.numpy()), 99), 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, m, t, cmap, lo, hi in [
            (ax[0], mc.numpy(), "winner margin — clean\n(bright = confident)",
             "viridis", 0, vmax),
            (ax[1], ma.numpy(), "winner margin — patched", "viridis", 0, vmax),
            (ax[2], delta.numpy(), "delta\nblue = confidence eroded",
             "RdBu_r", -vlim, vlim)]:
        im = a.imshow(m, cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(t, fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.025)
    plt.suptitle("max_logit - second_max_logit — the most sensitive reach "
                 "measurement:\nthe blue region extends PAST the flip zone "
                 "(influence without completed flips)", fontsize=10)
    plt.tight_layout()
    _save(fig, out_path)


def _entropy_figure(ec, ea, out_path):
    plt = _plt()
    d = (ea - ec).numpy()
    vlim = max(np.percentile(np.abs(d), 99), 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, m, t, cmap, lo, hi in [
            (ax[0], ec.numpy(), "entropy — clean", "YlOrRd", 0, 1),
            (ax[1], ea.numpy(), "entropy — patched", "YlOrRd", 0, 1),
            (ax[2], d, "delta\nred = more confused", "RdBu_r", -vlim, vlim)]:
        im = a.imshow(m, cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(t, fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.025)
    plt.suptitle("Normalised prediction entropy — uncertainty across ALL "
                 "classes\n(the winner margin measures only the top-2 gap; "
                 "these can disagree)", fontsize=10)
    plt.tight_layout()
    _save(fig, out_path)


def _flip_rate_figure(rates, out_path):
    """Per-class flip rate, bars in each class's own palette colour."""
    if not rates:
        return
    plt = _plt()
    names = list(rates)
    vals = [rates[n] for n in names]
    idx = [next((i for i in range(19) if class_name(i) == n), None)
           for n in names]
    cols = [PALETTE[i].tolist() if i is not None else [.5, .5, .5] for i in idx]

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.8), 4.5))
    ax.bar(np.arange(len(names)), vals, color=cols, edgecolor="none", alpha=0.9)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("% of that class's remote pixels flipped")
    ax.set_title("Per-class flip rate — which classes did the attack disrupt?")
    plt.tight_layout()
    _save(fig, out_path)


def _flows_figure(flows, out_path):
    """
    Top source->destination flows. A single dominant bar means the untargeted
    loss is acting as an implicit class selector rather than a broad disruptor.
    """
    if not flows:
        return
    plt = _plt()
    lbls = [f"{s} -> {d}" for _, s, d in flows][::-1]
    vals = [n for n, _, _ in flows][::-1]
    total = sum(n for n, _, _ in flows)

    fig, ax = plt.subplots(figsize=(9, max(3, len(lbls) * 0.4)))
    ax.barh(np.arange(len(lbls)), vals, color="#dd8452", alpha=0.9)
    ax.set_yticks(np.arange(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=8)
    ax.set_xlabel("flipped pixels")
    top = 100 * vals[-1] / max(total, 1)
    ax.set_title(f"Top prediction flows — the dominant channel carries "
                 f"{top:.0f}% of these flips\n"
                 "one dominant bar = implicit class selection")
    plt.tight_layout()
    _save(fig, out_path)


def panels_for_images(model, dataset, indices, patch, out_dir, mean_t, std_t,
                      K: int = 19, target_class=None, img_h=None, img_w=None,
                      log=print):
    """
    Labelled panels + a per-class IoU chart for several images.

    Cheap: two forward passes per image and no ERF probe, so it is affordable
    at the end of every training run. A finished run should leave FIGURES
    behind, not only JSON — the numbers say how much degraded, the panels say
    where and into what.

    Placement is re-resolved per image, because semantic placement depends on
    scene content and a cached offset from another image would be wrong.

    Returns {image_index: {"flip_pct": ..., "per_class_iou": {...}}}.
    """
    import torch
    out_dir = Path(out_dir)
    out = {}

    for i in indices:
        img, label = dataset[i]
        img = img.unsqueeze(0).to(mean_t.device)
        label = label.unsqueeze(0).to(mean_t.device)
        hw = label.shape[-2:]

        with torch.no_grad():
            clean = upsample_to(model(img), hw)
            if img_h is not None:
                patch.resolve_placement(img_h, img_w, clean.argmax(1)[0])
            patched, fp = patch.apply(img)
            adv = upsample_to(model(patched), hw)

        flip = save_panels(img, label, patched, clean, adv, fp, patch,
                           mean_t, std_t, out_dir / f"img{i}", target_class)

        mc, ma = SegMetric(K, device=img.device), SegMetric(K, device=img.device)
        mc.update(clean.argmax(1), label)
        ma.update(adv.argmax(1), label)
        ciou, aiou = mc.per_class(), ma.per_class()
        present = [c for c in sorted(label[label != 255].unique().tolist())
                   if c < K]
        _iou_figure(ciou, aiou, present, out_dir / f"img{i}" / "per_class_iou.png",
                    target_class)

        out[i] = {"flip_pct": flip,
                  "per_class_iou": {class_name(c): {"clean": float(ciou[c]),
                                                    "adv": float(aiou[c])}
                                    for c in present}}
        log(f"  img {i:4d}: {flip:5.1f}% of remote pixels flipped "
            f"-> {out_dir / f'img{i}'}/")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run(model, img, label, patch, out_dir, loss_fn: str, K: int = 19,
        target_class=None, mean_t=None, std_t=None, n_probes: int = 16,
        skip_geometric: bool = False, log=print):
    """
    Full diagnostic suite on ONE image. Writes panels, figures and
    diagnostics.json.

    skip_geometric: the ERF probe costs n_probes forward passes; skip it when
    looping over many images and you only need it once.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hw = label.shape[-2:]

    with torch.no_grad():
        clean_logits = upsample_to(model(img), hw)
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), hw)

    lines = []

    def L(s):
        log(s)
        lines.append(str(s))

    L("=" * 66)
    L(f"  DIAGNOSTICS — loss_fn={loss_fn}  patch_mode={patch.cfg.mode}  "
      f"scale={patch.cfg.scale}")
    L("=" * 66)

    res = {"loss_fn": loss_fn, "target_class": target_class}

    if mean_t is not None:
        res["flip_pct"] = save_panels(img, label, patched, clean_logits,
                                      adv_logits, fp, patch, mean_t, std_t,
                                      out_dir / "panels", target_class)

    # ── per-class IoU, clean vs patched ──────────────────────────────────────
    mc, ma = SegMetric(K, device=img.device), SegMetric(K, device=img.device)
    mc.update(clean_logits.argmax(1), label)
    ma.update(adv_logits.argmax(1), label)
    ciou, aiou = mc.per_class(), ma.per_class()
    present = [c for c in sorted(label[label != 255].unique().tolist()) if c < K]
    _iou_figure(ciou, aiou, present, out_dir / "per_class_iou.png", target_class)
    L("\n[iou] per-class, clean -> patched:")
    for c in present:
        L(f"    {c:2d} {class_name(c):10s}: {ciou[c]:6.2f} -> {aiou[c]:6.2f}  "
          f"({aiou[c]-ciou[c]:+.1f})")
    res["per_class_iou"] = {class_name(c): {"clean": float(ciou[c]),
                                            "adv": float(aiou[c])}
                            for c in present}

    # ── geometric: the shared control ────────────────────────────────────────
    if not skip_geometric:
        _, stats = geometric.receptive_field(model, img, patch, n_probes, log=L)
        geometric.plot_erf(stats, out_dir / "receptive_field.png")
        res["receptive_field"] = [{"lo": lo, "hi": hi, "rate": r}
                                  for lo, hi, r in stats]

    curves = []

    # ── regime-specific ──────────────────────────────────────────────────────
    if loss_fn == "ipatch_cospgd" and target_class is not None:
        d, r = targeted_reach(adv_logits, target_class, fp)
        print_curve(d, r, f"reach — % predicted class {target_class}", L)
        curves.append((f"-> class {target_class} (targeted)", d, r))
        res["reach_targeted"] = {"d": d, "r": r,
                                 "collapse_px": collapse_point(d, r)}
        res["margin"] = semantic.class_margin(clean_logits, target_class, fp,
                                              K, log=L)
        res["target_prob"] = semantic.target_probability(
            clean_logits, adv_logits, target_class, fp, log=L)
    else:
        rates, flows = untargeted.confusion(clean_logits, adv_logits, label,
                                            fp, K, log=L)
        _flip_rate_figure(rates, out_dir / "flip_rate_by_class.png")
        _flows_figure(flows, out_dir / "prediction_flows.png")
        res["flip_rate_by_class"] = rates
        res["top_flows"] = flows
        res["reach_by_source"] = untargeted.reach_by_source_class(
            clean_logits, adv_logits, label, fp, K, log=L)

    res["contestability"] = semantic.contestability(clean_logits, fp, K, log=L)

    d, r = untargeted_reach(clean_logits, adv_logits, fp)
    print_curve(d, r, "reach — any-prediction-change", L)
    curves.append(("any change", d, r))
    res["reach_any"] = {"d": d, "r": r, "collapse_px": collapse_point(d, r)}
    _reach_figure(curves, out_dir / "reach_curve.png")

    wmc, wma, delta, ring = untargeted.winner_margin(clean_logits, adv_logits,
                                                     fp, K, log=L)
    _margin_figure(wmc, wma, delta, out_dir / "winner_margin.png")
    res["winner_margin"] = [{"lo": lo, "hi": hi, "clean": c, "adv": a}
                            for lo, hi, c, a in ring]

    ec, ea, ent_ring = untargeted.entropy(clean_logits, adv_logits, fp, K, log=L)
    _entropy_figure(ec, ea, out_dir / "entropy.png")
    res["entropy"] = [{"lo": lo, "hi": hi, "clean": c, "adv": a}
                      for lo, hi, c, a in ent_ring]

    (out_dir / "diagnostics.txt").write_text("\n".join(lines))
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(res, f, indent=2)

    log(f"\n  figures -> {out_dir}/")
    for n in ("panels/panel.png", "panels/d_adv_prediction.png",
              "panels/e_change_map.png", "per_class_iou.png",
              "reach_curve.png", "winner_margin.png", "entropy.png"):
        if (out_dir / n).exists():
            log(f"    {n}")
    return res