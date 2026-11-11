"""The single-image attack procedure, in ONE place."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .. import optim as optim_mod
from ..data.cityscapes import upsample_to
from ..losses import adversarial
from ..metrics.miou import (SegMetric, single_image_miou, attack_rates,
                            compare as miou_compare)
from .lap import magnitude_report

INTERMEDIATE_DIR = "intermediate_patches"


def prepare(model, img, patch, img_h: int, img_w: int,
            from_image: bool = False, mean_t=None, std_t=None,
            cam=None, label=None, log=None) -> torch.Tensor:
    """Clean forward -> sensitivity map -> resolve_placement -> image-derived base."""
    score_map = None
    if patch.cfg.placement == "gradcam":
        if cam is None:
            raise ValueError(
                "placement='gradcam' needs a SegmentationCAM. Build one with "
                "segmentation_cam.build(model, ...) and pass cam=. Centring "
                "silently here would report a gradcam run that never ran.")
        if label is None:
            raise ValueError(
                "placement='gradcam' needs `label` — the CAM uses its shape, "
                "and cam_target='gt' uses its values.")
        cam_map, logits = cam(img, label)
        clean_logits = upsample_to(logits, (img_h, img_w))
        score_map = cam_map[0, 0]
    else:
        with torch.no_grad():
            clean_logits = upsample_to(model(img), (img_h, img_w))

    patch.resolve_placement(img_h, img_w, clean_logits.argmax(1)[0],
                            score_map=score_map)
    if from_image:
        patch.set_reference_from_image(img, mean_t, std_t)
        if log:
            log(f"[patch] base      : image region at {patch.placement} "
                f"(--from_image)")
    return clean_logits


def attack_image(model, img, label, patch, *,
                 loss_fn: str = "cospgd",
                 target_class: int = 8,
                 steps: int = 300,
                 lr: float = 0.01,
                 optimiser: str = "adam",
                 num_classes: int = 19,
                 exclude_footprint: bool = True,
                 log_every: int = 20,
                 lr_schedule: str = "none",
                 classes: str = "gt",
                 margin_kappa: float = 5.0,
                 clean_logits: Optional[torch.Tensor] = None,
                 out_dir: Optional[Path] = None,
                 save_step_images: bool = False,
                 save_best: bool = True,
                 verbose: bool = True,
                 log=print) -> dict:
    """Optimise `patch` against ONE image."""
    hw = label.shape[-2:]
    if clean_logits is None:
        with torch.no_grad():
            clean_logits = upsample_to(model(img), hw)

    _, fp0 = patch.apply(img)
    clean_all = single_image_miou(clean_logits, label, num_classes,
                                  classes=classes)
    clean_rem = single_image_miou(clean_logits, label, num_classes,
                                  exclude=fp0, classes=classes)
    if verbose:
        log(f"\n[clean] all {clean_all:.2f} | remote {clean_rem:.2f}  "
            f"<- remote is the baseline that matters")

    objective = adversarial.build(
        loss_fn, target_class if loss_fn == "ipatch_cospgd" else 8)
    if loss_fn == "tsallis":
        objective = adversarial.build(
            loss_fn, 8,
            tsallis_q=patch.cfg.tsallis_q,
            tsallis_schedule=patch.cfg.tsallis_schedule,
            tsallis_q_start=patch.cfg.tsallis_q_start,
            tsallis_q_end=patch.cfg.tsallis_q_end,
            tsallis_total_steps=steps)
        if verbose:
            log(f"[loss ] {objective!r}")
    if loss_fn in ("margin", "cos_margin"):
        ref = upsample_to(clean_logits, hw).argmax(1)
        ref[label == 255] = 255
        objective = adversarial.build(loss_fn, margin_ref=ref,
                                      margin_kappa=margin_kappa)
        if verbose:
            log(f"[loss ] {loss_fn} vs clean prediction, "
                f"kappa={margin_kappa:g}")
    opt = optim_mod.build(optimiser, [patch.param], lr,
                          betas=(0.9, 0.999))
    if verbose and optimiser != "adam":
        log(f"[optim] {optimiser} — step is exactly lr per coordinate")
    if lr_schedule not in ("none", "cosine"):
        raise ValueError(f"lr_schedule must be 'none' or 'cosine', "
                         f"got {lr_schedule!r}")
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
             if lr_schedule == "cosine" else None)
    if verbose and sched is not None:
        log(f"[sched] cosine, lr {lr:g} -> 0 over {steps} steps")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    history, best_drop = [], -1e9
    la = None
    for step in range(1, steps + 1):
        if hasattr(objective, "on_step_begin"):
            objective.on_step_begin(step - 1, steps)
        opt.zero_grad()
        patched, fp = patch.apply(img)
        logits = upsample_to(model(patched), hw)
        sup = (~fp) if exclude_footprint else None
        la = objective(logits, label, fp, sup)
        if not torch.isfinite(la):
            raise RuntimeError(f"non-finite loss at step {step}")
        extra = patch.regularisers()
        (la + extra["total"]).backward()

        if step == 1 and verbose:
            g = patch.param.grad
            log(f"[grad] {'None — GRAPH BROKEN' if g is None else f'abs mean {g.abs().mean():.3e}'}")
            if patch.cfg.mode == "lap":
                magnitude_report(la.item(), patch.render(), patch.reference,
                                 patch.active_mask(), patch.cfg, log=log)

        opt.step()
        patch.project()
        if sched is not None:
            sched.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                cur = single_image_miou(logits.detach(), label, num_classes,
                                        exclude=fp, classes=classes)
            st = patch.stats()
            if verbose:
                log(f"  step {step:4d}  remote={cur:6.2f} "
                    f"({clean_rem-cur:+6.2f})  {loss_fn}={la.item():.4f}  "
                    + " ".join(f"{k}={v:.4f}" for k, v in st.items()))
            history.append({"step": step, "miou_remote": cur,
                            "loss": la.item(), **st})
            if clean_rem - cur > best_drop:
                best_drop = clean_rem - cur
                if out_dir is not None and save_best:
                    patch.save(out_dir / "best.pt")
                    _save_png(patch, out_dir / "best_patch.png")
            if out_dir is not None and save_step_images:
                step_dir = out_dir / INTERMEDIATE_DIR
                step_dir.mkdir(parents=True, exist_ok=True)
                _save_png(patch, step_dir / f"patch_step{step:04d}.png")

    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), hw)

        pc = upsample_to(clean_logits, hw).argmax(1)
        pa = adv_logits.argmax(1)
        mc_all, ma_all = (SegMetric(num_classes, device=logits.device),
                          SegMetric(num_classes, device=logits.device))
        mc_rem, ma_rem = (SegMetric(num_classes, device=logits.device),
                          SegMetric(num_classes, device=logits.device))
        mc_all.update(pc, label)
        ma_all.update(pa, label)
        mc_rem.update(pc, label, exclude=fp)
        ma_rem.update(pa, label, exclude=fp)
        cmp_all = miou_compare(mc_all, ma_all)
        cmp_rem = miou_compare(mc_rem, ma_rem)

        final_all = single_image_miou(adv_logits, label, num_classes,
                                      classes=classes)
        final_rem = single_image_miou(adv_logits, label, num_classes,
                                      exclude=fp, classes=classes)
        rates = attack_rates(clean_logits, adv_logits, label, fp,
                             target_class if loss_fn == "ipatch_cospgd"
                             else None)

    final_stats = {f"final_{k}": v for k, v in patch.stats().items()}

    degraded = best_drop > (clean_rem - final_rem) + 1.0
    if degraded and verbose:
        hint = ("frac_at_bound in the history (-> 1 means the spectral "
                "allocation\n  is frozen and the run can only rotate phase; "
                "under --csf_param squash that is\n  expected and --csf_param "
                "pgd is the fix)"
                if patch.cfg.mode in ("csf", "universal_csf")
                else "frac_at_clip in the history for sigmoid saturation")
        log(f"\n  WARNING: best drop was {best_drop:+.2f} but the FINAL "
            f"patch only reaches {clean_rem-final_rem:+.2f}.\n"
            f"  The run degraded after its peak — check {hint},\n"
            f"  and use best.pt rather than final.pt.")

    H, W = hw
    p_side = patch.side(H, W)
    top, left = (patch.placement if patch.placement is not None
                 else ((H - p_side) // 2, (W - p_side) // 2))
    ctop, cleft = (H - p_side) // 2, (W - p_side) // 2
    on_border = int(top) in (0, H - p_side) or int(left) in (0, W - p_side)

    return {"clean_all": clean_all, "clean_remote": clean_rem,
            "final_all": final_all, "final_remote": final_rem,
            "drop_all": clean_all - final_all,
            "drop_remote": clean_rem - final_rem,
            "best_drop_remote": best_drop,
            "degraded_after_peak": bool(degraded),
            "last_loss": float(la.item()) if la is not None else None,
            "placement": [int(top), int(left)],
            "placement_policy": patch.cfg.placement,
            "placement_dist_from_centre": float(
                ((top - ctop) ** 2 + (left - cleft) ** 2) ** 0.5),
            "placement_on_border": bool(on_border),
            "classes": classes,
            "drop_all_union": cmp_all["drop_union"],
            "drop_remote_union": cmp_rem["drop_union"],
            "n_classes_gt": cmp_rem["n_classes_gt"],
            "n_classes_clean_union": cmp_rem["n_classes_clean_union"],
            "n_classes_adv_union": cmp_rem["n_classes_adv_union"],
            "class_set_moved": bool(cmp_rem["n_classes_clean_union"]
                                    != cmp_rem["n_classes_adv_union"]),
            "lr_schedule": lr_schedule,
            "optimiser": optimiser,
            "pixel_param": patch.cfg.pixel_param,
            **({"tsallis_q": objective.q} if loss_fn == "tsallis" else {}),
            **final_stats, **rates, "history": history}


def _save_png(patch, path):
    from torchvision.utils import save_image
    save_image(patch.render().detach().cpu(), path)
