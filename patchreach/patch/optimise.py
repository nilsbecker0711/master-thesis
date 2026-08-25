r"""
The single-image attack procedure, in ONE place.

WHY THIS EXISTS
---------------
scripts/overfit.py owned this loop inline. The moment a second caller needed it
— running the same per-image attack over N images to get a population rather
than an anecdote — the choice was to copy forty lines or to extract them.

This repository has already paid for the copy-paste answer twice: three bugs
came from redeclaring the argument parsers in train.py and overfit.py and
updating one copy, which is why they now live in _common.py; and
composite_batch() carries a test asserting it still agrees with Patch.apply()
precisely because a mirrored implementation drifts silently. A second copy of
the optimisation loop would drift the same way, and the drift would look like a
result: the population numbers would stop matching the single-image numbers and
nothing would say why.

So overfit.py and overfit_population.py call the SAME function here. Anything
that changes the attack changes both, or neither.

WHAT IT DOES NOT DO
-------------------
No argument parsing, no model construction, no diagnostics. It takes a built
model and a built Patch, runs the attack, and returns numbers. Diagnostics are
the caller's business — which is the whole point for the population run, where
the suite is far too expensive to run on every image and is deliberately
restricted to a handful afterwards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ..data.cityscapes import upsample_to
from ..losses import adversarial
from ..metrics.miou import (SegMetric, single_image_miou, attack_rates,
                            compare as miou_compare)
from .lap import magnitude_report


def prepare(model, img, patch, img_h: int, img_w: int,
            from_image: bool = False, mean_t=None, std_t=None,
            cam=None, label=None, log=None) -> torch.Tensor:
    r"""
    Clean forward -> sensitivity map -> resolve_placement -> image-derived base.

    ORDERING IS LOAD-BEARING AND IS THE REASON THIS IS A FUNCTION. Semantic
    placement reads the CLEAN PREDICTION and gradcam placement reads the
    SENSITIVITY MAP, so neither can be resolved before the model has seen the
    image; done backwards, placement silently falls back to centre and every
    downstream number is quietly about a different patch position.
    set_reference_from_image() then has its own ordering constraint on top —
    it needs the resolved placement to know which region to copy, which is why
    --from_image and --placement gradcam compose correctly only in this order.

    cam : a segmentation_cam.SegmentationCAM, required for placement='gradcam'.
          Its forward pass also returns the clean logits, so the CAM path costs
          ONE backward rather than an extra forward — the clean pass is reused
          rather than repeated.

    Returns the clean logits, upsampled to the label resolution.
    """
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
        # Only changes behaviour for modes that treat the reference as a BASE
        # (csf); every pre-existing mode ignores it.
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
                 num_classes: int = 19,
                 exclude_footprint: bool = True,
                 log_every: int = 20,
                 lr_schedule: str = "none",
                 classes: str = "gt",
                 clean_logits: Optional[torch.Tensor] = None,
                 out_dir: Optional[Path] = None,
                 save_step_images: bool = False,
                 save_best: bool = True,
                 verbose: bool = True,
                 log=print) -> dict:
    r"""
    Optimise `patch` against ONE image. Returns the result record.

    `patch` is left holding its FINAL parameter. The BEST parameter — by remote
    mIoU drop — is written to out_dir/best.pt when `save_best`, because a run
    that saturates late otherwise destroys the result it had already achieved.
    project() exists to make that rare; best.pt exists because rare is not
    never, and one observed run reached a 17.5-point drop at step 660 and was
    inert by step 700.

    `save_step_images` writes a PNG every log_every steps. Right for a single
    interactive run, catastrophic across hundreds of images — it is off by
    default and overfit.py turns it on.

    `classes` selects which classes enter the mIoU mean, and 'gt' is the only
    safe default for an ATTACK measurement. Under 'union' a class with no
    ground truth that the model predicts anywhere scores IoU 0 and drags the
    nanmean down; suppress that prediction and the class leaves the denominator
    and the mean jumps by ~mean/(n-1). Clean and attacked are then averaged over
    different class sets and their difference is not a measurement — this is the
    bug that made a 150-epoch run report drop_remote = -3.63. The record carries
    the union figures too, so runs from before that fix stay comparable and any
    divergence stays visible rather than silent.

    `lr_schedule` anneals the step size to zero across the run, and 'cosine'
    is the answer to a MEASURED failure rather than a tidiness preference. Four
    runs of one identical single-image config (segformer, csf, tau 0.25, lr 0.2,
    400 steps, image 420) returned remote drops of 48.19, 44.02, 41.65 and
    38.60 — mean 43.1, sd 4.1, range 9.6, with no code change between them.
    Adam's update is on the order of `lr` per coordinate whatever the gradient
    magnitude, so a FLAT lr is still taking full-size steps at step 400: the run
    never settles, it wanders, and the reported number is wherever the walk
    happened to be when the step counter ran out. The spread was not smooth
    jitter either — any_flip_rate came out bimodal at 97-98% and 68-69% with
    nothing in between, i.e. two basins, and float-level nondeterminism decided
    which one each run found (the bilinear-upsample backward in upsample_to
    accumulates with atomics and has no deterministic CUDA kernel, so even a
    fixed seed does not repeat bitwise — two seed-42 reruns differed at step 1
    in the sixth decimal). Annealing does not remove the basins; it stops the
    tail of the run from hopping between them, so the endpoint is decided by the
    basin rather than by the step counter.

    The default is 'none' so that every number produced before this existed
    keeps its meaning. Callers opt in, and the choice is recorded in the
    returned record, so a run always states which regime it was.

    NOTE ON COMPARABILITY, inherited from overfit.py and still true: a
    single-image attack is a much EASIER problem than a universal patch trained
    across the dataset. Numbers from here belong beside per-image published
    numbers, never beside universal-patch ones.
    """
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
    # tsallis needs its q schedule, which the two-argument build() cannot
    # carry. Rebound ONLY on this branch and read off patch.cfg, so no other
    # loss_fn and no existing call site of attack_image() is touched.
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
    opt = torch.optim.Adam([patch.param], lr=lr, betas=(0.9, 0.999),
                           amsgrad=True)
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
        # `step` is 1-BASED here, so step-1 is the 0-based progress counter the
        # schedule is defined on (t = 0 at the first step, 1 at the last).
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
                _save_png(patch, out_dir / f"patch_step{step:04d}.png")

    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), hw)

        # Both class sets, via the shared helper, so a population run reports
        # exactly what train.py and evaluate.py report and a moved class set is
        # visible rather than buried in the mean.
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

    # The END-OF-RUN patch statistics, promoted into the record.
    # For csf that is realised visibility — tau is an INTENT, this is the
    # OUTCOME, and only the outcome is reportable. It previously lived solely
    # in `history`, which the population scripts strip before writing
    # summary.json, so a population run could not report the one number the
    # whole CSF family's claim rests on. For raw it carries frac_at_clip,
    # which is the early warning for saturation collapse across a population.
    final_stats = {f"final_{k}": v for k, v in patch.stats().items()}

    degraded = best_drop > (clean_rem - final_rem) + 1.0
    if degraded and verbose:
        log(f"\n  WARNING: best drop was {best_drop:+.2f} but the FINAL "
            f"patch only reaches {clean_rem-final_rem:+.2f}.\n"
            f"  The run degraded after its peak — check frac_at_clip in the "
            f"history for\n  sigmoid saturation, and use best.pt rather "
            f"than final.pt.")

    # WHERE the patch ended up is part of the result, not bookkeeping. Under
    # --placement gradcam it varies per image, so a population run cannot be
    # interpreted without it: a drop measured at the image centre and one
    # measured at a near-field hotspot are not the same measurement, and the
    # distance from centre is what separates them.
    H, W = hw
    p_side = int(H * patch.cfg.scale)
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
            # The 'union' figures, kept so pre-fix runs stay comparable. A gap
            # between drop_remote and drop_remote_union means a rare class moved
            # in or out of the denominator, which is a diagnostic in itself.
            "drop_all_union": cmp_all["drop_union"],
            "drop_remote_union": cmp_rem["drop_union"],
            "n_classes_gt": cmp_rem["n_classes_gt"],
            "n_classes_clean_union": cmp_rem["n_classes_clean_union"],
            "n_classes_adv_union": cmp_rem["n_classes_adv_union"],
            "class_set_moved": bool(cmp_rem["n_classes_clean_union"]
                                    != cmp_rem["n_classes_adv_union"]),
            "lr_schedule": lr_schedule,
            # tsallis ONLY, so the record schema for every other loss_fn is
            # byte-for-byte what it was and downstream parsers do not move.
            **({"tsallis_q": objective.q} if loss_fn == "tsallis" else {}),
            **final_stats, **rates, "history": history}


def _save_png(patch, path):
    from torchvision.utils import save_image
    save_image(patch.render().detach().cpu(), path)
