"""IAP — Invisible Adversarial Patch (Dutta & Zhang, ICCV 2025), ported to semantic segmentation."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..data.cityscapes import upsample_to
from ..losses import adversarial
from ..metrics.miou import single_image_miou, attack_rates
from .placement import find_max_response_placement

IAP_LRS: Tuple[float, ...] = (0.05, 0.09, 0.2)
IAP_STEPS = 1000
IAP_W1, IAP_W2, IAP_W3 = 1.5, 1.0, 3.0
SENS_EPS = 1e-3
REF_PATCH_SIDE = 84

REG_REDUCTIONS = ("sum", "sum84", "mean")
LOCALISE_SCORES = ("cam", "cam_over_sens", "center")
STOP_CRITERIA = ("none", "flip", "target", "drop")
REG_GRADS = ("live", "released")
TAR_LOSSES = ("ce_target", "ipatch_cospgd", "none")
UNTAR_LOSSES = ("ce", "cospgd", "margin", "cos_margin", "none")


def window_std(x: torch.Tensor, dh: int, dw: int) -> torch.Tensor:
    """[B,h,w] of channel-averaged standard deviations over a dh x dw window, h = H-dh+1, w = W-dw+1."""
    if dh < 1 or dw < 1:
        raise ValueError(f"window must be at least 1x1, got {dh}x{dw}")
    H, W = x.shape[-2:]
    h, w = H - dh + 1, W - dw + 1
    if h < 1 or w < 1:
        raise ValueError(
            f"a {dh}x{dw} window does not fit a {H}x{W} image")
    sl = torch.stack([x[:, :, i:i + h, j:j + w]
                      for i in range(dh) for j in range(dw)], dim=-1)
    if sl.shape[-1] < 2:
        raise ValueError("window must contain at least two pixels")
    return sl.std(dim=-1, unbiased=True).mean(dim=1)


def _cuda_peak(device) -> Optional[int]:
    """Peak CUDA allocation since the counter was last reset, or None on CPU."""
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:                                      # pragma: no cover
        return None


def _anchor(std: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Place a [B,h,w] window statistic at the window's TOP-LEFT pixel and leave the remainder zero — utils.py's `stdten[i][j] = std` with i, j the loop variables."""
    out = std.new_zeros(std.shape[0], H, W)
    out[:, :std.shape[-2], :std.shape[-1]] = std
    return out


def sensitivity(x01: torch.Tensor, eps: float = SENS_EPS) -> torch.Tensor:
    """[B,H,W] sensitivity, HIGH where a perturbation would be visible."""
    if x01.dim() != 4:
        raise ValueError(f"expected [B,3,H,W], got {tuple(x01.shape)}")
    H, W = x01.shape[-2:]
    x = x01.float()
    sx = _anchor(window_std(x, 1, 3), H, W)
    sy = _anchor(window_std(x, 3, 1), H, W)
    return 1.0 / (torch.minimum(sx, sy).sqrt() + eps)


@torch.no_grad()
def localise(cam: torch.Tensor, sens: torch.Tensor, p: int,
             score: str = "cam", margin: int = 0,
             eps: float = SENS_EPS) -> Tuple[int, int]:
    """Top-left of the p x p window IAP attacks."""
    if score not in LOCALISE_SCORES:
        raise ValueError(f"score must be one of {LOCALISE_SCORES}, "
                         f"got {score!r}")
    if score == "center":
        H, W = sens.shape[-2:]
        return (H - p) // 2, (W - p) // 2
    if cam is None or cam.dim() != 2:
        raise ValueError(
            f"score={score!r} needs the Grad-CAM as [H,W], got "
            f"{None if cam is None else tuple(cam.shape)}")
    if score == "cam_over_sens":
        if sens is None:
            raise ValueError("score='cam_over_sens' needs the sensitivity map")
        field = cam / (sens + eps)
    else:
        field = cam
    return find_max_response_placement(field, p, centre_if_constant=True,
                                       margin=margin)


def regulariser(delta: torch.Tensor, base: torch.Tensor,
                sens_patch: torch.Tensor, reduce: str = "sum",
                side_ref: int = REF_PATCH_SIDE) -> torch.Tensor:
    """R = REDUCE( sens * |delta - base| )"""
    if reduce not in REG_REDUCTIONS:
        raise ValueError(f"reduce must be one of {REG_REDUCTIONS}, "
                         f"got {reduce!r}")
    w = sens_patch.detach() * (delta - base).abs()
    if reduce == "mean":
        return w.mean()
    total = w.sum()
    if reduce == "sum84":
        p = delta.shape[-1]
        total = total * (float(side_ref) / float(p)) ** 2
    return total


@torch.no_grad()
def colour_constant_step_(param: torch.Tensor, sens_patch: torch.Tensor,
                          lr: float, eps: float = SENS_EPS) -> dict:
    """IAP's update, in place on `param`."""
    g = param.grad
    if g is None:
        raise RuntimeError(
            "IAP step with param.grad = None — the graph never reached the "
            "patch. Check that the composite was built from the parameter "
            "(Patch.apply) and that backward() ran.")
    if not torch.isfinite(g).all():
        raise RuntimeError("non-finite gradient at the IAP step")

    step = lr * g.mean(dim=-3, keepdim=True) * param.data * (1.0 / sens_patch)
    param.data.sub_(step)
    return {"step_rms": float(step.pow(2).mean().sqrt()),
            "step_absmax": float(step.abs().max()),
            "grad_absmean": float(g.abs().mean())}


class Objective:
    """IAP's three-term loss with this repository's segmentation losses in the first two slots."""

    def __init__(self, tar: str = "ce_target", untar: str = "ce",
                 target_class: int = 8,
                 w1: float = IAP_W1, w2: float = IAP_W2, w3: float = IAP_W3,
                 reg_reduce: str = "sum", reg_grad: str = "live",
                 margin_ref: Optional[torch.Tensor] = None,
                 margin_kappa: float = 5.0):
        if reg_grad not in REG_GRADS:
            raise ValueError(f"reg_grad must be one of {REG_GRADS}, "
                             f"got {reg_grad!r}")
        if tar not in TAR_LOSSES:
            raise ValueError(f"iap_tar_loss must be one of {TAR_LOSSES}, "
                             f"got {tar!r}")
        if untar not in UNTAR_LOSSES:
            raise ValueError(f"iap_untar_loss {untar!r} is not a loss in "
                             f"losses/adversarial.py")
        if tar == "none" and untar == "none":
            raise ValueError("both adversarial terms are 'none' — that run "
                             "optimises the regulariser alone and converges "
                             "back to the clean image region")
        if untar in ("margin", "cos_margin") and margin_ref is None:
            raise ValueError(f"iap_untar_loss {untar!r} scores against the "
                             f"CLEAN PREDICTION; pass margin_ref")
        self.tar_name, self.untar_name = tar, untar
        self.w1, self.w2, self.w3 = float(w1), float(w2), float(w3)
        self.reg_reduce = reg_reduce
        self.reg_grad = reg_grad
        self.margin_ref = margin_ref
        self._tar = (None if tar == "none"
                     else adversarial.build(tar, target_class))
        if untar == "none":
            self._untar = None
        elif untar in ("margin", "cos_margin"):
            self._untar = adversarial.build(untar, margin_ref=margin_ref,
                                            margin_kappa=margin_kappa)
        else:
            self._untar = adversarial.build(untar, target_class)

    def __repr__(self):
        g = "" if self.reg_grad == "live" else ",NO GRADIENT"
        return (f"IAP({self.w1:g}*{self.tar_name} + {self.w2:g}*"
                f"{self.untar_name} + {self.w3:g}*reg[{self.reg_reduce}{g}])")

    def on_step_begin(self, step: int, total: int):
        """Forward the progress counter to whichever term is stateful."""
        for fn in (self._tar, self._untar):
            if hasattr(fn, "on_step_begin"):
                fn.on_step_begin(step, total)

    def __call__(self, logits, label, footprint, support,
                 delta, base, sens_patch) -> dict:
        """Every term separately, plus 'total'."""
        out = {}
        total = logits.new_zeros(())
        if self._untar is not None:
            lu = self._untar(logits, label, footprint, support)
            out["untar"] = float(lu)
            total = total + self.w2 * lu
        if self._tar is not None:
            lt = self._tar(logits, label, footprint, support)
            out["tar"] = float(lt)
            total = total + self.w1 * lt
        r = regulariser(delta, base, sens_patch, self.reg_reduce)
        out["reg"] = float(r)
        total = total + self.w3 * (r.detach() if self.reg_grad == "released"
                                   else r)
        out["total"] = total
        return out


@torch.no_grad()
def progress_rate(clean_logits, adv_logits, label, footprint,
                  criterion: str = "target",
                  target_class: Optional[int] = None,
                  num_classes: int = 19,
                  clean_remote: Optional[float] = None) -> float:
    """One number in the unit the stop threshold is quoted in."""
    if criterion not in STOP_CRITERIA:
        raise ValueError(f"criterion must be one of {STOP_CRITERIA}, "
                         f"got {criterion!r}")
    if criterion == "none":
        return 0.0
    if criterion == "drop":
        if clean_remote is None:
            raise ValueError("criterion='drop' needs the clean remote mIoU")
        return clean_remote - single_image_miou(
            adv_logits, label, num_classes, exclude=footprint)
    if criterion == "target" and target_class is None:
        raise ValueError("criterion='target' needs a target_class")
    r = attack_rates(clean_logits, adv_logits, label, footprint,
                     target_class if criterion == "target" else None)
    key = "any_flip_rate" if criterion == "flip" else "target_hit_rate"
    return r[key] / 100.0


def ssim_psnr(a: torch.Tensor, b: torch.Tensor,
              data_range: float = 1.0) -> Tuple[float, Optional[float]]:
    """(SSIM, PSNR) for one [1,3,h,w] pair in [0,1], via torchmetrics with data_range=1.0 — the reference's `SSIM(data_range=1.0)` and `PSNR(data_range=1.0)`."""
    try:
        from torchmetrics.image import (PeakSignalNoiseRatio,
                                        StructuralSimilarityIndexMeasure)
    except ImportError as e:                               # pragma: no cover
        raise ImportError(
            "IAP's invisibility metrics need torchmetrics (it is pinned in "
            "requirements.txt and in envs/iap.yml). "
            "pip install torchmetrics==0.11.4") from e
    dev = a.device
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(dev)
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(dev)
    s, p = float(ssim(a, b)), float(psnr(a, b))

    return s, (None if p == float("inf") else p)


def invisibility(clean01: torch.Tensor, adv01: torch.Tensor,
                 box: Tuple[int, int, int]) -> dict:
    """SSIM and PSNR on the FOOTPRINT and on the WHOLE FRAME, which is the local/global split the paper reports every invisibility metric under."""
    top, left, p = box
    cp = clean01[:, :, top:top + p, left:left + p]
    ap = adv01[:, :, top:top + p, left:left + p]
    s_loc, p_loc = ssim_psnr(cp, ap)
    s_glb, p_glb = ssim_psnr(clean01, adv01)
    return {"ssim_patch": s_loc, "psnr_patch": p_loc,
            "ssim_image": s_glb, "psnr_image": p_glb}


def prepare(model, img, patch, *, mean_t, std_t, label, cam,
            score: str = "cam", margin: int = 0, cam_untiled: bool = False,
            log=print) -> dict:
    """Everything IAP does BEFORE its first optimisation step, in the one order that works"""
    H, W = img.shape[-2:]
    p = patch.side(H, W)
    if p != patch.cfg.size:
        raise ValueError(
            f"IAP needs the patch pasted at its parameter resolution: "
            f"side(H,W) = {p} but --patch_size = {patch.cfg.size}. Any "
            f"resampling between the two interpolates the perturbation, so "
            f"the sensitivity map, the regulariser and the update rule would "
            f"all be computed on a different signal from the one pasted. At "
            f"{H}x{W} with --patch_scale {patch.cfg.scale:g} that means "
            f"--patch_size {p}.")
    if patch.cfg.mode != "raw" or patch.cfg.pixel_param != "direct":
        raise ValueError(
            f"IAP is --patch_mode raw --pixel_param direct (the parameter IS "
            f"the pixel), got mode={patch.cfg.mode!r} "
            f"pixel_param={patch.cfg.pixel_param!r}. Under a sigmoid "
            f"parameterisation the update's `* delta` factor multiplies by a "
            f"LOGIT, which is negative below 0.5 and flips the sign of the "
            f"step for half the patch.")

    x01 = (img * std_t + mean_t).clamp(0.0, 1.0)
    sens_full = sensitivity(x01)

    if score == "center":
        with torch.no_grad():
            clean_logits = upsample_to(model(img), label.shape[-2:])
        cam_map = None
    else:
        if cam is None:
            raise ValueError(
                f"score={score!r} needs a SegmentationCAM. Build one with "
                f"segmentation_cam.build(model, ...), or pass score='center' "
                f"for the localisation ablation. Centring silently here would "
                f"report an IAP-localised run that was never localised.")
        if cam_untiled:
            if img.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(img.device)
            try:
                cam_map, _ = cam(img, label)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                raise RuntimeError(
                    "the whole-frame Grad-CAM pass ran out of memory (D10). "
                    "That pass is one un-tiled forward AND backward at the "
                    "full input size, which the 2026-09-21 log entry flags as "
                    "memory-bound at native resolution — it is the attack "
                    "loop that slide + --slide_checkpoint makes cheap, not "
                    "this. Either drop to --iap_locate center, which builds "
                    "no CAM at all and keeps the slide attack, or run the "
                    "whole arm at a resolution where the frame fits "
                    "(for SegFormer/Swin, slide and whole are the same "
                    "computation at 1024x1024 and below anyway)."
                ) from e
            peak = _cuda_peak(img.device)
            with torch.no_grad():
                clean_logits = upsample_to(model(img), label.shape[-2:])
            if peak is not None:
                log(f"[iap  ] cam pass  : whole-frame on the un-tiled model, "
                    f"peak {peak / 2 ** 30:.2f} GiB (D10)")
        else:
            cam_map, clean_logits = cam(img, label)
            clean_logits = upsample_to(clean_logits, label.shape[-2:])
    top, left = localise(None if cam_map is None else cam_map[0, 0],
                         sens_full[0], p, score=score, margin=margin)

    patch.cfg.placement = "fixed"
    patch.cfg.placement_xy = ((top + p / 2) / H, (left + p / 2) / W)
    patch.placement = (int(top), int(left))

    base = x01[0, :, top:top + p, left:left + p].clone()
    sens_patch = sens_full[0, top:top + p, left:left + p].clone()
    with torch.no_grad():
        patch.param.data.copy_(base)

    ctop, cleft = (H - p) // 2, (W - p) // 2
    log(f"[iap  ] localise  : score={score} -> (top,left)=({top},{left}), "
        f"{((top - ctop) ** 2 + (left - cleft) ** 2) ** 0.5:.0f}px from centre")
    log(f"[iap  ] sens patch: min {float(sens_patch.min()):.1f}  "
        f"median {float(sens_patch.median()):.1f}  "
        f"max {float(sens_patch.max()):.1f}   (ceiling 1/eps = "
        f"{1.0 / SENS_EPS:.0f}, a perfectly flat neighbourhood)")
    log(f"[iap  ] base      : the image region itself, {p}x{p} at "
        f"({top},{left})")
    return {"clean_logits": clean_logits, "x01": x01,
            "sens": sens_full, "sens_patch": sens_patch, "base": base,
            "box": (int(top), int(left), int(p))}


def _rung(model, img, label, patch, prepared, objective, lr, steps, *,
          stop, stop_thresh, target_class, num_classes, exclude_footprint,
          classes, clamp, clean_remote, log_every, verbose, log) -> dict:
    """ONE learning rate, from a fresh delta — the reference's inner `for t in range(iterations)` loop."""
    hw = label.shape[-2:]
    base, sens_patch = prepared["base"], prepared["sens_patch"]
    clean_logits = prepared["clean_logits"]

    with torch.no_grad():
        patch.param.data.copy_(base)
    best_drop, best_param, history = -1e9, patch.param.detach().clone(), []
    stopped_at, last = None, {}

    for step in range(1, steps + 1):
        objective.on_step_begin(step - 1, steps)
        will_log = (step % log_every == 0 or step == 1)
        if patch.param.grad is not None:
            patch.param.grad = None
        patched, fp = patch.apply(img)
        logits = upsample_to(model(patched), hw)
        sup = (~fp) if exclude_footprint else None

        parts = objective(logits, label, fp, sup,
                          patch.render(), base, sens_patch)
        total = parts["total"]
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite IAP loss at step {step}: {parts}")

        rate = progress_rate(clean_logits, logits.detach(), label, fp,
                             criterion=stop, target_class=target_class,
                             num_classes=num_classes,
                             clean_remote=clean_remote)
        last = {k: float(v) for k, v in parts.items() if k != "total"}
        last["total"] = float(total)
        last["rate"] = rate
        if stop != "none" and rate >= stop_thresh:
            stopped_at = step
            best_param = patch.param.detach().clone()
            best_drop = clean_remote - single_image_miou(
                logits.detach(), label, num_classes, exclude=fp,
                classes=classes)
            if verbose:
                log(f"  step {step:4d}  STOP: {stop} = {rate:.4f} "
                    f">= {stop_thresh:g}   (this patch is the rung's result, "
                    f"drop {best_drop:+.2f})")
            break

        cand = patch.param.detach().clone() if will_log else None

        total.backward()
        st = colour_constant_step_(patch.param, sens_patch, lr)
        if clamp:
            patch.project()

        if will_log:
            with torch.no_grad():
                cur = single_image_miou(logits.detach(), label, num_classes,
                                        exclude=fp, classes=classes)
                drift = float((cand - base).abs().mean())
            row = {"step": step, "miou_remote": cur, "drift_mean_abs": drift,
                   **last, **st}
            history.append(row)
            if verbose:
                log(f"  step {step:4d}  remote={cur:6.2f} "
                    f"({clean_remote - cur:+6.2f})  "
                    + " ".join(f"{k}={row[k]:.4g}" for k in
                               ("tar", "untar", "reg") if k in row)
                    + f"  |d|={drift:.4f}  {stop}={row['rate']:.3f}")
            if clean_remote - cur > best_drop:
                best_drop = clean_remote - cur
                best_param = cand

    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), hw)
        final_rem = single_image_miou(adv_logits, label, num_classes,
                                      exclude=fp, classes=classes)
    final_param = patch.param.detach().clone()
    if clean_remote - final_rem > best_drop:
        best_drop, best_param = clean_remote - final_rem, final_param.clone()

    return {"lr": float(lr), "steps_run": stopped_at or steps,
            "stopped_at": stopped_at,
            "final_remote": final_rem,
            "drop_remote": clean_remote - final_rem,
            "best_drop_remote": best_drop,
            "last": last, "history": history,
            "best_param": best_param, "final_param": final_param}


def attack_image(model, img, label, patch, *, mean_t, std_t, cam,
                 target_class: int = 8,
                 tar_loss: str = "ce_target",
                 untar_loss: str = "ce",
                 w1: float = IAP_W1, w2: float = IAP_W2, w3: float = IAP_W3,
                 reg_reduce: str = "sum", reg_grad: str = "live",
                 locate: str = "cam", cam_untiled: bool = False,
                 lrs: Sequence[float] = IAP_LRS,
                 steps: int = IAP_STEPS,
                 stop: str = "target", stop_thresh: float = 0.9,
                 margin_kappa: float = 5.0,
                 placement_margin: int = 0,
                 num_classes: int = 19,
                 exclude_footprint: bool = True,
                 classes: str = "gt",
                 clamp: bool = True,
                 log_every: int = 50,
                 out_dir=None,
                 verbose: bool = True,
                 log=print) -> dict:
    """The whole IAP procedure against ONE image."""
    if stop not in STOP_CRITERIA:
        raise ValueError(f"stop must be one of {STOP_CRITERIA}, got {stop!r}")
    if not len(lrs):
        raise ValueError("the lr ladder is empty")
    hw = label.shape[-2:]

    prepared = prepare(model, img, patch, mean_t=mean_t, std_t=std_t,
                       label=label, cam=cam, score=locate,
                       margin=placement_margin, cam_untiled=cam_untiled,
                       log=log)
    clean_logits = prepared["clean_logits"]
    _, fp0 = patch.apply(img)
    clean_all = single_image_miou(clean_logits, label, num_classes,
                                  classes=classes)
    clean_rem = single_image_miou(clean_logits, label, num_classes,
                                  exclude=fp0, classes=classes)

    margin_ref = None
    if untar_loss in ("margin", "cos_margin"):
        margin_ref = clean_logits.argmax(1)
        margin_ref[label == 255] = 255

    objective = Objective(tar=tar_loss, untar=untar_loss,
                          target_class=target_class, w1=w1, w2=w2, w3=w3,
                          reg_reduce=reg_reduce, reg_grad=reg_grad,
                          margin_ref=margin_ref, margin_kappa=margin_kappa)
    if verbose:
        log(f"[clean] all {clean_all:.2f} | remote {clean_rem:.2f}  "
            f"<- remote is the baseline that matters")
        log(f"[iap  ] loss      : {objective!r}")
        if reg_grad == "released":
            log(f"[iap  ] reg grad  : RELEASED — the regulariser is reported "
                f"and optimises NOTHING (D9).\n"
                f"                   w3 cannot affect this run; that is the "
                f"point of the arm.")
        log(f"[iap  ] ladder    : lr {', '.join(f'{x:g}' for x in lrs)}  "
            f"x {steps} steps, stop on {stop} >= {stop_thresh:g}")
        if reg_reduce == "sum":
            p = prepared["box"][2]
            log(f"[iap  ] reg scale : sum over {p}x{p}x3 elements, "
                f"{(p / REF_PATCH_SIDE) ** 2:.3g}x the element count the "
                f"published w3={IAP_W3:g} was tuned at (D3). --iap_reg_reduce "
                f"sum84 rescales it.")

    rungs = []
    for i, lr in enumerate(lrs):
        if verbose:
            log(f"\n{'-' * 66}\n  rung {i + 1}/{len(lrs)}   lr {lr:g}"
                f"\n{'-' * 66}")
        r = _rung(model, img, label, patch, prepared, objective, lr, steps,
                  stop=stop, stop_thresh=stop_thresh,
                  target_class=target_class, num_classes=num_classes,
                  exclude_footprint=exclude_footprint, classes=classes,
                  clamp=clamp, clean_remote=clean_rem, log_every=log_every,
                  verbose=verbose, log=log)
        rungs.append(r)
        if r["stopped_at"] is not None:
            break

    i_win = next((i for i, r in enumerate(rungs)
                  if r["stopped_at"] is not None), None)
    if i_win is None:
        i_win = max(range(len(rungs)),
                    key=lambda i: rungs[i]["best_drop_remote"])
    win = rungs[i_win]
    with torch.no_grad():
        patch.param.data.copy_(win["best_param"])

    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), hw)
        final_all = single_image_miou(adv_logits, label, num_classes,
                                      classes=classes)
        final_rem = single_image_miou(adv_logits, label, num_classes,
                                      exclude=fp, classes=classes)
        rates = attack_rates(clean_logits, adv_logits, label, fp,
                             target_class if tar_loss != "none" else None)
        adv01 = (patched * std_t + mean_t).clamp(0.0, 1.0)
        vis = invisibility(prepared["x01"], adv01, prepared["box"])
        reg_final = float(regulariser(patch.render(), prepared["base"],
                                      prepared["sens_patch"], reg_reduce))

    if out_dir is not None:
        from pathlib import Path
        from torchvision.utils import save_image
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        patch.save(out_dir / "best.pt")
        save_image(patch.render().detach().cpu(), out_dir / "best_patch.png")
        with torch.no_grad():
            keep = patch.param.detach().clone()
            patch.param.data.copy_(win["final_param"])
            patch.save(out_dir / "final.pt")
            save_image(patch.render().detach().cpu(),
                       out_dir / "final_patch.png")
            patch.param.data.copy_(keep)
        s = prepared["sens_patch"]
        save_image((s / s.max()).detach().cpu().unsqueeze(0),
                   out_dir / "sensitivity_patch.png")

    top, left, p = prepared["box"]
    ctop, cleft = (hw[0] - p) // 2, (hw[1] - p) // 2
    H, W = hw
    record = {
        "clean_all": clean_all, "clean_remote": clean_rem,
        "final_all": final_all, "final_remote": final_rem,
        "drop_all": clean_all - final_all,
        "drop_remote": clean_rem - final_rem,
        "best_drop_remote": win["best_drop_remote"],
        "last_loss": win["last"].get("total"),
        "placement": [top, left],
        "placement_policy": "fixed",
        "placement_dist_from_centre": float(
            ((top - ctop) ** 2 + (left - cleft) ** 2) ** 0.5),
        "placement_on_border": bool(top in (0, H - p) or left in (0, W - p)),
        "classes": classes,
        "optimiser": "iap_colour_constant",
        "pixel_param": patch.cfg.pixel_param,
        "lr_schedule": "iap_ladder",
        **rates,
        "iap_locate": locate,
        "iap_cam_untiled": bool(cam_untiled),
        "iap_cam_inference": ("whole" if cam_untiled else "same_as_attack"),
        "iap_lr": win["lr"],
        "iap_lr_ladder": [float(x) for x in lrs],
        "iap_rung_index": i_win,
        "iap_stopped_at": win["stopped_at"],
        "iap_stop": stop, "iap_stop_thresh": stop_thresh,
        "iap_steps_run": sum(r["steps_run"] for r in rungs),
        "iap_w1": w1, "iap_w2": w2, "iap_w3": w3,
        "iap_reg_reduce": reg_reduce, "iap_reg_grad": reg_grad,
        "iap_tar_loss": tar_loss, "iap_untar_loss": untar_loss,
        "iap_clamp": bool(clamp),
        "iap_reg_final": reg_final,
        "iap_sens_patch_median": float(prepared["sens_patch"].median()),
        "iap_drift_mean_abs": float(
            (patch.render() - prepared["base"]).abs().mean()),
        "iap_drift_absmax": float(
            (patch.render() - prepared["base"]).abs().max()),
        **vis,
        "iap_rungs": [{k: v for k, v in r.items()
                       if k not in ("best_param", "final_param")}
                      for r in rungs],
        "history": win["history"],
    }
    if verbose:
        why = (f"stopped at step {win['stopped_at']}"
               if win["stopped_at"] is not None else "ran out, best by drop")
        log(f"\n{'=' * 66}")
        log(f"  winning rung      : lr {win['lr']:g} ({why})")
        log(f"  clean  all/remote : {clean_all:.2f} / {clean_rem:.2f}")
        log(f"  final  all/remote : {final_all:.2f} / {final_rem:.2f}")
        log(f"  drop   REMOTE     : {clean_rem - final_rem:+.2f}   <- "
            f"the result")
        for k in ("any_flip_rate", "target_hit_rate"):
            if k in rates:
                log(f"  {k:18s}: {rates[k]:.1f}%")
        log(f"  SSIM patch/image  : {vis['ssim_patch']:.4f} / "
            f"{vis['ssim_image']:.4f}")
        log(f"  PSNR patch/image  : {vis['psnr_patch']:.2f} / "
            f"{vis['psnr_image']:.2f} dB")
        log(f"{'=' * 66}")
    return record
