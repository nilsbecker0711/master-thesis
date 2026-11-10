#!/usr/bin/env python
"""IAP (Dutta & Zhang, ICCV 2025) against this repository's segmentors — the comparison arm for csf."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (add_model_args, add_patch_args, build_patch,
                     image_indices, make_dataset, setup_model)
from patchreach.data.cityscapes import norm_tensors, upsample_to
from patchreach.diagnostics import report
from patchreach.metrics import population as pop_mod
from patchreach.patch import iap as iap_mod
from patchreach.patch import segmentation_cam
from patchreach.utils import (get_device, increment_path, seed_everything,
                              tee_output)


def build_parser():
    p = add_patch_args(add_model_args(argparse.ArgumentParser(
        description="IAP (ICCV 2025) with this repository's segmentation "
                    "objectives, one patch per image",
        formatter_class=argparse.RawDescriptionHelpFormatter)))

    p.set_defaults(img_h=1024, img_w=2048, patch_size=256, patch_scale=0.25,
                   patch_mode="raw", pixel_param="direct",
                   raw_init="grey",
                   cam_objective="ce", cam_target="pred")

    g = p.add_argument_group("IAP")
    g.add_argument("--iap_lrs", default=",".join(str(x) for x in
                                                 iap_mod.IAP_LRS),
                   help="the restart ladder, tried in order and stopped at the "
                        "first rung that meets --iap_stop. The released "
                        "setting is 0.05,0.09,0.2 ('the step size is "
                        "reinitialized up to three times').")
    g.add_argument("--iap_steps", type=int, default=iap_mod.IAP_STEPS,
                   help="steps per rung. 1000 is the released setting.")
    g.add_argument("--iap_w1", type=float, default=iap_mod.IAP_W1,
                   help="weight on the TARGETED term. 1.5 released.")
    g.add_argument("--iap_w2", type=float, default=iap_mod.IAP_W2,
                   help="weight on the term away from the ground truth. 1.0 "
                        "released (an implicit minus on a CE).")
    g.add_argument("--iap_w3", type=float, default=iap_mod.IAP_W3,
                   help="weight on the perceptibility regulariser. 3 "
                        "released, AT 84x84 — see --iap_reg_reduce and D3. "
                        "Sweep it; do not inherit it.")
    g.add_argument("--iap_tar_loss", default="ce_target",
                   choices=list(iap_mod.TAR_LOSSES),
                   help="the targeted term. 'ce_target' (DEFAULT) is IAP's "
                        "OWN objective — a plain per-pixel CE toward "
                        "--target_class, no weighting, which is what "
                        "nn.CrossEntropyLoss() does in IAP.py. "
                        "'ipatch_cospgd' is targeted CosPGD (Agnihotri et "
                        "al., Eq. 8), this repository's own targeted "
                        "objective: use it for the arm that has to be "
                        "comparable to every other targeted row here, and "
                        "report the pair — the difference IS what the cosine "
                        "weight costs. 'none' drops the term and makes the "
                        "run untargeted, which IAP never is.")
    g.add_argument("--iap_untar_loss", default="ce",
                   choices=list(iap_mod.UNTAR_LOSSES),
                   help="the term away from the truth. 'ce' is IAP's literal "
                        "-CE(pred, truth) and needs no sign flip, because "
                        "ce_loss already returns -CE.")
    g.add_argument("--iap_reg_reduce", default="sum",
                   choices=list(iap_mod.REG_REDUCTIONS),
                   help="'sum' is the released arithmetic and its scale grows "
                        "with p**2; 'sum84' rescales by (84/p)**2 so w3 keeps "
                        "its published meaning at 256px; 'mean' is scale-free "
                        "and is not IAP.")
    g.add_argument("--iap_reg_grad", default="live",
                   choices=list(iap_mod.REG_GRADS),
                   help="D9, and the port's one deliberate preference for the "
                        "PAPER over the released code. 'live' (default): the "
                        "regulariser's gradient is w3*sens*sign(delta-base). "
                        "'released': the released code builds its reference as "
                        "temp = delta.clone(), which is differentiable, so the "
                        "term's two gradient paths cancel EXACTLY and it "
                        "optimises nothing while still being reported. Under "
                        "'released' w3 cannot affect the result at all — that "
                        "is the measurement, not a bug in this script.")
    g.add_argument("--iap_locate", default="cam",
                   choices=list(iap_mod.LOCALISE_SCORES),
                   help="window score. 'cam' is the RELEASED code (Grad-CAM "
                        "alone); 'cam_over_sens' is the paper's text (J/Sens) "
                        "— they disagree, see D1. 'center' is the ablation "
                        "that isolates localisation: same loss, same update, "
                        "the position every other arm of this thesis uses.")
    g.add_argument("--iap_stop", default="target",
                   choices=list(iap_mod.STOP_CRITERIA),
                   help="early stop, the analogue of 'target confidence >= "
                        "0.9'. 'target'/'flip' are fractions of remote pixels; "
                        "'drop' is remote mIoU points; 'none' always runs every "
                        "rung to --iap_steps.")
    g.add_argument("--iap_stop_thresh", type=float, default=0.9,
                   help="in the unit of --iap_stop. 0.9 mirrors the paper's "
                        "confidence bar and is a far harder bar over a "
                        "megapixel frame than over one label (D5).")
    g.add_argument("--iap_no_clamp", dest="iap_clamp", action="store_false",
                   default=True,
                   help="do NOT project the patch back into [0,1] after each "
                        "step. The released code never clips (the paper says "
                        "it does), so this reproduces the code exactly and "
                        "can leave the patch outside the pixel range — see D6.")
    g.add_argument("--margin_kappa", type=float, default=5.0,
                   help="--iap_untar_loss margin/cos_margin only.")

    g = p.add_argument_group("population")
    g.add_argument("--target_class", type=int, default=8,
                   help="the targeted class. 8 = vegetation, this "
                        "repository's default; 16 = train is the usual "
                        "targeted arm.")
    g.add_argument("--images", default="fixed10",
                   help="'fixed10' (the ten val images every thesis number is "
                        "reported on) | 'random' | 'all' | an explicit list "
                        "like '2 5 45'")
    g.add_argument("--n_images", type=int, default=0,
                   help="truncate the list, or sample this many under "
                        "--images random. 0 = all of it.")
    g.add_argument("--sample_seed", type=int, default=0,
                   help="seed for --images random. RECORD IT.")
    g.add_argument("--classes", default="gt", choices=["gt", "union"],
                   help="which classes enter the mIoU mean; 'gt' keeps clean "
                        "and attacked on one denominator.")
    g.add_argument("--select_key", default="drop_remote",
                   help="metric ranking images for --panel_select")
    g.add_argument("--n_panels", type=int, default=3,
                   help="images getting the full diagnostic suite and the "
                        "a_clean/c_patched pair scripts/lpips_eval.py reads. "
                        "0 disables, and then only the checkpoint source works.")
    g.add_argument("--panel_select", default="spread",
                   choices=list(pop_mod.SELECT_MODES),
                   help="'spread' = best + median + worst. 'best' is what a "
                        "reviewer will call cherry-picking.")
    g.add_argument("--exclude_footprint", action="store_true", default=True)
    g.add_argument("--log_every", type=int, default=100)
    g.add_argument("--quiet_images", dest="verbose_images",
                   action="store_false", default=True,
                   help="suppress per-step logging. Leave it on for a handful "
                        "of images; turn it off past ~50.")
    g.add_argument("--out_root", default="results/iap")
    g.add_argument("--tag", default="")
    return p


def resolve_images(a, n_val: int):
    """The image list, recorded in config.json either way."""
    if a.images == "random":
        g = torch.Generator().manual_seed(a.sample_seed)
        perm = torch.randperm(n_val, generator=g).tolist()
        return sorted(perm[:min(a.n_images or n_val, n_val)])
    return image_indices(a.images, n_val)[:a.n_images or None]


def check_args(a, p):
    """Refusals, each of which would otherwise produce a plausible number for a procedure that did not run."""
    want = iap_mod.REF_PATCH_SIDE
    side = int(a.img_h * a.patch_scale)
    if side != a.patch_size:
        p.error(
            f"--patch_size {a.patch_size} != int(--img_h {a.img_h} * "
            f"--patch_scale {a.patch_scale:g}) = {side}. IAP's sensitivity "
            f"map, regulariser and update rule are all defined on the PASTED "
            f"pixels, so any resampling between parameter and paste makes "
            f"three of its four parts describe a different signal. Set "
            f"--patch_size {side}, or --patch_scale "
            f"{a.patch_size / a.img_h:g}.")
    if a.cam_objective == "attack" and a.iap_locate != "center":
        if a.iap_untar_loss in ("ce", "cospgd"):
            a.cam_objective = a.iap_untar_loss
        else:
            p.error(
                f"--cam_objective attack means 'reuse the attack loss', but "
                f"IAP has two and --iap_untar_loss is {a.iap_untar_loss!r}, "
                f"which SegmentationCAM cannot build (it scores against the "
                f"clean prediction, which is what the CAM is computing). Pass "
                f"--cam_objective ce or cospgd explicitly.")
    if a.placement != "center":
        p.error(f"--placement {a.placement!r} has no effect under IAP; "
                f"localisation is --iap_locate (and --iap_locate center is "
                f"the centred control).")
    if a.shape != "square":
        p.error(f"IAP is a square patch; --shape {a.shape!r} is not part of "
                f"the procedure and its silhouette would leave part of the "
                f"footprint unperturbed while the regulariser still scored it.")
    if a.iap_stop == "target" and a.iap_tar_loss == "none":
        p.error("--iap_stop target needs a targeted run; pass "
                "--iap_stop flip (or drop) with --iap_tar_loss none.")
    if a.iap_reg_grad == "released" and a.iap_w3 != iap_mod.IAP_W3:
        p.error(
            f"--iap_reg_grad released with --iap_w3 {a.iap_w3:g}: under the "
            f"released code the regulariser's gradient cancels exactly (D9), "
            f"so w3 cannot change the optimisation and a swept value would "
            f"produce identical runs under different directory names. Leave "
            f"w3 at {iap_mod.IAP_W3:g}, or use --iap_reg_grad live to sweep "
            f"it.")
    if (a.iap_reg_reduce == "sum" and a.patch_size != want
            and a.iap_reg_grad == "live"):
        print(f"\n  NOTE (D3): --iap_reg_reduce sum at {a.patch_size}px sums "
              f"{(a.patch_size / want) ** 2:.1f}x more elements than the "
              f"{want}px patch\n        the published w3="
              f"{iap_mod.IAP_W3:g} was tuned at, against adversarial terms "
              f"that are per-pixel\n        means. Sweep --iap_w3, or use "
              f"--iap_reg_reduce sum84 to rescale it.\n")
    return a


def main():
    p = build_parser()
    a = check_args(p.parse_args(), p)

    arm = "_".join(x for x in
                   ["" if a.iap_locate == "cam" else a.iap_locate,
                    "" if a.iap_reg_reduce == "sum" else a.iap_reg_reduce,
                    "" if a.iap_reg_grad == "live" else "reggrad-released",
                    {"ce_target": "", "ipatch_cospgd": "tarcospgd",
                     "none": "untargeted"}[a.iap_tar_loss],
                    "" if a.iap_clamp else "noclamp"] if x)
    tag = "_".join(x for x in [a.arch, "iap", a.iap_untar_loss,
                               f"w3{a.iap_w3:g}", arm, a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(out_dir / "run.log"):
        _run(a, out_dir)
    print("\n  -> " + str(out_dir) + "/")


def _run(a, out_dir: Path):
    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)

    from patchreach.models.wrapper import SlidingWindowSegModel
    cam_model, cam_untiled = model, False
    if isinstance(model, SlidingWindowSegModel) and a.iap_locate != "center":
        inner = getattr(model, "model", None)
        if inner is None or isinstance(inner, SlidingWindowSegModel):
            raise SystemExit(
                "--iap_locate needs a Grad-CAM and the model is tiled, but "
                "the un-tiled model could not be recovered from it. Use "
                "--iap_locate center, or --inference whole.")
        cam_model, cam_untiled = inner, True

    mean_t, std_t = norm_tensors(device)

    ds = make_dataset(a, "val")
    idxs = resolve_images(a, len(ds))
    if not idxs:
        raise SystemExit("--images resolved to an empty list")
    patch_dir = out_dir / "patches"

    lrs = [float(x) for x in str(a.iap_lrs).replace(",", " ").split()]
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(a), "images": idxs, "iap_lr_ladder": lrs,
                   "iap_cam_untiled": cam_untiled,
                   "bracket": spec.bracket, "backbone_channels": n_ch,
                   "backbone_active_channels": n_act}, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f" IAP — {len(idxs)} images, one patch per image, "
          f"{len(lrs)} rungs x {a.iap_steps} steps")
    print(f" {a.arch} ({spec.bracket} attention) at {a.img_h}x{a.img_w}, "
          f"{a.patch_size}px patch "
          f"({100 * a.patch_size ** 2 / (a.img_h * a.img_w):.2f}% of frame)")
    print(f" procedure: patchreach/patch/iap.py — read D1-D10 before quoting "
          f"these numbers")
    print(f"{'=' * 72}")

    cam = None
    if a.iap_locate == "center":
        print("[cam ] none       : --iap_locate center, the localisation "
              "ablation")
    else:
        cam = segmentation_cam.build(cam_model, a.cam_objective,
                                     a.target_class, a.cam_layer,
                                     a.cam_module, a.cam_target)
        print(f"[cam ] objective  : {a.cam_objective}  target: {a.cam_target}"
              f"  layer {a.cam_layer} of {a.cam_module}")
        if cam_untiled:
            print(f"[cam ] D10        : map from a WHOLE-FRAME pass on the "
                  f"un-tiled model, attack runs SLIDE.")
            print(f"                    The two see different model "
                  f"behaviour — quote the pair, not just the drop.")

    pop = pop_mod.Population(a.num_classes, device=device,
                             target_class=(a.target_class
                                           if a.iap_tar_loss != "none"
                                           else None))
    t0 = time.time()
    for n, i in enumerate(idxs, 1):
        img, label = ds[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        print(f"\n{'#' * 72}")
        print(f"#  [{n}/{len(idxs)}] val image {i} — classes present "
              f"{sorted(label.unique().tolist())}")
        print(f"{'#' * 72}")

        patch = build_patch(a, device, mean_t, std_t)
        t_img = time.time()
        rec = iap_mod.attack_image(
            model, img, label, patch, mean_t=mean_t, std_t=std_t, cam=cam,
            target_class=a.target_class,
            tar_loss=a.iap_tar_loss, untar_loss=a.iap_untar_loss,
            w1=a.iap_w1, w2=a.iap_w2, w3=a.iap_w3,
            reg_reduce=a.iap_reg_reduce, reg_grad=a.iap_reg_grad,
            locate=a.iap_locate, cam_untiled=cam_untiled,
            lrs=lrs, steps=a.iap_steps,
            stop=a.iap_stop, stop_thresh=a.iap_stop_thresh,
            margin_kappa=a.margin_kappa,
            placement_margin=a.placement_margin,
            num_classes=a.num_classes,
            exclude_footprint=a.exclude_footprint, classes=a.classes,
            clamp=a.iap_clamp, log_every=a.log_every,
            out_dir=patch_dir / f"img{i:04d}",
            verbose=a.verbose_images)
        rec["image"] = i
        rec["wall_clock_s"] = time.time() - t_img

        with torch.no_grad():
            patched, fp = patch.apply(img)
            adv_logits = upsample_to(model(patched), label.shape[-2:])
            clean_logits = upsample_to(model(img), label.shape[-2:])
        pop.update(clean_logits, adv_logits, label, fp, rec)

        with open(patch_dir / f"img{i:04d}" / "results.json", "w") as f:
            json.dump({"config": vars(a), "image": i, **rec}, f, indent=2)
        with open(out_dir / "records.jsonl", "a") as fh:
            fh.write(json.dumps({k: v for k, v in rec.items()
                                 if k not in ("history", "iap_rungs")}) + "\n")

        el = time.time() - t0
        print(f"  [{n:3d}/{len(idxs)}] image {i:4d}  "
              f"drop_remote {rec['drop_remote']:+7.2f}  "
              f"best {rec['best_drop_remote']:+7.2f}  "
              f"flip {rec['any_flip_rate']:5.1f}%  "
              f"ssim_patch {rec['ssim_patch']:.3f}  "
              f"lr {rec['iap_lr']:g}  "
              f"({el / n / 60:.1f}m/img, eta {el / n * (len(idxs) - n) / 60:.0f}m)")

    if cam is not None:
        if cam.n_degenerate:
            print(f"\n  NOTE: the Grad-CAM was degenerate (ReLU suppressed "
                  f"every channel) on {cam.n_degenerate} images;")
            print(f"        those fell back to CENTRE placement and are not "
                  f"IAP-localised. They are still in the population.")
        cam.close()

    summary = pop.summarise(key=a.select_key)
    _report_invisibility(pop.records)

    panels = pop_mod.select(pop.records, a.n_panels, a.panel_select,
                            a.select_key)
    pop_mod.plot_distribution(pop.records, out_dir / "distribution.png",
                              key=a.select_key,
                              title=f"{a.arch} — IAP / {a.iap_untar_loss}, "
                                    f"w3={a.iap_w3:g}",
                              highlight=[r["image"] for r in panels])

    target = a.target_class if a.iap_tar_loss != "none" else None
    for rank, r in enumerate(panels):
        i = r["image"]
        img, label = ds[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        from patchreach.patch.spec import Patch
        patch = Patch.load(patch_dir / f"img{i:04d}" / "best.pt", device,
                           mean_t, std_t)
        d = out_dir / "diagnostics" / f"img{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            clean_logits = upsample_to(model(img), label.shape[-2:])
            patched, fp = patch.apply(img)
            adv_logits = upsample_to(model(patched), label.shape[-2:])
        report.per_class_iou_figure(clean_logits, adv_logits, label,
                                    a.num_classes, d / "per_class_iou.png",
                                    target, title=f"val image {i}")
        report.run(model, img, label, patch, d, a.iap_untar_loss,
                   a.num_classes, target, mean_t, std_t,
                   skip_geometric=(rank > 0), log=lambda *_, **__: None)
        print(f"    image {i}: drop_remote {r['drop_remote']:+.2f} -> {d}/")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"config": {**vars(a), "images": idxs,
                              "iap_lr_ladder": lrs},
                   "summary": summary,
                   "invisibility": _invisibility_stats(pop.records),
                   "panels": [r["image"] for r in panels],
                   "records": [{k: v for k, v in r.items()
                                if k not in ("history", "iap_rungs")}
                               for r in pop.records]}, f, indent=2)


INVIS_KEYS = ("ssim_patch", "ssim_image", "psnr_patch", "psnr_image",
              "iap_drift_mean_abs", "iap_drift_absmax", "iap_reg_final")


def _invisibility_stats(records) -> dict:
    """mean / sd / min / max per invisibility key, over the population."""
    import statistics
    out = {}
    for k in INVIS_KEYS:
        v = [r[k] for r in records if r.get(k) is not None]
        if not v:
            continue
        out[k] = {"mean": statistics.fmean(v),
                  "sd": statistics.stdev(v) if len(v) > 1 else None,
                  "min": min(v), "max": max(v), "n": len(v)}
    return out


def _report_invisibility(records):
    """The stealthiness half of the table, printed beside the strength half."""
    st = _invisibility_stats(records)
    if not st:
        return
    print(f"\n{'=' * 72}")
    print(" INVISIBILITY — footprint and whole frame, mean +/- sd [min, max]")
    print(f"{'-' * 72}")
    for k, s in st.items():
        sd = "    n/a" if s["sd"] is None else f"{s['sd']:7.4f}"
        print(f"  {k:22s} {s['mean']:10.4f} +/- {sd}   "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"{'=' * 72}")
    print("  SSIM/PSNR only. LPIPS, CLIPScore, UIQ and SRE — the rest of the")
    print("  paper's suite — come from scripts/lpips_eval.py in the "
          "lpips_eval env;")
    print("  it reads this run directly.")


if __name__ == "__main__":
    main()
