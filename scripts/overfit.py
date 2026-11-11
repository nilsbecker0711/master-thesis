#!/usr/bin/env python
"""Single-image overfit — the sanity check before any multi-hour run."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from _common import (add_model_args, add_patch_args, setup_model,
                     build_patch, tsallis_kwargs, tsallis_tag)
from _common import make_dataset
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import report
from patchreach.patch import optimise, segmentation_cam
from patchreach.patch.lap import magnitude_report
from patchreach.utils import (get_device, seed_everything, increment_path,
                             tee_output)


AGGREGATE_KEYS = ("drop_remote", "drop_all", "best_drop_remote",
                  "final_remote", "final_all",
                  "any_flip_rate", "target_hit_rate",
                  "final_visibility", "final_visibility_local",
                  "final_resid_rms", "final_frac_at_clip",
                  "final_frac_at_bound", "final_spend_mean",
                  "wall_clock_s")


def run_one(a, seed: int, model, img, label, device, mean_t, std_t, G,
            out_dir: Path, run_diagnostics: bool) -> dict:
    """ONE complete attack, from a fresh patch, into its own directory."""
    seed_everything(seed)
    t0 = time.time()
    patch = build_patch(a, device, mean_t, std_t, generator=G)

    cam = None
    if a.placement == "gradcam":
        cam = segmentation_cam.build(
            model, a.loss_fn if a.cam_objective == "attack" else a.cam_objective,
            a.target_class, a.cam_layer, a.cam_module, a.cam_target,
            tsallis=tsallis_kwargs(a, a.steps))
    clean_logits = optimise.prepare(model, img, patch, a.img_h, a.img_w,
                                    a.from_image, mean_t, std_t,
                                    cam=cam, label=label, log=print)
    if cam is not None:
        cam.close()
    patch.describe(a.img_h, a.img_w)

    res = optimise.attack_image(
        model, img, label, patch,
        loss_fn=a.loss_fn, target_class=a.target_class, steps=a.steps,
        lr=a.lr, optimiser=a.optimiser, num_classes=a.num_classes,
        exclude_footprint=a.exclude_footprint, log_every=a.log_every,
        lr_schedule=a.lr_schedule, margin_kappa=a.margin_kappa,
        clean_logits=clean_logits, out_dir=out_dir,
        save_step_images=True, save_best=True, verbose=True)

    if patch.cfg.mode == "lap" and patch.reference is not None:
        print("\n[lap] END-OF-RUN magnitudes — set stage-2 alpha from THIS "
              "rat row,\n      not from the step-1 report (where L_rat is 0 "
              "by construction):")
        magnitude_report(res["last_loss"], patch.render(), patch.reference,
                         patch.active_mask(), patch.cfg)

    patch.save(out_dir / "final.pt")
    save_image(patch.render().detach().cpu(), out_dir / "final_patch.png")

    print(f"\n{'='*66}")
    print(f"  clean  all/remote : {res['clean_all']:.2f} / "
          f"{res['clean_remote']:.2f}")
    print(f"  final  all/remote : {res['final_all']:.2f} / "
          f"{res['final_remote']:.2f}")
    print(f"  drop   REMOTE     : {res['drop_remote']:+.2f}   <- the result")
    for k in ("any_flip_rate", "target_hit_rate"):
        if k in res:
            print(f"  {k:18s}: {res[k]:.1f}%")
    print(f"{'='*66}")

    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), label.shape[-2:])

    res["wall_clock_s"] = time.time() - t0
    out = {"config": vars(a), "seed": seed, **res}
    if patch.cfg.mode == "lap":
        from patchreach.patch.lap import rationality_report
        out["rationality"] = rationality_report(patch.render(),
                                                patch.reference)
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    best_ck = out_dir / "best.pt"
    if best_ck.exists():
        with torch.no_grad():
            patch.param.copy_(torch.load(best_ck, map_location="cpu")["param"]
                              .to(patch.param.device))
            patched, fp = patch.apply(img)
            adv_logits = upsample_to(model(patched), label.shape[-2:])
    else:
        print("  [diagnostics] no best.pt; figures use the final patch")

    report.per_class_iou_figure(
        clean_logits, adv_logits, label, a.num_classes,
        out_dir / "per_class_iou.png",
        a.target_class if a.loss_fn == "ipatch_cospgd" else None,
        title=f"val image {a.image}")

    if run_diagnostics and not a.no_diagnostics:
        report.run(model, img, label, patch, out_dir / "diagnostics",
                   a.loss_fn, a.num_classes,
                   a.target_class if a.loss_fn == "ipatch_cospgd" else None,
                   mean_t, std_t)
    return res


def summarise(rows, out_dir: Path) -> dict:
    """Mean and SAMPLE standard deviation across repeats, plus the raw rows."""
    stats = {}
    for k in AGGREGATE_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        stats[k] = {
            "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "min": min(vals), "max": max(vals), "n": len(vals),
        }
    summary = {"n_seeds": len(rows), "per_seed": rows, "stats": stats}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  {len(rows)} REPEATS — mean +/- sd (sample), [min, max]")
    print(f"{'-'*66}")
    for k, s in stats.items():
        sd = "    n/a" if s["sd"] is None else f"{s['sd']:7.2f}"
        print(f"  {k:26s} {s['mean']:8.2f} +/- {sd}   "
              f"[{s['min']:.2f}, {s['max']:.2f}]")
    print(f"{'='*66}")
    print("  Quote the mean and the sd. One repeat is a single sample from\n"
          "  this spread, and the best of N is not a measurement of anything.")
    return summary


def main():
    p = add_patch_args(add_model_args(argparse.ArgumentParser()))
    p.add_argument("--loss_fn", default="cospgd",
                   choices=["ce", "cospgd", "ce_target",
                            "ipatch_cospgd", "tsallis",
                            "margin", "cos_margin"])
    p.add_argument("--margin_kappa", type=float, default=5.0,
                   help="--loss_fn margin/cos_margin only: required logit "
                        "margin by "
                        "which a pixel must leave its CLEAN prediction "
                        "before it stops drawing gradient.")
    p.add_argument("--from_image", action="store_true",
                    help="Initialise the patch with the image region it will replace.")
    p.add_argument("--target_class", type=int, default=8)
    p.add_argument("--image", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimiser", default="adam",
                   choices=["adam", "sign"],
                   help="step rule. 'adam' is every run to date. "
                        "'sign' is PGD's update, p -= lr*sign(grad), "
                        "matched to adam at EQUAL lr because both "
                        "displace a coordinate by ~lr per step — see "
                        "patchreach/optim.py.")
    p.add_argument("--lr_schedule", default="cosine",
                   choices=["none", "cosine"],
                   help="Anneal lr to zero over the run. 'none' restores the "
                        "flat-lr regime, which was measured swinging 9.6 mIoU "
                        "points across four identical runs.")
    p.add_argument("--seeds", type=int, default=1,
                   help="Repeat the WHOLE attack N times, from seed, seed+1, "
                        "..., and report mean +/- sd. N > 1 writes one "
                        "subdirectory per repeat plus summary.json. This "
                        "measures the spread; it does not remove it.")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--exclude_footprint", action="store_true", default=True)
    p.add_argument("--no_diagnostics", action="store_true")
    p.add_argument("--out_root", default="results/overfit")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    if a.seeds < 1:
        p.error("--seeds must be at least 1")

    arm = "_".join(x for x in
                   ["" if a.optimiser == "adam" else a.optimiser,
                    "" if a.pixel_param == "sigmoid" else a.pixel_param]
                   if x)
    tag = "_".join(x for x in [a.arch, a.patch_mode, a.loss_fn,
                               f"img{a.image}", arm, tsallis_tag(a),
                               a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tee_output(out_dir / "run.log"):
        _run(a, out_dir)
    print("\n  -> " + str(out_dir) + "/")


def _run(a, out_dir: Path):
    """The invocation proper, with stdout and stderr already mirrored into out_dir/run.log."""
    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    ds = make_dataset(a, "val")
    img, label = ds[a.image]
    img, label = img.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
    print(f"[data] image {a.image}, classes present "
          f"{sorted(label.unique().tolist())}")

    G = None
    if a.patch_mode in ("gan", "raw_ganinit"):
        from GANLatentDiscovery.loading import load_from_dir
        from GANLatentDiscovery.utils import is_conditional
        _, G, _ = load_from_dir(
            "./GANLatentDiscovery/models/pretrained/deformators/BigGAN/",
            G_weights="./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth")
        if is_conditional(G):
            G.set_classes(259)
        G.eval().to(device)
        for q in G.parameters():
            q.requires_grad_(False)

    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(a), "bracket": spec.bracket,
                   "backbone_channels": n_ch,
                   "backbone_active_channels": n_act}, f, indent=2)

    rows = []
    for i in range(a.seeds):
        seed = a.seed + i
        run_dir = out_dir if a.seeds == 1 else out_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if a.seeds > 1:
            print(f"\n{'#'*66}")
            print(f"#  repeat {i+1}/{a.seeds}   seed {seed}")
            print(f"{'#'*66}")
        res = run_one(a, seed, model, img, label, device, mean_t, std_t, G,
                      run_dir, run_diagnostics=(i == 0))
        rows.append({"seed": seed,
                     **{k: res[k] for k in AGGREGATE_KEYS if k in res}})

    if a.seeds > 1:
        summarise(rows, out_dir)


if __name__ == "__main__":
    main()
