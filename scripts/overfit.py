#!/usr/bin/env python
r"""
Single-image overfit — the sanity check before any multi-hour run.

    python scripts/overfit.py --arch segformer_b0 --cityscapes_root $CS \
        --patch_mode raw --loss_fn cospgd --image 2 --steps 300

If mIoU on ONE image drops sharply, the pipeline is correct — scale up. If it
does not, there is a bug, and finding it here costs two minutes instead of five
hours. This isolates "is the attack possible at all?" from "does it generalise?".

ONE IMAGE IS AN ANECDOTE. For a result rather than a check, run the same attack
over a population and report the distribution:

    python scripts/overfit_population.py --images random --n_images 100 ...

Both scripts call patchreach.patch.optimise.attack_image(), so the procedure is
identical by construction and a change to one cannot silently fail to reach the
other.

ONE RUN IS ALSO AN ANECDOTE — see --seeds. At a flat lr the same invocation was
measured swinging 9.6 mIoU points across four runs with no code change between
them, so a number from a single run is one sample from a distribution rather
than a measurement.

NOTE ON COMPARABILITY: a single-image overfit is a much EASIER problem than a
universal patch trained across the dataset. Yuan et al.'s numbers come from
per-image patches at 400 iterations; ours are universal. Do not compare a
result from this script against a published universal-patch number.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torchvision.utils import save_image

from _common import (add_model_args, add_patch_args, setup_model,
                     build_patch, tsallis_kwargs, tsallis_tag)
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import report
from patchreach.patch import optimise, segmentation_cam
from patchreach.patch.lap import magnitude_report
from patchreach.utils import get_device, seed_everything, increment_path


# The keys a repeat run aggregates. Deliberately a FIXED list rather than
# "every numeric key in the record": averaging a placement coordinate or a
# class count produces a number that looks like a result and means nothing.
# Keys absent from a record — target_hit_rate outside a targeted run, the csf
# visibilities outside csf mode — are skipped rather than defaulted to zero.
AGGREGATE_KEYS = ("drop_remote", "drop_all", "best_drop_remote",
                  "final_remote", "final_all",
                  "any_flip_rate", "target_hit_rate",
                  "final_visibility", "final_visibility_local",
                  "final_resid_rms", "final_frac_at_clip")


def run_one(a, seed: int, model, img, label, device, mean_t, std_t, G,
            out_dir: Path, run_diagnostics: bool) -> dict:
    """
    ONE complete attack, from a fresh patch, into its own directory.

    Seeding happens HERE rather than once in main(), because the seed's whole
    job is to set the patch initialisation and a repeat run needs a different
    one per repeat.

    Note what the seed does NOT buy: reproducibility. Two seed-42 runs of one
    identical config were measured differing from step 1 onward, because the
    bilinear upsample in the loss path has a backward that accumulates with
    atomics and has no deterministic CUDA kernel. The seed controls the
    initialisation; the arithmetic varies on its own, and --seeds samples both
    sources of spread together rather than pretending to control either.
    """
    seed_everything(seed)
    patch = build_patch(a, device, mean_t, std_t, generator=G)

    # ORDERING: clean forward -> resolve_placement -> first apply().
    # Semantic placement reads the CLEAN PREDICTION, so it cannot be resolved
    # before the model has seen an image. Backwards, it silently uses centre.
    # --placement gradcam needs the sensitivity map. Built here rather than
    # inside prepare() so the hook is registered once and closed once; the CAM
    # holds a forward hook on the backbone and leaking one per repeat would
    # accumulate hooks across a --seeds run.
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
        lr=a.lr, num_classes=a.num_classes,
        exclude_footprint=a.exclude_footprint, log_every=a.log_every,
        lr_schedule=a.lr_schedule,
        clean_logits=clean_logits, out_dir=out_dir,
        save_step_images=True, save_best=True, verbose=True)

    # ── calibration for stage 2 ──────────────────────────────────────────────
    # L_rat at step 1 is ~0 by construction: a stage-1 run starts AT the
    # reference, so ||q - c|| = 0 and any weight derived from it is a division
    # by zero. The meaningful scale is the DRIFT the attack produced — measured
    # here, at the end. Set alpha from this row, not from the step-1 one.
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

    out = {"config": vars(a), "seed": seed, **res}
    if patch.cfg.mode == "lap":
        from patchreach.patch.lap import rationality_report
        out["rationality"] = rationality_report(patch.render(),
                                                patch.reference)
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    report.per_class_iou_figure(
        clean_logits, adv_logits, label, a.num_classes,
        out_dir / "per_class_iou.png",
        a.target_class if a.loss_fn == "ipatch_cospgd" else None,
        title=f"val image {a.image}")

    # The suite runs ONCE per invocation, not once per repeat. It is the
    # expensive part of the script and it characterises the attack
    # qualitatively; N copies of the same figure set is cost without
    # information.
    if run_diagnostics and not a.no_diagnostics:
        report.run(model, img, label, patch, out_dir / "diagnostics",
                   a.loss_fn, a.num_classes,
                   a.target_class if a.loss_fn == "ipatch_cospgd" else None,
                   mean_t, std_t)
    return res


def summarise(rows, out_dir: Path) -> dict:
    """
    Mean and SAMPLE standard deviation across repeats, plus the raw rows.

    Sample sd (n-1), not population sd: these repeats are a sample from the
    attack's outcome distribution, not the whole of it. With n = 1 the sd is
    undefined and is reported as null rather than as 0.0, because a zero there
    reads as "perfectly reproducible" — the exact opposite of what one run
    establishes.

    min and max are carried so the SHAPE stays visible. The spread that
    motivated this was bimodal — any_flip_rate at 97-98% and at 68-69% with
    nothing in between — and a mean with an sd describes that badly. If max
    minus min is several times the sd, look at per_seed before quoting either.
    """
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
                   choices=["ce", "cospgd", "ipatch_cospgd", "tsallis"])
    p.add_argument("--from_image", action="store_true",
                    help="Initialise the patch with the image region it will replace.")
    p.add_argument("--target_class", type=int, default=8)
    p.add_argument("--image", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
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

    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    ds = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
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

    tag = "_".join(x for x in [a.arch, a.patch_mode, a.loss_fn,
                               f"img{a.image}", tsallis_tag(a),
                               a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --seeds 1 keeps the ORIGINAL layout — results.json directly in out_dir,
    # no nesting, no summary — so everything that already reads these runs
    # (analysis/pick_lr.py globs for them) keeps working unchanged. Only a
    # repeat run nests, because only a repeat run has something to nest.
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
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()
