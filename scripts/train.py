#!/usr/bin/env python
r"""
Full-dataset adversarial patch training.

    python scripts/train.py --config configs/experiments/B_architecture.yaml \
                            --arch internimage --loss_fn cospgd

Every run writes a FLAT directory under results/runs/<run_id>/ containing
config.json + results.json. analysis/build_index.py walks those into one CSV.
Nested result paths were tried and broke: each new axis (arch, reach, lap
alpha, shape, placement) needs another directory level, and untargeted runs
ended up filed under cls-1. A flat store plus a generated index scales.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Model and patch arguments come from _common so they exist in exactly ONE
# place. train.py previously redeclared all of them, and three separate bugs
# came from updating one copy and not the other.
from _common import (add_model_args, add_patch_args, setup_model,
                     build_patch, tsallis_kwargs, tsallis_tag)
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import report
from patchreach.losses import adversarial, reach as reach_mod
from patchreach.metrics.miou import SegMetric, compare
from patchreach.patch.lap import magnitude_report, rationality_report
from patchreach.utils import (get_device, seed_everything, increment_path,
                              channel_probe)


def build_parser():
    """
    Model and patch arguments live in _common.add_model_args / add_patch_args.
    Only the ones specific to full-dataset TRAINING are declared here.
    """
    p = argparse.ArgumentParser(description="Train an adversarial patch")
    p.add_argument("--config", type=str, default=None,
                   help="YAML supplying defaults; CLI flags override it.")

    add_model_args(p)          # arch, cfg_path, weights, cityscapes_root,
                               # img_h, img_w, num_classes, seed
    add_patch_args(p)          # patch_mode, size, scale, logit_clip, shape*,
                               # placement*, reference*, lap_*, init_from

    p.add_argument("--loss_fn",
                   choices=["ce", "cospgd", "ipatch_cospgd", "tsallis"],
                   default="cospgd")
    p.add_argument("--target_class", type=int, default=8)

    # reach-restricted optimisation (training only — evaluation never uses it)
    p.add_argument("--reach_mode", choices=["off", "radial", "empirical"],
                   default="off")
    p.add_argument("--reach_radius", type=float, default=None)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr_schedule", default="plateau",
                   choices=["plateau", "cosine", "none"],
                   help="DEFAULT 'plateau' reproduces every run recorded so "
                        "far, byte for byte. 'cosine' anneals lr to zero over "
                        "the whole run, the schedule PR #15 gave overfit.py "
                        "and overfit_population.py for the reason measured "
                        "there: Adam's step is ~lr per coordinate whatever the "
                        "gradient, so a non-annealed run is still taking "
                        "full-size steps at the end and the reported number is "
                        "wherever the walk happened to stop (a 9.6-point swing "
                        "across four identical single-image invocations). "
                        "train.py did NOT get the flag in that PR; it does "
                        "now. Use cosine for universal_csf.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers. DEFAULT 4 is what every run so "
                        "far used, so it stays. Cityscapes decodes a 2048x1024 "
                        "PNG per sample and then resizes it, which at 4 "
                        "workers can cost more per batch than the segmentor "
                        "step it feeds — on a node with --cpus-per-task=16 "
                        "raise this to 8-16 and the epoch time may halve. "
                        "Check before assuming the GPU is the bottleneck.")
    p.add_argument("--exclude_footprint", action="store_true", default=True)
    p.add_argument("--val_images", type=int, default=20)
    p.add_argument("--val_every", type=int, default=5)

    p.add_argument("--out_root", type=str, default="results/runs")
    p.add_argument("--tag", type=str, default="",
                   help="Optional slug appended to the run id.")
    p.add_argument("--diag_image", type=int, default=2,
                   help="val image used for the post-training figure suite.")
    p.add_argument("--no_diagnostics", action="store_true",
                   help="skip the post-training figure suite.")
    p.add_argument("--panel_images", type=str, default="0 2 5",
                   help="Val image indices to render labelled panels for once "
                        "training finishes. '' disables. A finished run should "
                        "leave figures behind, not only JSON.")
    return p


def load_config(args):
    """YAML supplies defaults; explicit CLI flags win."""
    if not args.config:
        return args
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    supplied = {a.lstrip("-").split("=")[0].replace("-", "_")
                for a in sys.argv[1:] if a.startswith("--")}
    for k, v in cfg.items():
        if k not in supplied and hasattr(args, k):
            setattr(args, k, v)
    return args


def run_id(a) -> str:
    """Readable, sortable, unique. Encodes the axes that vary across blocks."""
    bits = [a.arch, a.patch_mode, a.loss_fn, f"{a.img_h}x{a.img_w}"]
    if a.loss_fn == "ipatch_cospgd":
        bits.append(f"cls{a.target_class}")
    if tsallis_tag(a):
        bits.append(tsallis_tag(a))
    if a.patch_mode == "lap":
        bits.append(f"a{a.lap_alpha:g}")
    if a.patch_mode in ("csf", "universal_csf"):
        bits.append(f"tau{a.csf_threshold:g}")
        if getattr(a, "csf_lref", 0.0) > 0:
            bits.append(f"lref{a.csf_lref:g}")
        if getattr(a, "csf_composite", "clip") != "clip":
            bits.append(a.csf_composite)
    if getattr(a, "lr_schedule", "plateau") != "plateau":
        bits.append(f"lr-{a.lr_schedule}")
    if a.shape != "square":
        bits.append(f"sh-{a.shape}")
    if a.placement != "center":
        bits.append(f"pl-{a.placement}{a.placement_class}")
    if a.reach_mode != "off":
        bits.append(f"reach-{a.reach_mode}")
    if a.tag:
        bits.append(a.tag)
    bits.append(f"s{a.seed}")
    return "_".join(bits)


@torch.no_grad()
def evaluate(model, loader, patch, device, K, target_class=None):
    """Clean vs patched mIoU over the fixed val subset, all + remote."""
    ms = {k: SegMetric(K, device=device)
          for k in ("clean_all", "clean_rem", "adv_all", "adv_rem")}
    flips, hits, n_flip, n_hit = 0, 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        hw = labels.shape[-2:]
        lc = upsample_to(model(imgs), hw)
        patched, fp = patch.apply(imgs)
        la = upsample_to(model(patched), hw)
        pc, pa = lc.argmax(1), la.argmax(1)

        ms["clean_all"].update(pc, labels)
        ms["clean_rem"].update(pc, labels, exclude=fp)
        ms["adv_all"].update(pa, labels)
        ms["adv_rem"].update(pa, labels, exclude=fp)

        remote = (labels != 255) & (~fp)
        n_flip += int(remote.sum())
        flips += int((remote & (pc != pa)).sum())
        if target_class is not None:
            nt = remote & (pc != target_class)
            n_hit += int(nt.sum())
            hits += int((nt & (pa == target_class)).sum())

    # See metrics/miou.py: 'gt' fixes the counted class set from the labels so
    # clean and adv share a denominator. 'union' is kept for continuity with
    # runs recorded before the distinction existed.
    out = {}
    for scope, ck, ak, tag in (("all", "clean_all", "adv_all", "all"),
                               ("remote", "clean_rem", "adv_rem", "rem")):
        d = compare(ms[ck], ms[ak])
        out[f"clean_{tag}"] = d["clean"]
        out[f"adv_{tag}"] = d["adv"]
        out[f"clean_{tag}_union"] = d["clean_union"]
        out[f"adv_{tag}_union"] = d["adv_union"]
        out[f"drop_{scope}"] = d["drop"]
        out[f"drop_{scope}_union"] = d["drop_union"]
        out[f"n_classes_{scope}"] = d["n_classes_gt"]
        out[f"n_classes_{scope}_union_clean"] = d["n_classes_clean_union"]
        out[f"n_classes_{scope}_union_adv"] = d["n_classes_adv_union"]
    out["any_flip_rate"] = 100.0 * flips / max(n_flip, 1)
    if target_class is not None:
        out["target_hit_rate"] = 100.0 * hits / max(n_hit, 1)
    return out


def main():
    args = load_config(build_parser().parse_args())
    seed_everything(args.seed)
    device = get_device()

    out_dir = increment_path(Path(args.out_root) / run_id(args))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patches").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n### {out_dir} ###")

    # setup_model resolves the registry entry, loads the segmentor (registering
    # custom backbones from what the CONFIG declares) and probes the head width.
    model, n_ch, n_active, spec = setup_model(args)
    if n_ch != args.num_classes:
        print(f"[head] NOTE: head emits {n_ch} but num_classes="
              f"{args.num_classes}; trainIds 0..18 are used, the rest are inert")

    mean_t, std_t = norm_tensors(device)
    train_ds = CityscapesSeg(args.cityscapes_root, "train", args.img_h, args.img_w)
    val_full = CityscapesSeg(args.cityscapes_root, "val", args.img_h, args.img_w)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        Subset(val_full, list(range(min(args.val_images, len(val_full))))),
        batch_size=1, shuffle=False, num_workers=2)
    print(f"[data] train {len(train_ds)} imgs / {len(train_loader)} batches, "
          f"val {args.val_images} imgs @ {args.img_h}x{args.img_w}")

    # BigGAN only when the mode needs it
    G = None
    if args.patch_mode in ("gan", "raw_ganinit"):
        from GANLatentDiscovery.loading import load_from_dir
        from GANLatentDiscovery.utils import is_conditional
        _, G, _ = load_from_dir(
            "./GANLatentDiscovery/models/pretrained/deformators/BigGAN/",
            G_weights="./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth")
        if is_conditional(G):
            G.set_classes(259)
        G.eval().to(device)
        for p in G.parameters():
            p.requires_grad_(False)

    if args.patch_mode == "universal_csf":
        p_px = int(args.img_h * args.patch_scale)
        if args.patch_size != p_px:
            raise SystemExit(
                f"universal_csf needs --patch_size to equal the pasted "
                f"footprint int(img_h*patch_scale) = {p_px}, got "
                f"{args.patch_size}. Resampling a residual resamples its "
                f"SPECTRUM, so the budget its bins were projected onto would "
                f"no longer describe the pasted signal and tau would be a "
                f"number about a different tensor. "
                f"fix: --patch_size {p_px}   (or --patch_scale "
                f"{args.patch_size/args.img_h:g})")
        if args.placement == "gradcam":
            raise SystemExit(
                "universal_csf does not support --placement gradcam: the CAM "
                "is per image while the patch is one shared tensor, and "
                "Patch.apply() composites a single (top,left) for the whole "
                "batch. Use center or fixed.")

    patch = build_patch(args, device, mean_t, std_t, generator=G)

    # ORDERING: clean forward -> resolve_placement -> first apply().
    # Semantic placement reads the CLEAN PREDICTION, so it cannot be resolved
    # before the model has seen an image. Getting this backwards silently
    # falls back to centre.
    probe_img = val_full[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        clean_pred = upsample_to(model(probe_img),
                                 (args.img_h, args.img_w)).argmax(1)[0]
    patch.resolve_placement(args.img_h, args.img_w, clean_pred)
    patch.describe(args.img_h, args.img_w)

    support = None
    if args.reach_mode != "off":
        _, fp = patch.apply(probe_img)
        if args.reach_mode == "radial":
            support = reach_mod.radial_mask(fp, args.reach_radius
                                            or 0.62 * args.img_h)
            n_r, n_t = int(support.sum()), int((~fp[0]).sum())
            print(f"[reach] radial r={args.reach_radius or 0.62*args.img_h:.0f}px"
                  f" — {n_r:,}/{n_t:,} remote px ({100*n_r/max(n_t,1):.1f}%)")
        else:
            _, support = reach_mod.empirical_mask(
                model, train_loader, patch, device)
        torch.save(support.cpu(), out_dir / "reach_mask.pt")

    if patch.reference is not None:
        save_image(patch.reference.cpu(), out_dir / "reference.png")
    if patch.shape_mask is not None:
        save_image(patch.shape_mask.float().unsqueeze(0).cpu(),
                   out_dir / "shape_mask.png")

    adv_loss = adversarial.build(
        args.loss_fn,
        args.target_class if args.loss_fn == "ipatch_cospgd" else 8)
    tgt = args.target_class if args.loss_fn == "ipatch_cospgd" else None

    opt = torch.optim.Adam([patch.param], lr=args.lr, betas=(0.5, 0.999),
                           amsgrad=True)
    # T_max is TOTAL OPTIMISER STEPS, not epochs. Annealing per epoch would
    # decay ~len(loader) times too slowly and leave the tail exactly as hot as
    # the flat schedule this exists to replace.
    total_steps = args.epochs * len(train_loader)
    if args.lr_schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=30, factor=0.5)
    elif args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps)
        print(f"[sched] cosine, lr {args.lr:g} -> 0 over {total_steps:,} steps "
              f"({args.epochs} epochs x {len(train_loader):,} batches)")
    else:
        sched = None
        print("[sched] none — flat lr. The run will not settle; see "
              "--lr_schedule.")

    # tsallis carries a q schedule that build()'s two-argument form cannot
    # express. Rebound HERE rather than above because the schedule's horizon is
    # total_steps, which does not exist until the loader is sized — and the
    # horizon is the WHOLE RUN, epochs x batches, exactly like the cosine
    # T_max above. Only this branch is touched; every other loss_fn keeps the
    # object built earlier.
    if args.loss_fn == "tsallis":
        adv_loss = adversarial.build(args.loss_fn, 8,
                                     **tsallis_kwargs(args, total_steps))
        print(f"[loss ] {adv_loss!r}")

    history, best, checked, t0 = [], -1e9, False, time.time()
    gstep = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            # GLOBAL step, deliberately not per-epoch: a universal patch is one
            # tensor optimised across the whole run, so its q schedule spans the
            # run. Resetting per epoch would sawtooth q and make every epoch a
            # different attack.
            if hasattr(adv_loss, "on_step_begin"):
                adv_loss.on_step_begin(gstep, total_steps)
            gstep += 1
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()

            patched, fp = patch.apply(imgs)
            logits = upsample_to(model(patched), labels.shape[-2:])

            sup = None if support is None else support.unsqueeze(0)
            if args.exclude_footprint:
                sup = (~fp) if sup is None else (sup & (~fp))
            la = adv_loss(logits, labels, fp, sup)

            # Fail loudly rather than training 100 epochs on NaN.
            if not torch.isfinite(la):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch}; valid remote px this "
                    f"batch = {int(((labels != 255) & (~fp)).sum())}")

            extra = patch.regularisers()
            (la + extra["total"]).backward()

            if not checked:
                g = patch.param.grad
                print(f"\n[grad] {'None — GRAPH BROKEN' if g is None else f'abs mean {g.abs().mean():.3e}'}")
                if patch.cfg.mode == "lap":
                    # Calibrate on the SAME active set the loop optimises —
                    # q = M(p), not the full rectangle.
                    magnitude_report(la.item(), patch.render(),
                                     patch.reference, patch.active_mask(),
                                     patch.cfg)
                checked = True

            opt.step()
            patch.project()
            # Cosine steps PER BATCH (its T_max is in optimiser steps);
            # plateau steps per epoch, on the epoch mean, below.
            if args.lr_schedule == "cosine":
                sched.step()
            running += la.item()

        running /= len(train_loader)
        if args.lr_schedule == "plateau":
            sched.step(running)
        st = " ".join(f"{k}={v:.4f}" for k, v in patch.stats().items())
        print(f"[epoch {epoch:3d}] {args.loss_fn}={running:.5f}  {st}")
        if args.loss_fn == "tsallis":
            print(f"           q={adv_loss.q:+.4f}  "
                  f"(step {gstep:,}/{total_steps:,})")
        save_image(patch.render().cpu(),
                   out_dir / "patches" / f"epoch{epoch:04d}.png")

        if epoch % args.val_every == 0 or epoch == 1:
            ev = evaluate(model, val_loader, patch, device, args.num_classes, tgt)
            extra_s = (f"  hit={ev['target_hit_rate']:.1f}%" if tgt is not None
                       else f"  flip={ev['any_flip_rate']:.1f}%")
            print(f"           VAL remote {ev['adv_rem']:.2f} "
                  f"({ev['drop_remote']:+.2f}){extra_s}   <- headline")
            history.append({"epoch": epoch, "loss": running, **ev})
            if ev["drop_remote"] > best:
                best = ev["drop_remote"]
                patch.save(out_dir / "best.pt")
                save_image(patch.render().cpu(), out_dir / "best_patch.png")
                print(f"           * new best drop_remote {best:+.2f}")

    final = evaluate(model, val_loader, patch, device, args.num_classes, tgt)
    # ── calibration for stage 2 ──────────────────────────────────────────────
    # L_rat at step 1 is ~0 by construction: a stage-1 run starts AT the
    # reference, so ||q - c|| = 0 and any weight derived from it is a division
    # by zero. The meaningful scale is the DRIFT the attack produced — measured
    # here, at the end. Set alpha from this row, not from the step-1 one.
    if patch.cfg.mode == "lap" and patch.reference is not None:
        print("\n[lap] END-OF-RUN magnitudes — set stage-2 alpha from THIS "
              "rat row,\n      not from the step-1 report (where L_rat is 0 "
              "by construction):")
        magnitude_report(running, patch.render(), patch.reference,
                         patch.active_mask(), patch.cfg)

    patch.save(out_dir / "final.pt")
    save_image(patch.render().cpu(), out_dir / "final_patch.png")

    # ── universal_csf: the metrics this mode exists to produce ───────────────
    # Run on the BEST checkpoint, not the last one, so the reported visibility
    # describes the patch the headline number came from.
    universal_report = None
    if patch.cfg.mode == "universal_csf":
        from patchreach.diagnostics import universal as univ
        best_ck = out_dir / "best.pt"
        if best_ck.exists():
            patch.param.data = torch.load(
                best_ck, map_location=device)["param"].to(device)
        universal_report = univ.realised_tau(patch, val_loader, device,
                                             mean_t, std_t)
        alloc = univ.spectral_allocation(patch)
        universal_report["spectral_allocation"] = alloc
        univ.log_report(universal_report, alloc)
        univ.plot(universal_report, alloc, out_dir / "universal_csf.png",
                  title=out_dir.name)

    results = {"run_id": out_dir.name, "config": vars(args),
               "patch_config": asdict(patch.cfg),
               "backbone_channels": n_ch, "backbone_active_channels": n_active,
               "final": final, "best_drop_remote": best,
               "wall_clock_s": time.time() - t0, "history": history}
    if universal_report is not None:
        results["universal_csf"] = universal_report
    # tsallis ONLY, so the results schema for every other loss_fn is unchanged
    # and analysis/build_index.py keeps producing the same columns.
    if args.loss_fn == "tsallis":
        results["tsallis_q"] = adv_loss.q
    if patch.cfg.mode == "lap":
        results["rationality"] = rationality_report(patch.render(),
                                                    patch.reference)
    if patch.shape_mask is not None:
        results["silhouette_frac"] = patch.shape_mask.float().mean().item()
    if patch.placement is not None:
        results["placement_px"] = list(patch.placement)

    # ── labelled figures ────────────────────────────────────────────────────
    # Panels: clean / clean-prediction / patched / adv-prediction / change map,
    # plus a per-class IoU chart. Panel (b) is the CLEAN PREDICTION rather than
    # ground truth — the attack's effect is (clean pred -> adv pred), and
    # scoring against GT would fold the model's own errors into what reads as
    # attack damage.
    if args.panel_images.strip():
        idxs = [int(x) for x in args.panel_images.replace(",", " ").split()]
        idxs = [i for i in idxs if i < len(val_full)]
        print(f"\n[panels] rendering {len(idxs)} labelled image(s)")
        results["per_class_iou"] = {
            str(k): v for k, v in report.panels_for_images(
                model, val_full, idxs, patch, out_dir / "panels", mean_t,
                std_t, args.num_classes, tgt, args.img_h, args.img_w).items()}

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── post-training figures ────────────────────────────────────────────────
    # Numbers alone do not show WHERE the attack acted or WHAT it converted
    # pixels into. The panels and the confusion/flow figures do, and the flip
    # map is what makes perspective effects legible (a flood that looks
    # scene-wide is often entirely within a few hundred px of the patch).
    if not args.no_diagnostics:
        print("\n" + "=" * 66)
        print(f" FIGURES on val image #{args.diag_image}")
        print("=" * 66)
        # Full suite (ERF probe, confusion, margins, curves) on ONE image —
        # the probe costs n_probes extra forward passes.
        d_img, d_lbl = val_full[args.diag_image]
        d_img = d_img.unsqueeze(0).to(device)
        d_lbl = d_lbl.unsqueeze(0).to(device)
        with torch.no_grad():
            d_clean = upsample_to(model(d_img), d_lbl.shape[-2:])
        patch.resolve_placement(args.img_h, args.img_w, d_clean.argmax(1)[0])
        report.run(model, d_img, d_lbl, patch, out_dir / "diagnostics",
                   args.loss_fn, args.num_classes, tgt, mean_t, std_t)

    print("\n" + "=" * 66)
    print(f" drop_remote {final['drop_remote']:+.2f}   "
          f"any_flip {final['any_flip_rate']:.1f}%")
    print(f" -> {out_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()