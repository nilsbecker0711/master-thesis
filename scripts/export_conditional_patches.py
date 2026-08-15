#!/usr/bin/env python
r"""
Test-time use of a trained generator: export p_i for arbitrary images, and
evaluate the attack. NO TRAINING, and NO GRADIENT-BASED OPTIMISATION OF p_i.

    python scripts/export_conditional_patches.py \
        --checkpoint results/conditional/<run_id>/checkpoints/best.pt \
        --arch segformer --cityscapes_root $CS --images fixed10

TEST-TIME BEHAVIOUR — THE DISTINCTION THAT DEFINES THIS THREAT MODEL
--------------------------------------------------------------------
    1. load frozen G_theta*            (eval(), no_grad, deterministic noise)
    2. take a previously unseen image  (--split val, or any index)
    3. compute its sensitivity map     M_i
    4. extract its centre crop         r_i
    5. generate its patch              p_i = G_theta*(x_i, r_i, M_i)
    6. place the patch                 argmax MeanPool(M_i) or centre
    7. evaluate segmentation

        x_test -> G_theta* -> p_test

There is no optimiser in this file. Contrast with scripts/overfit.py, which
RUNS gradient descent on a patch tensor for a single image — that is BASELINE
B, an approximate upper bound on direct optimisation, and it is a different
procedure with a different cost. Comparing the two is the point; conflating
them is not.

BASELINES
---------
  A  --method reference   p_i = r_i, the raw centre crop, no generator.
                          Identical placement and compositing, so the delta is
                          attributable to G_theta alone.
  B  scripts/overfit.py, e.g.

         python scripts/overfit.py --arch segformer --cityscapes_root $CS \
             --patch_mode raw --loss_fn cospgd --image 2 --steps 300 \
             --patch_scale 0.25 --patch_size 128

     Match --patch_scale/--patch_size to the generator run or the comparison
     is confounded by patch AREA, which Yuan et al. Table 6 shows dominates
     attack strength.
  C  --method generator   this file's default — the proposed attack.

Optional ablation: --gen_placement center isolates the contribution of
sensitivity-guided localisation from the contribution of the generator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_model_args, setup_model, image_indices
from patchreach.data.cityscapes import (CityscapesSeg, class_name,
                                        norm_tensors, upsample_to)
from patchreach.diagnostics import conditional as cviz
from patchreach.losses import adversarial
from patchreach.metrics.miou import SegMetric, single_image_miou, attack_rates
from patchreach.patch import conditional_generator as cg
from patchreach.patch import segmentation_cam
from patchreach.patch.lap import asi, agi, ade
from patchreach.utils import get_device, seed_everything, increment_path


def build_parser():
    p = add_model_args(argparse.ArgumentParser(
        description="Export / evaluate an image-conditioned patch generator"))
    p.add_argument("--checkpoint", default=None,
                   help="conditional-generator checkpoint. Not needed for "
                        "--method reference (baseline A uses no generator).")
    p.add_argument("--method", default="generator",
                   choices=["generator", "reference"],
                   help="generator = baseline C (proposed). "
                        "reference = baseline A (p_i = r_i).")
    p.add_argument("--split", default="val", choices=["val", "train", "test"])
    p.add_argument("--images", default="fixed10",
                   help="'fixed10' | 'all' | '2 5 45'")

    p.add_argument("--loss_fn", default=None,
                   help="defaults to the value stored in the checkpoint")
    p.add_argument("--target_class", type=int, default=None)

    # Geometry / placement / CAM default to the checkpoint's training values;
    # overriding one is how the placement ablation is run WITHOUT retraining.
    p.add_argument("--patch_scale", type=float, default=None)
    p.add_argument("--gen_placement", default=None,
                   choices=["center", "gradcam", "semantic", "fixed"],
                   help="override the trained placement policy. This is the "
                        "'generator without Grad-CAM placement' ablation and "
                        "it needs no retraining.")
    p.add_argument("--gen_placement_class", type=int, default=None)
    p.add_argument("--gen_placement_margin", type=int, default=None,
                   help="override the trained border margin for the gradcam "
                        "argmax; another no-retrain placement ablation.")
    p.add_argument("--gen_reference", default=None,
                   choices=["center", "window"],
                   help="override where r_i is sampled. Changing this at "
                        "evaluation time measures how the trained generator "
                        "behaves on a reference distribution it did not see.")
    p.add_argument("--cam_objective", default=None)
    p.add_argument("--cam_target", default=None, choices=["pred", "gt"])
    p.add_argument("--cam_layer", type=int, default=None)
    p.add_argument("--cam_module", default=None)

    p.add_argument("--save_patches", action="store_true", default=True,
                   help="write p_i, r_i and M_i per image")
    p.add_argument("--no_panels", action="store_true")
    p.add_argument("--lpips_net", default="alex",
                   choices=["alex", "vgg", "squeeze"])
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--out_root", default="results/conditional_eval")
    p.add_argument("--tag", default="")
    return p


def main():
    a = build_parser().parse_args()
    seed_everything(a.seed)
    device = get_device()

    # ── the trained generator + the configuration it was trained under ──────
    generator, ck, tcfg = None, {}, {}
    if a.method == "generator":
        if not a.checkpoint:
            raise SystemExit("--method generator needs --checkpoint")
        generator, ck = cg.load_checkpoint(a.checkpoint, device)
        tcfg = ck.get("train_config", {})
        geom = ck.get("patch_geometry", {})
        print(f"\n[gen ] loaded {a.checkpoint}")
        print(f"[gen ] epoch {ck.get('epoch')}  "
              f"{generator.n_parameters():,} shared parameters")
        generator.cfg.describe()
        for q in generator.parameters():
            q.requires_grad_(False)          # belt and braces: frozen at test
    else:
        geom = {}
        print("\n[base] BASELINE A — p_i = r_i, no generator")

    size = generator.cfg.size if generator else 128
    scale = (a.patch_scale if a.patch_scale is not None
             else geom.get("scale", tcfg.get("patch_scale", 0.25)))
    placement = a.gen_placement or geom.get(
        "placement", tcfg.get("gen_placement", "gradcam"))
    placement_class = (a.gen_placement_class
                       if a.gen_placement_class is not None
                       else tcfg.get("gen_placement_class", 0))
    placement_margin = (a.gen_placement_margin
                        if a.gen_placement_margin is not None
                        else tcfg.get("gen_placement_margin", 0))
    reference_mode = a.gen_reference or tcfg.get("gen_reference", "center")
    loss_fn = a.loss_fn or tcfg.get("loss_fn", "cospgd")
    target_class = (a.target_class if a.target_class is not None
                    else tcfg.get("target_class", 8))
    cam_objective = a.cam_objective or tcfg.get("cam_objective", "attack")
    cam_target = a.cam_target or tcfg.get("cam_target", "pred")
    cam_layer = a.cam_layer if a.cam_layer is not None else tcfg.get("cam_layer", -1)
    cam_module = a.cam_module or tcfg.get("cam_module", "backbone")

    trained_placement = geom.get("placement")
    if trained_placement and placement != trained_placement:
        print(f"[abl ] placement OVERRIDDEN: trained with "
              f"{trained_placement!r}, evaluating with {placement!r}. This is "
              f"a placement-transfer measurement, not a retrained model.")

    model, n_ch, n_active, spec = setup_model(a)
    mean_t, std_t = norm_tensors(device)

    adv_loss = adversarial.build(
        loss_fn, target_class if loss_fn == "ipatch_cospgd" else 8)
    tgt = target_class if loss_fn == "ipatch_cospgd" else None

    cam_loss_name = loss_fn if cam_objective == "attack" else cam_objective
    cam = segmentation_cam.build(
        model, cam_loss_name, target_class, layer=cam_layer, module=cam_module,
        target=cam_target,
        attack_loss=adv_loss if cam_objective == "attack" else None)
    print(f"[cam ] S_seg = -L_{cam_loss_name} vs {cam_target} labels, "
          f"d/d {cam_module}[{cam_layer}]")

    attack = cg.ConditionalAttack(
        model, cam, generator, mean_t, std_t, scale, size,
        placement=placement, placement_class=placement_class,
        method=a.method, reference=reference_mode,
        placement_margin=placement_margin)

    trained_reference = tcfg.get("gen_reference")
    if trained_reference and reference_mode != trained_reference:
        print(f"[abl ] reference OVERRIDDEN: trained with "
              f"{trained_reference!r}, evaluating with {reference_mode!r}. The "
              f"generator is seeing a reference distribution it was not "
              f"trained on — report this as a transfer measurement.")

    ds = CityscapesSeg(a.cityscapes_root, a.split, a.img_h, a.img_w)
    idxs = image_indices(a.images, len(ds))
    print(f"[data] {a.split} split, {len(idxs)} image(s) "
          f"@ {a.img_h}x{a.img_w}, patch {cg.patch_side(a.img_h, scale)}px")

    stem = Path(a.checkpoint).parent.parent.name if a.checkpoint else "baselineA"
    tag = "_".join(x for x in [stem, a.method, f"pl-{placement}",
                               f"ref-{reference_mode}",
                               f"{a.img_h}x{a.img_w}", a.tag] if x)
    out_dir = increment_path(Path(a.out_root) / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    lpips_metric = None if a.no_lpips else cg.build_lpips(device, a.lpips_net)

    m = {k: SegMetric(a.num_classes, device=device)
         for k in ("clean_all", "clean_rem", "adv_all", "adv_rem")}
    per_image, lpips_vals = [], []

    for i in idxs:
        img, label = ds[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        hw = label.shape[-2:]

        # THE ENTIRE TEST-TIME ATTACK: one forward pass. No optimiser, no loop.
        with torch.no_grad():
            out = attack(img, label, deterministic_noise=True)
            lc = upsample_to(out["clean_logits"], hw)
            la = upsample_to(model(out["patched"]), hw)

        fp = out["footprint"]
        pc, pa = lc.argmax(1), la.argmax(1)
        m["clean_all"].update(pc, label)
        m["clean_rem"].update(pc, label, exclude=fp)
        m["adv_all"].update(pa, label)
        m["adv_rem"].update(pa, label, exclude=fp)

        cr = single_image_miou(lc, label, a.num_classes, exclude=fp)
        ar = single_image_miou(la, label, a.num_classes, exclude=fp)
        rates = attack_rates(lc, la, label, fp, tgt)

        patch01 = out["patches"][0].detach()
        ref01 = out["references"][0].detach()
        lp = cg.lpips_distances(lpips_metric, patch01.unsqueeze(0),
                                ref01.unsqueeze(0))
        lpips_vals += lp

        with torch.no_grad():
            sup = ~fp
            att_loss = float(adv_loss(la, label, fp, sup))

        row = {"image": i, "clean_remote": cr, "adv_remote": ar,
               "drop_remote": cr - ar, "attack_loss": att_loss, **rates,
               "placement": list(out["placements"][0]),
               "ASI": asi(patch01), "AGI": agi(patch01), "ADE": ade(patch01),
               "l2_to_reference": float(
                   (patch01 - ref01).pow(2).sum().sqrt()),
               "lpips_to_reference": (lp[0] if lp else None)}
        per_image.append(row)
        print(f"  img {i:4d}: drop_remote {cr-ar:+6.2f}  "
              f"any_flip {rates['any_flip_rate']:5.1f}%  "
              f"place {tuple(out['placements'][0])}"
              + (f"  LPIPS {lp[0]:.4f}" if lp else ""))

        d = out_dir / "images" / f"img{i:04d}"
        if a.save_patches:
            d.mkdir(parents=True, exist_ok=True)
            save_image(patch01.clamp(0, 1).cpu(), d / "patch.png")
            save_image(ref01.clamp(0, 1).cpu(), d / "reference.png")
            save_image(out["cam"][0].cpu(), d / "sensitivity.png")
            torch.save({"patch": patch01.cpu(), "reference": ref01.cpu(),
                        "placement": list(out["placements"][0]),
                        "patch_side": out["patch_side"], "image_index": i,
                        "split": a.split},
                       d / "patch.pt")
        if not a.no_panels:
            cviz.save_conditional_panels(
                img, label, out["patched"], patch01, ref01, out["cam"][0, 0],
                out["clean_logits"], la, out["placements"][0],
                out["patch_side"], mean_t, std_t, d,
                title=f"{a.split} image {i}  method={a.method}")

    # ── aggregate ───────────────────────────────────────────────────────────
    agg = {k: v.compute() for k, v in m.items()}
    agg["drop_remote_dataset"] = agg["clean_rem"] - agg["adv_rem"]
    agg["drop_all_dataset"] = agg["clean_all"] - agg["adv_all"]
    drops = torch.tensor([r["drop_remote"] for r in per_image])
    flips = torch.tensor([r["any_flip_rate"] for r in per_image])
    ciou, aiou = m["clean_rem"].per_class(), m["adv_rem"].per_class()

    lp_stats = cviz.lpips_histogram(
        lpips_vals, out_dir / "lpips_distribution.png",
        title=f"LPIPS(r_i, p_i) — {a.method}")

    print(f"\n{'='*70}")
    print(f"  {len(idxs)} images @ {a.img_h}x{a.img_w}   method={a.method}   "
          f"placement={placement}")
    print(f"  drop_remote  per-image mean {drops.mean():+.2f} "
          f"+/- {drops.std():.2f}  "
          f"(range {drops.min():+.2f} to {drops.max():+.2f})")
    print(f"  any_flip     mean {flips.mean():.1f}% +/- {flips.std():.1f}%")
    print(f"  DATASET drop_remote {agg['drop_remote_dataset']:+.2f}")
    if lp_stats:
        print(f"  LPIPS(r_i,p_i)  median {lp_stats['median']:.4f}  "
              f"IQR [{lp_stats['p25']:.4f}, {lp_stats['p75']:.4f}]  "
              f"max {lp_stats['max']:.4f}")
        print(f"                  (evaluation metric only — never optimised)")
    print(f"\n  per-class IoU (remote), clean -> patched:")
    for c in range(min(a.num_classes, 19)):
        if not torch.isnan(ciou[c]):
            print(f"    {c:2d} {class_name(c):10s}: {ciou[c]:6.2f} -> "
                  f"{aiou[c]:6.2f}  ({aiou[c]-ciou[c]:+.1f})")
    print(f"{'='*70}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "checkpoint": a.checkpoint, "method": a.method,
            "test_time_optimisation": False,
            "config": vars(a),
            "resolved": {"scale": scale, "size": size, "placement": placement,
                         "placement_margin": placement_margin,
                         "reference": reference_mode,
                         "loss_fn": loss_fn, "target_class": tgt,
                         "cam_objective": cam_loss_name,
                         "cam_target": cam_target, "cam_layer": cam_layer},
            "generator_config": (ck.get("generator_config")
                                 if a.method == "generator" else None),
            "trained_placement": trained_placement,
            "aggregate": agg,
            "drop_remote_mean": float(drops.mean()),
            "drop_remote_std": float(drops.std()),
            "any_flip_mean": float(flips.mean()),
            "lpips": lp_stats,
            "cam_degenerate_maps": cam.n_degenerate,
            "per_class_iou": {
                class_name(c): {"clean": float(ciou[c]), "adv": float(aiou[c])}
                for c in range(min(a.num_classes, 19))
                if not torch.isnan(ciou[c])},
            "per_image": per_image}, f, indent=2)
    cam.close()
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
