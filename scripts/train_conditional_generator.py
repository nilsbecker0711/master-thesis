#!/usr/bin/env python
r"""
Dataset-level training of an IMAGE-CONDITIONED adversarial patch generator.

    python scripts/train_conditional_generator.py \
        --arch segformer --cityscapes_root $CS \
        --loss_fn cospgd --gen_placement gradcam --epochs 30

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
    theta* = argmin_theta E_{(x,y)~D} [ L_attack(f(x (+) G_theta(x,r,M)), y)
                                        + lambda_LAP * L_LAP(p_i, r_i) ]

theta is shared across the ENTIRE dataset. The PATCH is not — every image gets
its own p_i = G_theta(x_i, r_i, M_i) from one forward pass. This is NOT a
universal patch, and it is not comparable to one:

  train.py     ONE patch tensor optimised over the dataset, applied unchanged
               to every image. Transfers by construction.
  overfit.py   ONE patch tensor optimised on ONE image. An approximate upper
               bound on what direct optimisation achieves (BASELINE B).
  this script  A shared FUNCTION. At test time an unseen image is attacked with
               a forward pass and NO gradient-based optimisation of the patch.

The last point is the claim the evaluation has to support, so evaluation runs
the generator under torch.no_grad() in eval() mode with deterministic noise.
Nothing in the eval path can optimise a patch even by accident.

HYPOTHESIS UNDER TEST (nothing stronger is claimed)
---------------------------------------------------
We investigate whether image-conditioned generation combined with
segmentation-specific sensitivity localisation and LAP constraints can produce
more perceptually coherent adversarial patches for semantic segmentation.
Whether it does is an empirical question this script exists to answer; it is
not assumed anywhere in the code or the logging.

ABLATIONS (all one flag each)
    A  --gen_placement center        vs   B  --gen_placement gradcam
    C  --gen_lap_alpha <w>           vs   D  --gen_lap_alpha 0 (default)
    E  --gen_cond image | image+ref | image+ref+cam
    F  --cam_objective attack | ce | cospgd | ipatch_cospgd

Existing experiments are untouched: this script shares only add_model_args and
setup_model with train.py/overfit.py, and never constructs a Patch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (add_model_args, add_generator_args, build_generator_config,
                     setup_model)
from patchreach.data.cityscapes import CityscapesSeg, norm_tensors, upsample_to
from patchreach.diagnostics import conditional as cviz, report
from patchreach.losses import adversarial
from patchreach.metrics.miou import SegMetric
from patchreach.patch import conditional_generator as cg
from patchreach.patch import segmentation_cam
from patchreach.patch.lap import asi, agi, ade, magnitude_report
from patchreach.patch.spec import PatchConfig
from patchreach.utils import get_device, seed_everything, increment_path


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Train an image-conditioned adversarial patch generator")
    p.add_argument("--config", type=str, default=None,
                   help="YAML supplying defaults; CLI flags override it.")
    p.add_argument("--mode", default="conditional_generator",
                   choices=["conditional_generator"],
                   help="present so runs are self-describing and so a config "
                        "file names its own attack family. This script only "
                        "implements the one mode; the raw/lap/gan/raw_ganinit "
                        "modes live in train.py and are unchanged.")

    add_model_args(p)
    add_generator_args(p)

    p.add_argument("--loss_fn", choices=["ce", "cospgd", "ipatch_cospgd"],
                   default="cospgd")
    p.add_argument("--target_class", type=int, default=8)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Adam on theta. Much lower than train.py's 5e-3, which "
                        "is a step on PIXELS; this is a step on network "
                        "weights and the usual generator range applies.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--train_images", type=int, default=0,
                   help="0 = the full train split. A small number gives a fast "
                        "pipeline smoke test.")
    p.add_argument("--exclude_footprint", action="store_true", default=True)
    p.add_argument("--val_images", type=int, default=20)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--panel_images", type=str, default="0 2 5",
                   help="val indices to render the full panel for at every "
                        "validation. '' disables.")
    # Same two flags train.py uses, same meaning, so a conditional run leaves
    # the same figure suite behind as every other experiment block.
    p.add_argument("--diag_image", type=int, default=2,
                   help="val image used for the post-training figure suite "
                        "(ERF probe, reach curves, confusion/margin).")
    p.add_argument("--no_diagnostics", action="store_true",
                   help="skip the post-training figure suite. The ERF probe "
                        "costs n_probes extra forward passes.")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--no_lpips", action="store_true",
                   help="skip the perceptual metric. It is EVALUATION ONLY in "
                        "either case and never enters the objective.")
    p.add_argument("--eval_baseline_reference", action="store_true", default=True,
                   help="also evaluate BASELINE A (p_i = r_i, no generator) "
                        "through the identical pipeline")

    p.add_argument("--out_root", type=str, default="results/conditional")
    p.add_argument("--tag", type=str, default="")
    return p


def load_config(args):
    """YAML supplies defaults; explicit CLI flags win. Same rule as train.py."""
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
    """Readable, sortable, unique. Encodes the axes an ablation varies."""
    bits = ["condgen", a.arch, a.loss_fn, f"{a.img_h}x{a.img_w}",
            f"pl-{a.gen_placement}", f"cond-{a.gen_cond.replace('+', '-')}",
            f"res-{a.gen_residual}"]
    if a.loss_fn == "ipatch_cospgd":
        bits.append(f"cls{a.target_class}")
    if a.gen_lap_alpha or a.gen_lap_beta or a.gen_lap_gamma:
        bits.append(f"lap-a{a.gen_lap_alpha:g}")
    if a.cam_objective != "attack":
        bits.append(f"cam-{a.cam_objective}")
    if a.cam_target != "pred":
        bits.append(f"camtgt-{a.cam_target}")
    if a.tag:
        bits.append(a.tag)
    bits.append(f"s{a.seed}")
    return "_".join(bits)


# ═════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(attack: cg.ConditionalAttack, loader, device, K: int,
             adv_loss, exclude_footprint: bool, target_class=None,
             lpips_metric=None) -> dict:
    r"""
    Clean vs patched mIoU over the val subset, ALL and REMOTE, plus the LAP
    realism indicators and the LPIPS distribution.

    NO GRADIENT-BASED OPTIMISATION HAPPENS HERE. The generator runs in eval()
    under no_grad with the prior-mean noise, so what is measured is exactly
    x_test -> G_theta* -> p_test. The CAM internally needs grad w.r.t. its own
    activations and re-enables it locally; it detaches before returning, and
    the frozen model's parameters never receive a gradient.

    REMOTE is the headline: drop_all conflates occlusion with adversarial
    influence, drop_remote excludes the footprint and measures only the latter.
    Same convention as train.py and evaluate.py.
    """
    was_training = attack.generator.training if attack.generator else False
    if attack.generator is not None:
        attack.generator.eval()

    ms = {k: SegMetric(K, device=device)
          for k in ("clean_all", "clean_rem", "adv_all", "adv_rem")}
    flips = hits = n_flip = n_hit = 0
    losses, lpips_vals = [], []
    realism = {"ASI": [], "AGI": [], "ADE": []}
    l2_to_ref, placements = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        hw = labels.shape[-2:]

        with torch.no_grad():
            out = attack(imgs, labels, deterministic_noise=True)
            lc = upsample_to(out["clean_logits"], hw)
            la = upsample_to(attack.model(out["patched"]), hw)

            fp = out["footprint"]
            sup = (~fp) if exclude_footprint else None
            losses.append(float(adv_loss(la, labels, fp, sup)))

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

        # Per-image realism, on the patch that image actually received.
        p_batch, r_batch = out["patches"].detach(), out["references"].detach()
        for i in range(p_batch.shape[0]):
            realism["ASI"].append(asi(p_batch[i]))
            realism["AGI"].append(agi(p_batch[i]))
            realism["ADE"].append(ade(p_batch[i]))
            l2_to_ref.append(float(
                (p_batch[i] - r_batch[i]).pow(2).sum().sqrt()))
        placements += [list(x) for x in out["placements"]]
        lpips_vals += cg.lpips_distances(lpips_metric, p_batch, r_batch)

    res = {k: m.compute() for k, m in ms.items()}
    res["drop_all"] = res["clean_all"] - res["adv_all"]
    res["drop_remote"] = res["clean_rem"] - res["adv_rem"]
    res["any_flip_rate"] = 100.0 * flips / max(n_flip, 1)
    res["attack_loss"] = sum(losses) / max(len(losses), 1)
    if target_class is not None:
        res["target_hit_rate"] = 100.0 * hits / max(n_hit, 1)

    n = max(len(l2_to_ref), 1)
    res["rationality"] = {k: sum(v) / max(len(v), 1) for k, v in realism.items()}
    res["l2_to_reference_mean"] = sum(l2_to_ref) / n
    res["placements"] = placements
    res["lpips"] = lpips_vals            # the DISTRIBUTION, not just a mean

    if was_training and attack.generator is not None:
        attack.generator.train()
    return res


def summarise(tag: str, ev: dict, target_class=None, log=print):
    extra = (f"  hit={ev['target_hit_rate']:.1f}%" if target_class is not None
             else f"  flip={ev['any_flip_rate']:.1f}%")
    lp = ev.get("lpips") or []
    lp_s = (f"  LPIPS={sum(lp)/len(lp):.4f}" if lp else "")
    log(f"  [{tag:<9s}] remote {ev['adv_rem']:6.2f} "
        f"({ev['drop_remote']:+6.2f}){extra}"
        f"  ASI={ev['rationality']['ASI']:.3f} ADE={ev['rationality']['ADE']:.3f}"
        f"{lp_s}")


# ═════════════════════════════════════════════════════════════════════════════
#  Panels
# ═════════════════════════════════════════════════════════════════════════════

def frozen_patch_for(attack, dataset, idx, device, scale):
    """
    (img, label, Patch, ConditionalAttack output) for ONE validation image.

    The generator's result for a single image IS a patch plus a placement, so
    it is frozen into a real Patch and handed to the EXISTING diagnostic suite
    unchanged — same figures, same ERF probe, same reach curves as every other
    patch mode, and therefore directly comparable with them.
    """
    img, label = dataset[idx]
    img = img.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)
    was_training = attack.generator.training if attack.generator else False
    if attack.generator is not None:
        attack.generator.eval()
    with torch.no_grad():
        out = attack(img, label, deterministic_noise=True)
    if was_training and attack.generator is not None:
        attack.generator.train()
    patch = cg.as_patch(out["patches"][0], out["placements"][0], scale, device,
                        attack.mean_t, attack.std_t,
                        reference=out["references"][0])
    return img, label, patch, out


def per_class_panels(attack, model, dataset, indices, device, out_dir, a, tgt,
                     mean_t, std_t, log=print):
    """
    The standard labelled panels + per-class IoU chart, per image.

    report.panels_for_images() takes ONE patch for the whole list, which is the
    assumption this attack breaks — so it is called once per image with that
    image's own frozen patch. img_h is passed as None deliberately: the
    generator already resolved placement from its sensitivity map, and letting
    panels_for_images re-resolve it would overwrite that with the `center`
    default and silently mis-report where the patch actually sat.
    """
    out = {}
    for i in indices:
        _, _, patch, _ = frozen_patch_for(attack, dataset, i, device,
                                          a.patch_scale)
        out.update(report.panels_for_images(
            model, dataset, [i], patch, out_dir, mean_t, std_t,
            a.num_classes, tgt, None, None, log=log))
    return out


def run_diagnostics(attack, model, dataset, idx, device, out_dir, a, tgt,
                    mean_t, std_t, log=print):
    """
    The full diagnostic suite on ONE image — ERF probe, reach curves, confusion
    or margin suite, entropy, winner margin.

    The ERF probe overwrites the frozen patch's param with random noise and
    restores it, so it measures the architecture's reach AT THE PLACEMENT THE
    GENERATOR CHOSE. Under `--gen_placement gradcam` that is the honest control:
    it answers "how far could ANY patch reach from here", separating the
    geometric ceiling from what this generator achieved.
    """
    img, label, patch, out = frozen_patch_for(attack, dataset, idx, device,
                                              a.patch_scale)
    log(f"\n[diag] val image #{idx}  placement {tuple(out['placements'][0])}  "
        f"patch {out['patch_side']}px")
    return report.run(model, img, label, patch, out_dir, a.loss_fn,
                      a.num_classes, tgt, mean_t, std_t, log=log)


def render_panels(attack, dataset, indices, device, out_dir, mean_t, std_t,
                  title="", log=print):
    was_training = attack.generator.training if attack.generator else False
    if attack.generator is not None:
        attack.generator.eval()

    for i in indices:
        img, label = dataset[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        with torch.no_grad():
            out = attack(img, label, deterministic_noise=True)
            adv_logits = attack.model(out["patched"])
        cviz.save_conditional_panels(
            img, label, out["patched"], out["patches"][0], out["references"][0],
            out["cam"][0, 0], out["clean_logits"], adv_logits,
            out["placements"][0], out["patch_side"], mean_t, std_t,
            Path(out_dir) / f"img{i}", title=f"val image {i}  {title}")
    log(f"  [panels   ] {len(indices)} image(s) -> {out_dir}/")

    if was_training and attack.generator is not None:
        attack.generator.train()


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    a = load_config(build_parser().parse_args())
    seed_everything(a.seed)
    device = get_device()

    out_dir = increment_path(Path(a.out_root) / run_id(a))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(a), f, indent=2)
    print(f"\n### {out_dir} ###")

    model, n_ch, n_active, spec = setup_model(a)
    if n_ch != a.num_classes:
        print(f"[head] NOTE: head emits {n_ch} but num_classes={a.num_classes}; "
              f"trainIds 0..18 are used, the rest are inert")

    mean_t, std_t = norm_tensors(device)
    train_full = CityscapesSeg(a.cityscapes_root, "train", a.img_h, a.img_w)
    val_full = CityscapesSeg(a.cityscapes_root, "val", a.img_h, a.img_w)
    train_ds = (Subset(train_full, list(range(min(a.train_images,
                                                  len(train_full)))))
                if a.train_images else train_full)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(
        Subset(val_full, list(range(min(a.val_images, len(val_full))))),
        batch_size=1, shuffle=False, num_workers=2)
    print(f"[data] train {len(train_ds)} imgs / {len(train_loader)} batches, "
          f"val {min(a.val_images, len(val_full))} imgs "
          f"@ {a.img_h}x{a.img_w}")

    # ── the attack objective, shared by training, the CAM and evaluation ─────
    adv_loss = adversarial.build(
        a.loss_fn, a.target_class if a.loss_fn == "ipatch_cospgd" else 8)
    tgt = a.target_class if a.loss_fn == "ipatch_cospgd" else None

    cam_objective = a.loss_fn if a.cam_objective == "attack" else a.cam_objective
    cam = segmentation_cam.build(
        model, cam_objective, a.target_class, layer=a.cam_layer,
        module=a.cam_module, target=a.cam_target,
        attack_loss=adv_loss if a.cam_objective == "attack" else None)
    print(f"[cam ] S_seg = -L_{cam_objective} vs {a.cam_target} labels, "
          f"d/d {a.cam_module}[{a.cam_layer}]")
    if a.cam_target == "gt":
        print("[cam ] NOTE: cam_target=gt makes this a LABEL-AWARE attacker — "
              "strictly stronger than --placement semantic, which reads only "
              "the clean prediction. Declare this in the writeup.")

    # ── the generator: the ONLY trainable component ─────────────────────────
    gcfg = build_generator_config(a)
    generator = cg.ConditionalPatchGenerator(gcfg).to(device)
    gcfg.describe()
    p_side = cg.patch_side(a.img_h, a.patch_scale)
    print(f"[gen ] theta      : {generator.n_parameters():,} shared parameters "
          f"(NOT per-image — the patch is a function, not a tensor)")
    print(f"[gen ] geometry   : {gcfg.size}px generated -> {p_side}px rendered "
          f"(scale {a.patch_scale})")
    print(f"[gen ] placement  : {a.gen_placement}")
    for q in model.parameters():
        assert not q.requires_grad, "the segmentation model must stay frozen"

    attack = cg.ConditionalAttack(
        model, cam, generator, mean_t, std_t, a.patch_scale, gcfg.size,
        placement=a.gen_placement, placement_class=a.gen_placement_class,
        placement_xy=tuple(a.gen_placement_xy), method="generator")

    lpips_metric = None if a.no_lpips else cg.build_lpips(device, a.lpips_net)

    opt = torch.optim.Adam(generator.parameters(), lr=a.lr, betas=(0.5, 0.999),
                           amsgrad=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3,
                                                       factor=0.5)

    panel_idx = [int(x) for x in a.panel_images.replace(",", " ").split()
                 if int(x) < len(val_full)] if a.panel_images.strip() else []

    # ── BASELINE A, measured BEFORE training ────────────────────────────────
    # The zero-initialised generator emits Delta == 0, so p_i == r_i exactly.
    # Baseline A is therefore the generator's own starting point and the
    # comparison is exact rather than approximate.
    baseline_a = None
    if a.eval_baseline_reference:
        print("\n[base] BASELINE A — p_i = r_i, raw centre crop, no generator")
        ref_attack = cg.ConditionalAttack(
            model, cam, None, mean_t, std_t, a.patch_scale, gcfg.size,
            placement=a.gen_placement, placement_class=a.gen_placement_class,
            placement_xy=tuple(a.gen_placement_xy), method="reference")
        baseline_a = evaluate(ref_attack, val_loader, device, a.num_classes,
                              adv_loss, a.exclude_footprint, tgt, lpips_metric)
        summarise("baselineA", baseline_a, tgt)
        if panel_idx:
            render_panels(ref_attack, val_full, panel_idx[:1], device,
                          out_dir / "panels" / "baseline_reference",
                          mean_t, std_t, title="BASELINE A (p_i = r_i)")

    # ── training ────────────────────────────────────────────────────────────
    history, best, checked, t0 = [], -1e9, False, time.time()
    generator.train()

    for epoch in range(1, a.epochs + 1):
        running_att = running_lap = 0.0
        for imgs, labels in tqdm(train_loader,
                                 desc=f"epoch {epoch}/{a.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()

            out = attack(imgs, labels)               # steps 1-6
            logits = upsample_to(model(out["patched"]), labels.shape[-2:])  # 7

            fp = out["footprint"]
            sup = (~fp) if a.exclude_footprint else None
            l_att = adv_loss(logits, labels, fp, sup)                       # 8
            lap = cg.lap_terms(out["patches"], out["references"],           # 9
                               a.gen_lap_alpha, a.gen_lap_beta,
                               a.gen_lap_gamma)
            total = l_att + lap["total"]                                    # 10

            if not torch.isfinite(total):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch}; valid remote px this "
                    f"batch = {int(((labels != 255) & (~fp)).sum())}")

            total.backward()                                                # 11

            if not checked:
                # The graph is longer here than in train.py — it runs back
                # through the compositor AND the whole generator — so verify it
                # once rather than discovering a dead head after an epoch.
                gs = [q.grad for q in generator.parameters() if q.grad is not None]
                if not gs:
                    raise RuntimeError(
                        "no generator parameter received a gradient — the "
                        "graph from theta to the attack loss is broken")
                gm = sum(float(g.abs().mean()) for g in gs) / len(gs)
                print(f"\n[grad] theta abs mean {gm:.3e} over {len(gs)} tensors")
                head = generator.head.weight.grad
                print(f"[grad] output head    {float(head.abs().mean()):.3e}"
                      + ("   <- ZERO: nothing reaches the patch"
                         if float(head.abs().mean()) == 0.0 else ""))
                # Same calibration train.py prints, on the SAME quantities the
                # loop optimises. LAP terms are sums; the attack loss is a mean.
                magnitude_report(
                    l_att.item(), out["patches"][0].detach(),
                    out["references"][0].detach(), None,
                    PatchConfig(mode="lap", size=gcfg.size,
                                lap_alpha=a.gen_lap_alpha,
                                lap_beta=a.gen_lap_beta,
                                lap_gamma=a.gen_lap_gamma,
                                reference="<per-image centre crop>"))
                checked = True

            opt.step()                                                      # 12
            running_att += float(l_att)
            running_lap += float(lap["total"])

        nb = max(len(train_loader), 1)
        running_att /= nb
        running_lap /= nb
        sched.step(running_att)
        print(f"[epoch {epoch:3d}] {a.loss_fn}={running_att:.5f}  "
              f"lap={running_lap:.5f}  lr={opt.param_groups[0]['lr']:.2e}")

        if epoch % a.val_every == 0 or epoch == 1:
            ev = evaluate(attack, val_loader, device, a.num_classes, adv_loss,
                          a.exclude_footprint, tgt, lpips_metric)
            summarise("generator", ev, tgt)
            history.append({"epoch": epoch, "loss_attack": running_att,
                            "loss_lap": running_lap,
                            **{k: v for k, v in ev.items()
                               if k not in ("lpips", "placements")}})
            if ev["drop_remote"] > best:
                best = ev["drop_remote"]
                cg.save_checkpoint(
                    out_dir / "checkpoints" / "best.pt", generator, opt, epoch,
                    vars(a), {"scale": a.patch_scale, "size": gcfg.size,
                              "placement": a.gen_placement,
                              "img_h": a.img_h, "img_w": a.img_w},
                    extra={"val_drop_remote": best})
                print(f"           * new best drop_remote {best:+.2f}")
            if panel_idx:
                render_panels(attack, val_full, panel_idx, device,
                              out_dir / "panels" / f"epoch{epoch:04d}",
                              mean_t, std_t, title=f"epoch {epoch}")

    # ── final ───────────────────────────────────────────────────────────────
    final = evaluate(attack, val_loader, device, a.num_classes, adv_loss,
                     a.exclude_footprint, tgt, lpips_metric)

    # ── calibration for a LAP-constrained rerun ─────────────────────────────
    # Same reasoning as train.py's end-of-run report, and it applies MORE
    # strongly here: the head is zero-initialised, so p_i == r_i exactly at
    # step 0 and L_rat is identically 0. Any alpha derived from the step-1
    # report is a division by ~0. The meaningful scale is the DRIFT training
    # produced, measured here on a real generated patch.
    generator.eval()
    with torch.no_grad():
        cal_img, cal_lbl = val_full[0]
        cal = attack(cal_img.unsqueeze(0).to(device),
                     cal_lbl.unsqueeze(0).to(device), deterministic_noise=True)
    print("\n[lap] END-OF-RUN magnitudes — set --gen_lap_alpha from THIS rat "
          "row,\n      not from the step-1 report (where L_rat is 0 because "
          "the zero-init\n      head makes p_i == r_i exactly):")
    magnitude_report(final["attack_loss"], cal["patches"][0],
                     cal["references"][0], None,
                     PatchConfig(mode="lap", size=gcfg.size,
                                 lap_alpha=a.gen_lap_alpha,
                                 lap_beta=a.gen_lap_beta,
                                 lap_gamma=a.gen_lap_gamma,
                                 reference="<per-image centre crop>"))
    cg.save_checkpoint(out_dir / "checkpoints" / "final.pt", generator, opt,
                       a.epochs, vars(a),
                       {"scale": a.patch_scale, "size": gcfg.size,
                        "placement": a.gen_placement,
                        "img_h": a.img_h, "img_w": a.img_w})

    lp_stats = cviz.lpips_histogram(final.get("lpips") or [],
                                    out_dir / "lpips_distribution.png")

    results = {
        "run_id": out_dir.name,
        "method": "conditional_generator",
        "threat_model": ("shared theta, per-image patch p_i = G_theta(x_i, "
                         "r_i, M_i); no test-time optimisation of p_i"),
        "config": vars(a),
        "generator_config": asdict(gcfg),
        "generator_parameters": generator.n_parameters(),
        "backbone_channels": n_ch, "backbone_active_channels": n_active,
        "patch_side_px": p_side,
        "cam_degenerate_maps": cam.n_degenerate,
        "final": {k: v for k, v in final.items() if k != "lpips"},
        "lpips": lp_stats,
        "lpips_values": final.get("lpips") or [],
        "baseline_A_reference": (
            {k: v for k, v in baseline_a.items() if k != "lpips"}
            if baseline_a else None),
        "best_drop_remote": best,
        "wall_clock_s": time.time() - t0,
        "history": history,
    }
    if panel_idx:
        render_panels(attack, val_full, panel_idx, device,
                      out_dir / "panels" / "final", mean_t, std_t,
                      title="final")

    # ── labelled figures, same suite as train.py ────────────────────────────
    # Panel (b) is the CLEAN PREDICTION rather than ground truth: the attack's
    # effect is (clean pred -> adv pred), and scoring against GT would fold the
    # model's own errors into what reads as attack damage.
    if panel_idx:
        print(f"\n[panels] rendering {len(panel_idx)} labelled image(s)")
        results["per_class_iou"] = {
            str(k): v for k, v in per_class_panels(
                attack, model, val_full, panel_idx, device,
                out_dir / "panels" / "labelled", a, tgt, mean_t,
                std_t).items()}

    # ── post-training figure suite ──────────────────────────────────────────
    # Numbers alone do not show WHERE the attack acted or WHAT it converted
    # pixels into, and for this attack family they also cannot show whether the
    # sensitivity-guided placement bought anything. The ERF probe answers the
    # second question directly.
    if not a.no_diagnostics and a.diag_image < len(val_full):
        print("\n" + "=" * 70)
        print(f" FIGURES on val image #{a.diag_image}")
        print("=" * 70)
        results["diagnostics"] = run_diagnostics(
            attack, model, val_full, a.diag_image, device,
            out_dir / "diagnostics", a, tgt, mean_t, std_t)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    cam.close()

    # ── report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  CONDITIONAL GENERATOR — {generator.n_parameters():,} shared params")
    print(f"  placement {a.gen_placement}   cond {a.gen_cond}   "
          f"residual {a.gen_residual}")
    print("-" * 70)
    if baseline_a:
        print(f"  BASELINE A  p_i = r_i        drop_remote "
              f"{baseline_a['drop_remote']:+7.2f}   "
              f"flip {baseline_a['any_flip_rate']:5.1f}%")
    print(f"  BASELINE C  p_i = G_theta*   drop_remote "
          f"{final['drop_remote']:+7.2f}   flip {final['any_flip_rate']:5.1f}%")
    if baseline_a:
        print(f"              generator contribution "
              f"{final['drop_remote'] - baseline_a['drop_remote']:+7.2f} mIoU")
    if lp_stats:
        print(f"  LPIPS(r_i,p_i)  median {lp_stats['median']:.4f}  "
              f"IQR [{lp_stats['p25']:.4f}, {lp_stats['p75']:.4f}]  "
              f"max {lp_stats['max']:.4f}   (evaluation only)")
    print(f"  ASI {final['rationality']['ASI']:.4f} (lower=natural)  "
          f"AGI {final['rationality']['AGI']:.4f}  "
          f"ADE {final['rationality']['ADE']:.4f} (higher=natural)")
    if cam.n_degenerate:
        print(f"  WARNING: {cam.n_degenerate} sensitivity map(s) were "
              f"degenerate (ReLU suppressed every channel) and fell back to "
              f"centre placement. Try a different --cam_layer.")
    print("-" * 70)
    print("  BASELINE B (per-image direct optimisation, the approximate upper")
    print("  bound) is scripts/overfit.py and is NOT run here — it optimises a")
    print("  patch tensor per image and is a different procedure. See the")
    print("  command in the module docstring of export_conditional_patches.py.")
    print(f"  -> {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
