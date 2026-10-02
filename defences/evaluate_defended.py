#!/usr/bin/env python
r"""
Evaluate a trained patch WITH a defence in front of the model. NO TRAINING.

    # the 2x2: clean/attacked x undefended/defended, on the fixed ten
    python defences/evaluate_defended.py --defence sac --sac_ckpt $SAC/coco_at.pth \
        --checkpoint results/overfit/<run>/best.pt --arch segformer_b0 \
        --cityscapes_root $CS --img_h 512 --img_w 1024 --from_image

    # the clean-image price alone, over the whole val split (no --checkpoint)
    python defences/evaluate_defended.py --defence sac --sac_ckpt $SAC/coco_at.pth \
        --arch segformer_b0 --cityscapes_root $CS --img_h 512 --img_w 1024 \
        --images all

THIS SCRIPT EXISTS SO scripts/ STAYS UNTOUCHED. It imports the shared plumbing
(`add_model_args`, `setup_model`, `make_dataset`, `image_indices`) from
scripts/_common.py without modifying it, then wraps the model it gets back. No
existing entry point gains a flag, so no existing run changes by a single digit.

TWO MODES, ONE SETUP PATH:
  --checkpoint given    the full 2x2. Four forwards per image.
  --checkpoint omitted  CLEAN-COST mode. No patch is composited, so every pixel
                        the defence erases is a false positive by construction
                        and the mIoU delta against your existing undefended
                        clean_baseline.py run is the price of the defence.

WHY THE 2x2 AND NOT JUST A DEFENDED ATTACK NUMBER. A defence can recover mIoU
while localising nothing (it blanked enough of the frame that the patch went with
it) and it can localise perfectly while recovering nothing. It can also buy both
by damaging every clean image it sees. One defended number cannot separate those;
four can.

BOTH COLUMNS COME FROM ONE MODEL INSTANCE. setup_model is called once and
DefendedSegModel.bypassed() turns the defence off for the undefended pass, so the
two columns share weights, images and process. Rebuilding the model to get the
undefended column would reintroduce every load-order and seeding difference the
comparison exists to exclude, and cost a second checkpoint load per architecture.

GRADIENTS ARE REFUSED. Every defence here thresholds somewhere, so the gradient
through it is zero. An attack optimised through one does not crash: it gets no
signal, the patch stops moving, and the run reports a robust model with nothing
in the logs to say otherwise. This script never differentiates, the training
entry points have no --defence flag to pass, and DefendedSegModel.forward()
raises if a grad-carrying tensor reaches it anyway. An ADAPTIVE attack through
the defence (SAC ships a straight-through estimator for it) is a different threat
model and separate work.

THE PER-IMAGE SEQUENCE IS COPIED FROM scripts/evaluate.py AND MUST TRACK IT:
resolve_placement against the UNDEFENDED clean prediction, then
set_reference_from_image, then apply. Semantic and gradcam placement read the
clean prediction and training read it from the undefended model, so resolving
against a defended prediction would move the patch for the defended column only
and the two columns would describe different attacks. test_defences.py pins the
ordering; it cannot notice a change made only in evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import add_model_args, setup_model, make_dataset, image_indices
from patchreach.data.cityscapes import class_name, norm_tensors, upsample_to
from patchreach.metrics.miou import (SegMetric, compare, single_image_miou,
                                     attack_rates)
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, seed_everything, increment_path

from defences import DEFENCES, DefendedSegModel, build_defence
from defences.detection import (aggregate_detection, mask_detection,
                                score_stats)
from defences.sac.defence import SAC_SQUARE_SIZES

NAN = float("nan")


def add_defence_args(p):
    g = p.add_argument_group("defence (evaluation only)")
    g.add_argument("--defence", default="sac", choices=sorted(DEFENCES),
                   help="input-space patch defence applied before the model. "
                        "'none' adds no wrapper and reduces this script to a "
                        "plain evaluation.")
    g.add_argument("--sac_ckpt", default=None,
                   help="SAC patch segmenter weights (their ckpts/coco_at.pth, "
                        "4.4 MB). Required for --defence sac.")
    g.add_argument("--sac_base_filter", type=int, default=16,
                   help="16 for coco_at.pth, 64 for the APRICOT-Mask checkpoint")
    g.add_argument("--sac_square_sizes", type=int, nargs="+", default=None,
                   help="candidate square sides for shape completion. Default "
                        "is the list SAC's released code actually executes, "
                        "[100 75 50 25]. Passing the TRUE patch side is an "
                        "ORACLE setting, NOT an upper bound — measured, it "
                        "lowers precision. Say which you ran.")
    g.add_argument("--sac_image_scale", type=float, default=1.0,
                   help="resize before the segmenter. 1.0 is SAC's COCO "
                        "setting; use 0.5 with the APRICOT-Mask checkpoint.")
    g.add_argument("--sac_no_complete", action="store_true",
                   help="SEGMENT only, skip shape completion. The ablation that "
                        "attributes recovery to the segmenter rather than to "
                        "the square prior.")
    g.add_argument("--jedi_calibration", default=None,
                   help="JSON of re-measured entropy constants for --defence "
                        "jedi. THE MOST IMPORTANT KNOB IT HAS: the defaults were "
                        "measured by the authors on PASCAL/INRIA-scale photos "
                        "with YOLO patches, and a Cityscapes dashcam frame has a "
                        "different entropy distribution. Write one with "
                        "defences.jedi.measure_entropy_stats(). Run the stock "
                        "constants FIRST so the recalibration is a measured "
                        "delta rather than an assumption.")
    g.add_argument("--jedi_fill", default="inpaint",
                   choices=["inpaint", "grey", "zero"],
                   help="how the located region is removed. 'inpaint' (cv2 "
                        "TELEA) is closest to their inpaintCoherent; 'zero' is "
                        "what SAC does and is therefore the apples-to-apples "
                        "setting when comparing the two defences.")
    g.add_argument("--jedi_no_blobs", action="store_true",
                   help="skip the small-blob cleanup. The ablation separating "
                        "the entropy threshold from the 0.5%%-of-frame area "
                        "floor, which at 512x1024 is a 51px side and is a hard "
                        "detection limit on small patches.")
    # The optional mask autoencoder is deliberately NOT a flag. Their weights are
    # not released and the topology is resolution-locked, so any usable one has
    # to be TRAINED at our resolution on our patch geometry — a square prior,
    # which is a modelling decision deserving its own script, not a switch.
    # Reach it as JediDefence(autoencoder=...) from defences.jedi.
    return p


def defence_tag(a) -> str:
    """Short slug for output paths. Empty when no defence is active."""
    if a.defence == "none":
        return ""
    if a.defence == "sac":
        bits = ["sac"]
        if a.sac_no_complete:
            bits.append("nocomp")
        if a.sac_square_sizes:
            bits.append("sq" + "-".join(str(s) for s in a.sac_square_sizes))
        if a.sac_image_scale != 1.0:
            bits.append(f"s{a.sac_image_scale:g}")
        return "_".join(bits)
    if a.defence == "jedi":
        bits = ["jedi"]
        if a.jedi_fill != "inpaint":
            bits.append(a.jedi_fill)
        if a.jedi_no_blobs:
            bits.append("noblob")
        # The calibration MUST appear in the path. A run on the authors'
        # constants and a run on ours are different measurements and must never
        # land on the same output directory.
        bits.append("cal-" + (Path(a.jedi_calibration).stem
                              if a.jedi_calibration else "stock"))
        return "_".join(bits)
    return a.defence


def wrap(a, model, device):
    """Wrap `model` in DefendedSegModel. Returns it untouched for 'none'."""
    if a.defence == "none":
        return model
    kw = {}
    if a.defence == "sac":
        if not a.sac_ckpt:
            raise SystemExit(
                "\n--defence sac needs --sac_ckpt. The patch segmenter weights "
                "are not part of patch-reach:\n\n"
                "  bash defences/sac/run_sac.sh --fetch\n")
        kw = dict(ckpt=a.sac_ckpt, base_filter=a.sac_base_filter,
                  square_sizes=(a.sac_square_sizes or SAC_SQUARE_SIZES),
                  image_scale=a.sac_image_scale,
                  complete=not a.sac_no_complete)
    if a.defence == "jedi":
        kw = dict(calibration=a.jedi_calibration, fill=a.jedi_fill,
                  blobs=not a.jedi_no_blobs)
    mean_t, std_t = norm_tensors(device)
    defence = build_defence(a.defence, device=device, **kw)
    wrapped = DefendedSegModel(model, defence, mean_t, std_t).to(device)
    wrapped.describe()
    return wrapped


def load_patch(a, device, mean_t, std_t):
    """
    The patch under test, with evaluate.py's mode='csf' refusal kept verbatim.

    A csf checkpoint stores the parameter and the placement, never the BASE —
    the base is the image region the patch covers. Rendered with reference=None
    it takes the 0.5 grey fallback, the patch becomes a visible square with an
    invisible texture, and every number below would silently describe that
    square instead of the residual under test. Which, with a defence in front,
    would also hand the segmenter a far easier target than the real patch.
    """
    ck = torch.load(a.checkpoint, map_location="cpu")
    pcfg = ck["config"]
    G = None
    if pcfg.get("mode") in ("gan", "raw_ganinit"):
        from GANLatentDiscovery.loading import load_from_dir
        from GANLatentDiscovery.utils import is_conditional
        _, G, _ = load_from_dir(
            "./GANLatentDiscovery/models/pretrained/deformators/BigGAN/",
            G_weights="./GANLatentDiscovery/models/pretrained/generators/"
                      "BigGAN/G_ema.pth")
        if is_conditional(G):
            G.set_classes(259)
        G.eval().to(device)
    if pcfg.get("mode") == "csf" and not a.from_image:
        raise SystemExit(
            "\nThis checkpoint is mode='csf'. Its base is the image region the\n"
            "patch covers, and that base is NOT stored in the checkpoint — only\n"
            "the parameter and the placement are. Re-run with --from_image so\n"
            "the base is rebuilt from each evaluated image, exactly as\n"
            "overfit.py --from_image did during training.\n\n"
            "Without it the base is flat grey, the patch becomes a visible\n"
            "square with an invisible texture, and both the drop AND the\n"
            "defence's localisation numbers would describe that square.\n")
    return Patch.load(a.checkpoint, device, mean_t, std_t, generator=G), pcfg


def main():
    p = add_defence_args(add_model_args(argparse.ArgumentParser()))
    p.add_argument("--checkpoint", default=None,
                   help="the patch to evaluate. OMIT for clean-cost mode: no "
                        "patch is composited and the defence's false-positive "
                        "cost is measured on clean images alone.")
    p.add_argument("--images", default="fixed10",
                   help="'fixed10' | 'all' | '2 5 45'")
    p.add_argument("--target_class", type=int, default=None)
    p.add_argument("--from_image", action="store_true",
                   help="rebuild the patch base from the image being evaluated, "
                        "as overfit.py --from_image did. REQUIRED for "
                        "mode='csf'.")
    p.add_argument("--figures", action="store_true",
                   help="per-image purified frame + predicted mask. The fastest "
                        "way to catch a segmenter firing in the wrong place.")
    p.add_argument("--out_root", default="results/defence")
    p.add_argument("--tag", default="T24_sac")
    a = p.parse_args()

    seed_everything(a.seed)
    device = get_device()
    model, n_ch, n_act, spec = setup_model(a)
    model = wrap(a, model, device)
    mean_t, std_t = norm_tensors(device)

    defended = isinstance(model, DefendedSegModel)
    bypass = model.bypassed if defended else nullcontext
    attacked = a.checkpoint is not None

    patch = pcfg = None
    if attacked:
        patch, pcfg = load_patch(a, device, mean_t, std_t)
        print(f"\n[patch] loaded {a.checkpoint}")
        patch.describe(a.img_h, a.img_w)
    else:
        print("\n[mode ] CLEAN-COST: no patch. Every pixel the defence erases "
              "is a false positive.\n        Compare this mIoU against your "
              "undefended clean_baseline.py run for\n        the same arch and "
              "resolution.")

    ds = make_dataset(a, "val")
    idxs = image_indices(a.images, len(ds))

    name = Path(a.checkpoint).parent.name if attacked else f"clean_{a.arch}"
    tag = "_".join(x for x in [name, f"eval{a.img_h}x{a.img_w}",
                               defence_tag(a), a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = ["clean_all", "clean_rem"]
    if attacked:
        keys += ["adv_all", "adv_rem"]
    if defended:
        keys += ["dclean_all", "dclean_rem"]
        if attacked:
            keys += ["dadv_all", "dadv_rem"]
    m = {k: SegMetric(a.num_classes, device=device) for k in keys}
    per_image = []
    det_adv_rows, det_raw_rows, det_clean_rows = [], [], []

    for i in idxs:
        img, label = ds[i]
        img, label = img.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
        hw = label.shape[-2:]
        row = {"image": i}

        with torch.no_grad():
            with bypass():
                lc = upsample_to(model(img), hw)
            m["clean_all"].update(lc.argmax(1), label)

            fp = None
            if attacked:
                # ORDER MATTERS AND MATCHES evaluate.py. See the module
                # docstring: placement off the undefended clean prediction,
                # reference copied at the RESOLVED placement, then composite.
                patch.resolve_placement(a.img_h, a.img_w, lc.argmax(1)[0])
                if a.from_image:
                    patch.set_reference_from_image(img, mean_t, std_t)
                patched, fp = patch.apply(img)
                with bypass():
                    la = upsample_to(model(patched), hw)

            if defended:
                dca = upsample_to(model(img), hw)
                det_clean = mask_detection(model.last.mask, None)
                det_clean["raw_fp_area_frac"] = mask_detection(
                    model.last.raw_mask, None)["fp_area_frac"]
                det_clean.update(score_stats(model.last.prob))
                det_clean_rows.append(det_clean)
                if attacked:
                    daa = upsample_to(model(patched), hw)
                    det = mask_detection(model.last.mask, fp)
                    det_raw = mask_detection(model.last.raw_mask, fp)
                    det.update(score_stats(model.last.prob, fp))
                    det_adv_rows.append(det)
                    det_raw_rows.append(det_raw)

        m["clean_rem"].update(lc.argmax(1), label, exclude=fp)
        cr = single_image_miou(lc, label, a.num_classes, exclude=fp)
        row["clean_remote"] = cr

        if attacked:
            m["adv_all"].update(la.argmax(1), label)
            m["adv_rem"].update(la.argmax(1), label, exclude=fp)
            ar = single_image_miou(la, label, a.num_classes, exclude=fp)
            row.update(adv_remote=ar, drop_remote=cr - ar,
                       **attack_rates(lc, la, label, fp, a.target_class))

        if defended:
            m["dclean_all"].update(dca.argmax(1), label)
            m["dclean_rem"].update(dca.argmax(1), label, exclude=fp)
            dcr = single_image_miou(dca, label, a.num_classes, exclude=fp)
            row.update(def_clean_remote=dcr, clean_cost=dcr - cr,
                       clean_fp_area_frac=det_clean["fp_area_frac"],
                       clean_raw_fp_area_frac=det_clean["raw_fp_area_frac"])
            if attacked:
                m["dadv_all"].update(daa.argmax(1), label)
                m["dadv_rem"].update(daa.argmax(1), label, exclude=fp)
                dar = single_image_miou(daa, label, a.num_classes, exclude=fp)
                # RECOVERY runs from the undefended attack (0.0) to the
                # UNDEFENDED clean score (1.0). Its denominator is the attack's
                # own effect, which is near zero on an image the attack failed
                # on — a ratio there is noise amplified, so it is NaN rather
                # than a large number that would dominate a mean.
                gap = cr - ar
                row.update(def_adv_remote=dar, def_drop_remote=dcr - dar,
                           recovery=((dar - ar) / gap if gap > 1.0 else NAN),
                           **{f"det_{k}": v for k, v in det.items()},
                           **{f"detraw_{k}": v for k, v in det_raw.items()})

        per_image.append(row)
        if attacked and defended:
            print(f"  img {i:4d}: drop_remote {cr-ar:+6.2f} -> {dcr-dar:+6.2f}"
                  f"   coverage {det['coverage']:.2f}"
                  f"   clean cost {dcr-cr:+5.2f}"
                  f"   clean FP area {100*det_clean['fp_area_frac']:5.2f}%")
        elif attacked:
            print(f"  img {i:4d}: drop_remote {cr-ar:+6.2f}")
        else:
            print(f"  img {i:4d}: clean {cr:6.2f} -> "
                  f"{row.get('def_clean_remote', cr):6.2f}"
                  f"   erased {100*row.get('clean_fp_area_frac', 0):5.2f}%")

        if a.figures and defended:
            from torchvision.utils import save_image
            d = out_dir / "figures"
            d.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                src = patched if attacked else img
                pur = model.defence(model.to_pixels(src))
            save_image(pur.mask, d / f"img{i:04d}_mask.png")
            save_image(pur.x01, d / f"img{i:04d}_purified.png")

    report(a, m, per_image, idxs, out_dir, attacked, defended,
           {"adv": aggregate_detection(det_adv_rows),
            "adv_raw": aggregate_detection(det_raw_rows),
            "clean": aggregate_detection(det_clean_rows)}, pcfg, n_ch, n_act)


def _read_empty_mask(da) -> str:
    r"""
    An empty mask has three causes with different conclusions. Say which one
    this is, rather than leaving the reader a table of zeros.

    The deciding quantity is the IN-OUT GAP, not the raw score: a detector
    scoring 0.4 inside the footprint and 0.4 everywhere else has found nothing,
    it is uniformly uncertain, and no threshold can create signal that is not
    there.
    """
    if da["in_max"] < 0.1:
        return f"""    READ THIS AS: OUT OF DOMAIN. The segmenter scored the
    patch {da['in_max']:.4f} at best, so no threshold, no shape prior and no
    operating point recovers it.

    SAC is MODEL-independent -- the U-Net never sees the segmentor -- but it is
    NOT dataset- or SCALE-independent. It is a LEARNED detector, self
    adversarially trained on COCO patches at detection resolution, and its
    filters are fixed in absolute pixels.

    TRY, IN ORDER:
      1. --img_h 512 --img_w 1024, the operating point the rest of the results
         use. The patch is then 128px rather than 256px and the frame is close
         to COCO scale. This run was at 1024x2048, the Nesti benchmark
         resolution, which is the worst case for a learned detector.
      2. --sac_image_scale to put the patch near the ~100px it was trained on.
      3. If the in-out gap stays flat under both, only retraining its self
         adversarial loop on our patches will move it -- and "off-the-shelf SAC
         does not transfer to Cityscapes" is then a RESULT, not a bug. It just
         has to be reported with 1 and 2 ruled out, or a reviewer will read it
         as a configuration error."""
    if da["in_out_gap"] > 0.05:
        return f"""    READ THIS AS: IN DOMAIN, WRONG OPERATING POINT. The patch
    scores {da['in_max']:.4f} inside against {da['out_mean']:.4f} outside, so
    there IS signal and it is losing to the 0.5 cut. Sweep --sac_image_scale
    first. Moving the threshold is a DEVIATION from the published method and has
    to be reported as one."""
    return f"""    READ THIS AS: UNIFORMLY UNCERTAIN, NO SIGNAL. The in-out gap
    is only {da['in_out_gap']:+.4f}, so the detector is not distinguishing the
    patch from the scene at all. A threshold cannot create signal that is not
    there, and neither can a scale change."""


def report(a, m, per_image, idxs, out_dir, attacked, defended, detection,
           pcfg, n_ch, n_act):
    agg = {k: v.compute() for k, v in m.items()}
    if attacked:
        agg.update(compare(m["clean_all"], m["adv_all"], prefix="all_"))
        agg.update(compare(m["clean_rem"], m["adv_rem"], prefix="rem_"))
        agg["drop_remote_dataset"] = agg["rem_drop"]
    if defended:
        agg["clean_cost_dataset"] = agg["dclean_rem"] - agg["clean_rem"]
        if attacked:
            agg.update(compare(m["dclean_all"], m["dadv_all"], prefix="dall_"))
            agg.update(compare(m["dclean_rem"], m["dadv_rem"], prefix="drem_"))
            agg["def_drop_remote_dataset"] = agg["drem_drop"]
            gap = agg["clean_rem"] - agg["adv_rem"]
            agg["recovery_dataset"] = ((agg["dadv_rem"] - agg["adv_rem"]) / gap
                                       if gap > 1.0 else NAN)

    def col(k):
        return torch.tensor([float(r.get(k, NAN)) for r in per_image])

    print(f"\n{'='*70}")
    print(f"  {a.arch} @ {a.img_h}x{a.img_w}, {len(idxs)} images, "
          f"defence={a.defence}")
    print(f"{'='*70}")
    print(f"  clean  remote mIoU (undefended)  {agg['clean_rem']:6.2f} dataset")
    if attacked:
        d = col("drop_remote")
        print(f"  drop_remote  per-image mean {d.mean():+.2f} +/- {d.std():.2f}"
              f"   DATASET {agg['drop_remote_dataset']:+.2f}")

    if defended:
        cc, fpa = col("clean_cost"), col("clean_fp_area_frac")
        print(f"\n  --- CLEAN-IMAGE COST (the price of the defence) ---")
        print(f"    clean mIoU {agg['clean_rem']:6.2f} -> "
              f"{agg['dclean_rem']:6.2f}   "
              f"cost {agg['clean_cost_dataset']:+.2f} dataset, "
              f"{cc.mean():+.2f} +/- {cc.std():.2f} per image")
        n_fire = int((fpa > 0).sum())
        print(f"    fired on {n_fire} of {len(per_image)} clean images; "
              f"frame erased mean {100*fpa.mean():.3f}%  "
              f"max {100*fpa.max():.2f}%")
        if n_fire:
            nz = fpa[fpa > 0]
            print(f"    on the {n_fire} it fired on: mean {100*nz.mean():.2f}% "
                  f" min {100*nz.min():.3f}%  max {100*nz.max():.2f}%")
        # THE DISTRIBUTION IS BIMODAL — see sac.defence.completion_pass. The
        # pair (how often it fired, how much it took when it did) is the
        # summary; a mean over all images is a near-zero number averaged over
        # mostly-zero and says nothing.
        print(f"    ^ bimodal by construction. Read the pair, not the mean.")

        if attacked:
            dd, rec = col("def_drop_remote"), col("recovery")
            da, dr = detection["adv"], detection["adv_raw"]
            print(f"\n  --- ATTACK, DEFENDED ---")
            print(f"    drop_remote {col('drop_remote').mean():+.2f} -> "
                  f"{dd.mean():+.2f} per image   DATASET "
                  f"{agg['drop_remote_dataset']:+.2f} -> "
                  f"{agg['def_drop_remote_dataset']:+.2f}")
            print(f"    recovery {agg['recovery_dataset']:+.3f} dataset, "
                  f"{rec.nanmean():+.3f} per image (n="
                  f"{int((~rec.isnan()).sum())} of {len(per_image)} with an "
                  f"effect to undo)")
            print(f"\n  --- LOCALISATION vs the TRUE footprint ---")
            print(f"    completed mask: coverage {da['coverage']:.3f}   "
                  f"precision {da['precision']:.3f}   IoU {da['mask_iou']:.3f}"
                  f"   located {100*da['detected']:.0f}% of images")
            # SEGMENT vs COMPLETE, separated. Raw coverage near zero with high
            # completed coverage means the square prior did the work and the
            # segmenter only had to fire somewhere plausible. Both low means the
            # segmenter never saw the patch and no prior can rescue it.
            print(f"    raw mask (segmenter only): coverage "
                  f"{dr['coverage']:.3f}   area "
                  f"{100*dr['pred_area_frac']:.2f}% of frame")
            if "in_max" in da:
                print()
                print("  --- DETECTOR SCORE, pre-threshold (the 0.5 cut) ---")
                print(f"    inside  the footprint: max {da['in_max']:.4f}   "
                      f"mean {da['in_mean']:.4f}")
                print(f"    outside the footprint: max {da['out_max']:.4f}   "
                      f"mean {da['out_mean']:.4f}")
                print(f"    in-out gap {da['in_out_gap']:+.4f}")
                if da["pred_area_frac"] == 0.0:
                    print()
                    print(_read_empty_mask(da))

    ciou = m["clean_rem"].per_class()
    key = "dadv_rem" if (attacked and defended) else (
        "adv_rem" if attacked else "dclean_rem" if defended else "clean_rem")
    oiou = m[key].per_class()
    print(f"\n  per-class IoU (remote), clean -> {key}:")
    for c in range(min(a.num_classes, 19)):
        if not torch.isnan(ciou[c]):
            print(f"    {c:2d} {class_name(c):10s}: {ciou[c]:6.2f} -> "
                  f"{oiou[c]:6.2f}  ({oiou[c]-ciou[c]:+.1f})")
    print(f"{'='*70}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"checkpoint": a.checkpoint, "config": vars(a),
                   "patch_config": pcfg, "aggregate": agg,
                   "mode": ("attacked" if attacked else "clean_cost"),
                   "defence": {"name": a.defence, "tag": defence_tag(a),
                               "detection": detection},
                   "backbone_channels": n_ch, "backbone_active": n_act,
                   "per_class_iou": {
                       class_name(c): {"clean": float(ciou[c]),
                                       key: float(oiou[c])}
                       for c in range(min(a.num_classes, 19))
                       if not torch.isnan(ciou[c])},
                   "per_image": per_image}, f, indent=2)
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
