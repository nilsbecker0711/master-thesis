#!/usr/bin/env python
r"""
Do the patches transfer to SAM 3? Prompt-conditioned cross-paradigm transfer.

    python scripts/sam3_transfer.py --checkpoint results/.../best.pt \
        --cityscapes_root $CS --img_h 1024 --img_w 2048 --from_image

RUNS IN THE SAM 3 ENV, NOT THE MMSEG ENV. Nothing here imports mmseg, mmcv or
patchreach.models — only patchreach.patch (torch-only), patchreach.data and
patchreach.metrics. Keep the two environments apart: mmcv 1.x and a recent
transformers/torch stack do not coexist.

WHAT IS AND IS NOT MEASURED HERE. SAM 3 has no Cityscapes-trained classifier.
Its "semantic segmentation" is 19 promptable-concept queries merged into one
label map by a rule THIS SCRIPT DEFINES (see merge_prompts). That rule is a
per-pixel argmax over prompt scores, so it is part of the attack surface: a
patch that nudges scores slightly can flip the argmax a lot or not at all
depending on how the merge is written. The rule is therefore a recorded
parameter, not an implementation detail, and `results.json` carries it.

THREE THINGS MAKE A NULL RESULT READABLE, and all three are on by default:

  * the FLOOR (--floor grey|random). A grey square of identical footprint.
    If SAM 3 moves as much under the floor as under the patch, there is no
    transfer, there is occlusion sensitivity. Without this row the table
    cannot be interpreted at all.
  * the HEADROOM AUDIT. Per-class clean IoU is dumped before any attack
    number is quoted. A class SAM 3 never predicts has ~0 clean IoU, so its
    drop is structurally 0 and "no effect" is indistinguishable from "nothing
    left to break". Only classes with headroom enter the claim.
  * the GRADED READOUT. mIoU here is a thresholded argmax and will be
    insensitive. Per-prompt score means inside and outside the footprint are
    logged clean-vs-patched, so a perturbation that does not cross the argmax
    boundary is still visible, and presence suppression (a prompt returning
    no instances at all) is counted separately from a score shift.

THE PATCH IS BUILT IN PIXEL SPACE AND HANDED OVER AS A PIL IMAGE. Patch.apply
composites in NORMALISED space, so the image is normalised with the SAME
IMG_MEAN/IMG_STD the patch was trained under, patched, then denormalised and
clamped. The pixel patch SAM 3 sees is therefore bit-identical to the one the
mmseg eval sees — which is the only reason the transfer number means anything.
Do not "simplify" this by normalising with SAM 3's own statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.data.cityscapes import (CityscapesSeg, class_name,
                                        denormalise, norm_tensors)
from patchreach.metrics.miou import SegMetric, compare
from patchreach.patch.spec import Patch
from patchreach.utils import get_device, increment_path, seed_everything

# The ten fixed val images every number in the thesis is reported on. Imported
# rather than copied: two lists that must agree and are maintained separately
# is the drift this repo keeps paying for.
from _common import FIXED10, image_indices

# ── prompts ──────────────────────────────────────────────────────────────────
# NOT patchreach.data.cityscapes.NAMES. Those are display abbreviations ("veg",
# "moto", "light") and make poor noun phrases for a text-conditioned model.
#
# THE PROMPT WORDING IS A FREE PARAMETER AND IT IS PART OF THE RESULT. Two of
# these are outright interpretations of the Cityscapes definition — class 8
# 'vegetation' is trees and bushes, class 9 'terrain' is grass and soil — and
# class 12 'rider' is a person-in-a-role that no single noun names. Fix this
# table once, record it in results.json (it is written there automatically),
# and if you change it, rerun the clean baseline too: a prompt change moves the
# denominator of every drop below.
PROMPTS = {
    0:  "road",
    1:  "sidewalk",
    2:  "building",
    3:  "wall",
    4:  "fence",
    5:  "pole",
    6:  "traffic light",
    7:  "traffic sign",
    8:  "tree",                                  # Cityscapes 'vegetation'
    9:  "grass",                                 # Cityscapes 'terrain'
    10: "sky",
    11: "person",
    12: "person riding a bicycle or motorcycle",  # Cityscapes 'rider'
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

K_GT = 19          # Cityscapes trainIds 0..18
UNASSIGNED = 19    # pixels no prompt claimed

# SegMetric is built with K_GT + 1 so UNASSIGNED has somewhere to land.
# classes='gt' keeps only rows with gt > 0, and class 19 never appears in the
# ground truth, so it is NaN and drops out of the mean — while a GT pixel
# predicted UNASSIGNED still loses its true class a true positive. The penalty
# is counted, the phantom class is not averaged in. No special-casing needed.
K_METRIC = K_GT + 1


# ── SAM 3 ────────────────────────────────────────────────────────────────────

BPE_ASSET = "assets/bpe_simple_vocab_16e6.txt.gz"


def find_bpe(explicit: str | None) -> str | None:
    r"""
    Locate the BPE vocab the text encoder needs, WITHOUT pkg_resources.

    WHY THIS EXISTS. build_sam3_image_model() falls back to
    pkg_resources.resource_filename("sam3", <asset>) when bpe_path is None, and
    that call dies with

        TypeError: expected str, bytes or os.PathLike object, not NoneType

    whenever sys.modules['sam3'] is a NAMESPACE package — namespace packages
    have __file__ = None, and resource_filename takes os.path.dirname of it.
    A clone sitting at <repo>/sam3/ (the repo root, no __init__.py) next to the
    real package at <repo>/sam3/sam3/ can resolve exactly that way, and this
    script putting the repo root on sys.path makes it more likely rather than
    less. Passing bpe_path explicitly skips the fallback and the whole problem.

    importlib.resources is NOT used as the fallback here on purpose: for a
    namespace package files() returns a MultiplexedPath, which does not
    str()-ify to a filesystem path, so it would trade one TypeError for
    another. A plain filesystem probe is the thing that actually works.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"\n--sam3_bpe {explicit} is not a file.\n")
        return str(p)

    root = Path(__file__).resolve().parents[1]
    for cand in (root / "sam3" / "sam3" / BPE_ASSET,     # clone at <repo>/sam3
                 root / "sam3" / BPE_ASSET,              # package at <repo>/sam3
                 Path("sam3") / "sam3" / BPE_ASSET):     # cwd-relative clone
        if cand.is_file():
            print(f"[sam3] bpe vocab {cand}")
            return str(cand)

    # Let the upstream fallback try. It works when 'sam3' is a regular package.
    print("[sam3] bpe vocab not found on disk; falling back to "
          "pkg_resources (fails if 'sam3' resolves as a namespace package — "
          "pass --sam3_bpe <path to bpe_simple_vocab_16e6.txt.gz>)")
    return None


def load_sam3(ckpt: str | None, version: str, resolution: int, conf: float,
              hw, bpe: str | None = None):
    """
    Build the image model + processor. Imported late so --help works in an env
    without SAM 3 installed.

    CHECKPOINT: build_sam3_image_model() defaults to load_from_HF=True with
    checkpoint_path=None, which calls hf_hub_download twice. A SLURM COMPUTE
    NODE USUALLY HAS NO OUTBOUND INTERNET, so that is a connection error unless
    the files are already in the HF cache. Pass --sam3_ckpt with a path
    downloaded on the login node, or pre-warm the cache there and export
    HF_HUB_OFFLINE=1.

    RESOLUTION is the one number that decides what SAM 3 actually sees.
    Sam3Processor resizes the frame to `resolution` (default 1008) internally,
    and _forward_grounding interpolates the masks back to the ORIGINAL size
    before returning them — so the returned mask shape always matches the input
    and cannot reveal the rescaling. It is reported here instead, because a
    1024x2048 frame at resolution 1008 means the patch reaches the model
    smaller than it was trained, and that is a protocol deviation rather than
    a detail.
    """
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor

    if ckpt is None:
        print(f"[sam3] no --sam3_ckpt; downloading {version} from HF "
              f"(fails on a node without internet unless the cache is warm)")
        ckpt = download_ckpt_from_hf(version=version)
    print(f"[sam3] checkpoint {ckpt}")

    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False,
                                   bpe_path=find_bpe(bpe))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor(model, resolution=resolution, device=device,
                         confidence_threshold=conf)

    # The frame is squeezed into `resolution` on the way in. Say by how much,
    # per axis: the squeeze is ANISOTROPIC on a 1:2 Cityscapes frame, so a
    # square patch does not stay square inside the model.
    sy, sx = resolution / hw[0], resolution / hw[1]
    print(f"[sam3] internal resolution {resolution} vs frame {hw[0]}x{hw[1]} "
          f"-> scale y={sy:.3g} x={sx:.3g}"
          + ("  !! ANISOTROPIC: the patch is not square inside the model"
             if abs(sy - sx) > 1e-6 else ""))
    print(f"[sam3] confidence_threshold {conf} (filters instances INSIDE the "
          f"processor; --score_thresh is the separate merge threshold)")
    return proc


def _as_prob(masks: torch.Tensor, hw) -> torch.Tensor:
    """[N,H,W] in [0,1] at the image resolution.

    SAM 3 may hand back booleans, probabilities or raw logits, at its own
    working resolution. Normalising that here rather than at the call site
    keeps one place to fix when the API shifts under us.
    """
    if masks.ndim == 4:                       # [N,1,h,w]
        masks = masks.squeeze(1)
    m = masks.float()
    if m.dtype == torch.bool or ((m.min() >= 0.0) and (m.max() <= 1.0)):
        pass                                  # already bool-as-float or prob
    else:
        m = torch.sigmoid(m)                  # logits
    if m.shape[-2:] != tuple(hw):
        m = F.interpolate(m.unsqueeze(1), size=tuple(hw), mode="bilinear",
                          align_corners=False).squeeze(1)
    return m.clamp(0.0, 1.0)


@torch.no_grad()
def prompt_all(processor, pil: Image.Image, hw, prompts=PROMPTS):
    """
    One image, 19 prompts -> per-class score map [K_GT,H,W] + presence flags.

    set_image ONCE and prompt the same state repeatedly: the image embedding is
    the expensive half and re-encoding per prompt would be 19x the cost. If a
    future SAM 3 release makes prompting stateful (one prompt leaking into the
    next) this is the line to revisit — check it by prompting one image twice in
    opposite orders and diffing the score maps.

    BF16 AUTOCAST IS MANDATORY, not an optimisation. SAM 3's weights load as
    fp32 but parts of the ViT trunk run in bf16, so without autocast the
    forward dies partway through an MLP block with

        RuntimeError: mat1 and mat2 must have the same dtype,
                      but got BFloat16 and Float

    Autocast casts input AND weight together, which is what makes the mixed
    state consistent. Every notebook under sam3/examples/ opens with
    `torch.autocast("cuda", dtype=torch.bfloat16).__enter__()`; the README's
    image snippet omits it, and that omission is this traceback.

    BOTH calls must be inside it. set_image stores bf16 activations in `state`
    and set_text_prompt consumes them, so autocasting only the first would move
    the same error into the decoder.
    """
    # enabled=False on CPU, so the fp32 path still works for local testing.
    with torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=torch.cuda.is_available()):
        return _prompt_all_inner(processor, pil, hw, prompts)


def _prompt_all_inner(processor, pil, hw, prompts):
    state = processor.set_image(pil)
    score = torch.zeros(K_GT, *hw)
    present, n_inst = {}, {}

    for c, text in prompts.items():
        out = processor.set_text_prompt(state=state, prompt=text)
        masks, scores = out["masks"], out["scores"]
        if masks is None or len(masks) == 0:
            present[c], n_inst[c] = False, 0
            continue
        m = _as_prob(torch.as_tensor(masks), hw)                   # [N,H,W]
        s = torch.as_tensor(scores).float().view(-1, 1, 1)        # [N,1,1]
        # max over instances: a pixel's evidence for class c is the best
        # instance claiming it, weighted by that instance's confidence.
        score[c] = (m * s).amax(0)
        present[c], n_inst[c] = True, int(len(m))

    return score, present, n_inst


def merge_prompts(score: torch.Tensor, thresh: float):
    """
    THE MERGE RULE. Per-pixel argmax over prompt scores; pixels whose best
    score is below `thresh` go to UNASSIGNED rather than being forced into the
    least-bad class.

    Forcing them would hide exactly the failure mode worth measuring — a patch
    that suppresses every prompt in a region produces no winner, and that is
    a different event from a patch that swaps the winner. UNASSIGNED keeps the
    two distinguishable in the confusion matrix.
    """
    best, idx = score.max(0)
    return torch.where(best >= thresh, idx,
                       torch.full_like(idx, UNASSIGNED))


# ── patch application ────────────────────────────────────────────────────────

def build_patched(patch, img, mean_t, std_t, H, W, from_image: bool,
                  floor: str = "none"):
    """
    (patched_normalised [1,3,H,W], footprint [1,H,W]).

    ORDER IS LOAD-BEARING and matches optimise.prepare() and evaluate.py:
    resolve_placement THEN set_reference_from_image. The reference is the image
    region at the RESOLVED placement, so reversing these copies the wrong
    window and a csf patch becomes a residual on top of the wrong base.
    """
    patch.resolve_placement(H, W)            # clean_pred=None; guarded by caller
    if from_image:
        patch.set_reference_from_image(img, mean_t, std_t)
    patched, fp = patch.apply(img)

    if floor != "none":
        # THE CONTROL. Identical footprint, no optimised content. Built from the
        # patch's own footprint so a cutout shape is honoured rather than
        # replaced by a square — the floor has to match the patch in area AND
        # silhouette or it is not a floor for this patch.
        m = fp[0].unsqueeze(0).unsqueeze(0).float()               # [1,1,H,W]
        if floor == "grey":
            fill01 = torch.full_like(img, 0.5)
        elif floor == "random":
            fill01 = torch.rand_like(img)
        else:
            raise ValueError(f"unknown --floor {floor!r}")
        fill = (fill01 - mean_t) / std_t
        patched = img * (1.0 - m) + fill * m
    return patched, fp


def to_pil(normalised: torch.Tensor, mean_t, std_t) -> Image.Image:
    """Normalised [1,3,H,W] -> 8-bit PIL.

    denormalise() drops the batch dim, clamps to [0,1] and moves to CPU, so it
    returns [3,H,W] and needs no indexing here. Its clamp is where a residual
    that ran off the representable range gets truncated: _apply_residual already
    clamps in [0,1] for universal_csf, so for that mode it is a no-op, and for
    the others it is the only clamp there is.

    The round-to-uint8 is a REAL quantisation of the patch — 1/255 is coarse
    next to a CSF budget at low tau — and it is unavoidable, because SAM 3 is
    being handed an image file. The mmseg eval does not quantise, so a tiny
    clean-vs-patched difference here is expected and is a property of the
    handover, not of the model. Quantify it once with --save_png if a low-tau
    row looks suspiciously flat.
    """
    x = denormalise(normalised, mean_t, std_t)
    arr = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="patch .pt")
    p.add_argument("--cityscapes_root", required=True)
    p.add_argument("--images", default="fixed10",
                   help="'fixed10' | 'all' | '430 46 441'")
    p.add_argument("--img_h", type=int, default=1024,
                   help="1024x2048 is the NATIVE frame under --scale resize "
                        "(the resize is then a no-op).")
    p.add_argument("--img_w", type=int, default=2048)
    p.add_argument("--scale", choices=["resize", "crop"], default="resize")
    p.add_argument("--from_image", action="store_true",
                   help="rebuild the csf base from the evaluated image. "
                        "REQUIRED for mode='csf' — refused without it below.")
    p.add_argument("--placement", default=None, metavar="TOP,LEFT",
                   help="override the resolved placement, for checkpoints "
                        "whose policy needs a model prediction. Export it from "
                        "the mmseg side and paste it here.")
    p.add_argument("--floor", choices=["none", "grey", "random"], default="none",
                   help="replace the patch with a grey/random fill of the "
                        "IDENTICAL footprint. Run this row. See the module "
                        "docstring for why the table is unreadable without it.")
    p.add_argument("--score_thresh", type=float, default=0.5,
                   help="below this, a pixel is UNASSIGNED rather than forced "
                        "into its least-bad class. Part of the merge rule — "
                        "ablate it once and record the value.")
    p.add_argument("--prompts_json", default=None,
                   help="override the prompt table: {\"0\": \"road\", ...}")
    p.add_argument("--save_png", action="store_true",
                   help="dump clean/patched PNGs for inspection")
    p.add_argument("--sam3_ckpt", default=None,
                   help="LOCAL path to sam3.pt / sam3.1_multiplex.pt. Without "
                        "it the build downloads from HF, which fails on a "
                        "compute node with no internet. Download on the login "
                        "node first.")
    p.add_argument("--sam3_version", choices=["sam3", "sam3.1"],
                   default="sam3.1",
                   help="only used when --sam3_ckpt is absent")
    p.add_argument("--sam3_resolution", type=int, default=1008,
                   help="Sam3Processor's internal working resolution. The "
                        "frame is squeezed into it, so it decides how large "
                        "the patch actually arrives.")
    p.add_argument("--sam3_bpe", default=None,
                   help="path to bpe_simple_vocab_16e6.txt.gz. Auto-detected "
                        "under ./sam3/ when absent. Needed because the "
                        "upstream pkg_resources fallback dies when 'sam3' "
                        "resolves as a namespace package.")
    p.add_argument("--sam3_conf", type=float, default=0.5,
                   help="Sam3Processor.confidence_threshold — drops instances "
                        "INSIDE the processor, before the merge. Distinct from "
                        "--score_thresh and recorded separately.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", default="results/sam3")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    seed_everything(a.seed)
    device = get_device()
    mean_t, std_t = norm_tensors(device)
    H, W = a.img_h, a.img_w

    prompts = dict(PROMPTS)
    if a.prompts_json:
        prompts = {int(k): v for k, v in
                   json.loads(Path(a.prompts_json).read_text()).items()}
    if sorted(prompts) != list(range(K_GT)):
        raise SystemExit(f"prompts must cover trainIds 0..{K_GT - 1}, "
                         f"got {sorted(prompts)}")

    # ── patch, with the two refusals evaluate.py makes ───────────────────────
    ck = torch.load(a.checkpoint, map_location="cpu")
    pcfg = ck["config"]
    mode = pcfg.get("mode")

    if mode in ("gan", "raw_ganinit"):
        raise SystemExit(
            f"\nmode={mode!r} needs the BigGAN generator to render, which is "
            f"not installable next to SAM 3.\nRender the patched frames in the "
            f"mmseg env, write them to PNG, and point this script's SAM 3 half "
            f"at those files instead.\n")

    if mode == "csf" and not a.from_image:
        raise SystemExit(
            "\nThis checkpoint is mode='csf'. Its base is the image region the\n"
            "patch covers and that base is NOT in the checkpoint. Re-run with\n"
            "--from_image. Without it the base is flat grey, the patch becomes a\n"
            "visible square with an invisible texture, and you would be\n"
            "measuring whether SAM 3 notices a grey square.\n")

    patch = Patch.load(a.checkpoint, device, mean_t, std_t)

    # resolve() SILENTLY CENTRES for placement='semantic' when clean_pred is
    # None (placement.py:186). There is no segmentation model in this env to
    # supply one, so refuse rather than report a semantic run that was centred.
    policy = pcfg.get("placement", "center")
    if a.placement:
        top, left = (int(v) for v in a.placement.replace(",", " ").split())
        patch.placement = (top, left)
        needs_model = False
    else:
        needs_model = policy in ("semantic", "gradcam")
    if needs_model:
        raise SystemExit(
            f"\nplacement={policy!r} is resolved from the SOURCE MODEL's clean\n"
            f"prediction (semantic) or its sensitivity map (gradcam). Neither\n"
            f"exists here, and resolve() would centre silently for 'semantic'.\n\n"
            f"Export the resolved (top,left) from the mmseg side at THIS\n"
            f"resolution ({H}x{W} — placement is resolution-dependent, so the\n"
            f"value stored in the checkpoint is wrong at any other size) and\n"
            f"pass --placement TOP,LEFT.\n")

    print(f"\n[patch] {a.checkpoint}")
    patch.describe(H, W)

    # RESAMPLING WARNING. The parameter grid is cfg.size; the pasted footprint
    # is side(H,W). When they differ, apply() interpolates — and resampling a
    # spectrally-budgeted residual RESAMPLES ITS SPECTRUM, so tau stops
    # describing the tensor SAM 3 is shown. universal_csf refuses outright
    # (_apply_residual). csf does not refuse, which is exactly why this warns:
    # at native 1024x2048 a patch trained at 512x1024 is silently 2x upsampled
    # and every visibility claim attached to it is void.
    side = patch.side(H, W)
    if mode in ("csf", "universal_csf") and side != pcfg.get("size"):
        print(f"\n  !! parameter grid {pcfg.get('size')} != pasted footprint "
              f"{side} at {H}x{W}.\n"
              f"  !! The residual will be resampled and tau NO LONGER "
              f"DESCRIBES IT.\n"
              f"  !! Run at the resolution the patch was trained at, or treat "
              f"the resample\n"
              f"  !! as part of the experiment and say so. "
              f"(universal_csf refuses instead of warning.)\n")
    print(f"[merge] argmax over {K_GT} prompts, thresh {a.score_thresh}, "
          f"below -> UNASSIGNED")
    print(f"[floor] {a.floor}")

    ds = CityscapesSeg(a.cityscapes_root, "val", H, W, scale=a.scale,
                       random_crop=False)
    idxs = image_indices(a.images, len(ds))

    tag = "_".join(x for x in [Path(a.checkpoint).parent.name,
                               f"sam3_{H}x{W}",
                               a.floor if a.floor != "none" else "",
                               a.tag] if x)
    out_dir = Path(increment_path(Path(a.out_root) / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = load_sam3(a.sam3_ckpt, a.sam3_version,
                          a.sam3_resolution, a.sam3_conf, (H, W),
                          a.sam3_bpe)

    m = {k: SegMetric(K_METRIC, device="cpu")
         for k in ("clean_all", "clean_rem", "adv_all", "adv_rem")}
    per_image, score_rows = [], []

    for i in idxs:
        img, label = ds[i]
        img = img.unsqueeze(0).to(device)
        label = label.unsqueeze(0)                                # [1,H,W] cpu
        hw = tuple(label.shape[-2:])

        patched, fp = build_patched(patch, img, mean_t, std_t, H, W,
                                    a.from_image, a.floor)
        pil_clean, pil_adv = (to_pil(img, mean_t, std_t),
                              to_pil(patched, mean_t, std_t))
        if a.save_png:
            d = out_dir / "images"
            d.mkdir(exist_ok=True)
            pil_clean.save(d / f"img{i:04d}_clean.png")
            pil_adv.save(d / f"img{i:04d}_patched.png")

        sc_c, pres_c, nin_c = prompt_all(processor, pil_clean, hw, prompts)
        sc_a, pres_a, nin_a = prompt_all(processor, pil_adv, hw, prompts)
        pc = merge_prompts(sc_c, a.score_thresh).unsqueeze(0)
        pa = merge_prompts(sc_a, a.score_thresh).unsqueeze(0)

        fp_cpu = fp.cpu()
        m["clean_all"].update(pc, label)
        m["clean_rem"].update(pc, label, exclude=fp_cpu)
        m["adv_all"].update(pa, label)
        m["adv_rem"].update(pa, label, exclude=fp_cpu)

        # Per-image remote mIoU, on its own confusion matrix so it is not
        # contaminated by the dataset pool.
        one = {k: SegMetric(K_METRIC) for k in ("c", "a")}
        one["c"].update(pc, label, exclude=fp_cpu)
        one["a"].update(pa, label, exclude=fp_cpu)
        cr, ar = one["c"].compute(), one["a"].compute()

        # THE GRADED READOUT. mIoU is an argmax and can sit still while the
        # scores underneath it move; these cannot. 'remote' is outside the
        # footprint, which is the only place transfer is a claim about reach
        # rather than about occlusion.
        rem = ~fp_cpu[0]
        for c in range(K_GT):
            score_rows.append({
                "image": i, "class": class_name(c), "prompt": prompts[c],
                "clean_score_remote": float(sc_c[c][rem].mean()),
                "adv_score_remote": float(sc_a[c][rem].mean()),
                "clean_n_inst": nin_c[c], "adv_n_inst": nin_a[c],
                # presence SUPPRESSION: the prompt fired on the clean frame and
                # returned nothing on the patched one. A different event from a
                # score that merely slid, and invisible in mIoU.
                "presence_lost": bool(pres_c[c] and not pres_a[c]),
                "presence_gained": bool(pres_a[c] and not pres_c[c]),
            })

        lost = sum(r["presence_lost"] for r in score_rows[-K_GT:])
        row = {"image": i, "clean_remote": cr, "adv_remote": ar,
               "drop_remote": cr - ar,
               "rel_drop_remote": (cr - ar) / cr * 100.0 if cr > 0 else float("nan"),
               "presence_lost": lost}
        per_image.append(row)
        print(f"  img {i:4d}: clean {cr:6.2f} -> adv {ar:6.2f}  "
              f"drop {cr - ar:+6.2f} ({row['rel_drop_remote']:+5.1f}%)  "
              f"presence_lost {lost}/{K_GT}")

    # ── headroom audit, printed BEFORE the aggregate ─────────────────────────
    ciou, aiou = m["clean_rem"].per_class(), m["adv_rem"].per_class()
    print(f"\n{'=' * 72}")
    print("  HEADROOM AUDIT — read this before quoting any drop.")
    print("  A class SAM 3 barely predicts has no room to fall, so its 0.00")
    print("  drop is NOT evidence that the patch failed on it.")
    print(f"  {'class':<10} {'clean':>7} {'adv':>7} {'drop':>7}   headroom")
    headroom = {}
    for c in range(K_GT):
        if torch.isnan(ciou[c]):
            continue
        ok = ciou[c].item() >= 10.0
        headroom[class_name(c)] = ok
        print(f"  {class_name(c):<10} {ciou[c]:7.2f} {aiou[c]:7.2f} "
              f"{aiou[c] - ciou[c]:+7.2f}   {'yes' if ok else 'NO  (<10 mIoU)'}")

    agg = {k: v.compute() for k, v in m.items()}
    agg.update(compare(m["clean_rem"], m["adv_rem"], prefix="rem_"))
    agg.update(compare(m["clean_all"], m["adv_all"], prefix="all_"))
    drops = torch.tensor([r["drop_remote"] for r in per_image])
    n_lost = sum(r["presence_lost"] for r in per_image)

    # Restricted mIoU: only the classes that had somewhere to fall. This is the
    # number the transfer claim is allowed to rest on.
    keep = [c for c in range(K_GT)
            if not torch.isnan(ciou[c]) and ciou[c].item() >= 10.0]
    c_keep = float(ciou[keep].nanmean()) if keep else float("nan")
    a_keep = float(aiou[keep].nanmean()) if keep else float("nan")

    print(f"\n  {len(idxs)} images @ {H}x{W}   floor={a.floor}")
    print(f"  remote mIoU  clean {agg['clean_rem']:6.2f} -> "
          f"adv {agg['adv_rem']:6.2f}   drop {agg['rem_drop']:+.2f}")
    print(f"  WITH HEADROOM ONLY ({len(keep)}/{K_GT} classes)  "
          f"{c_keep:6.2f} -> {a_keep:6.2f}   drop {a_keep - c_keep:+.2f}")
    # torch.std is unbiased, so n=1 gives nan. Print the count instead of a nan
    # that reads like a failed computation rather than "no spread is defined".
    spread = (f"+/- {drops.std():.2f}" if len(per_image) > 1
              else "(n=1, no spread)")
    print(f"  per-image drop  mean {drops.mean():+.2f} {spread}")
    print(f"  presence lost   {n_lost} prompt-image pairs "
          f"of {len(idxs) * K_GT}")
    print(f"{'=' * 72}")
    if a.floor == "none":
        print("  NO FLOOR ROW IN THIS RUN. Re-run with --floor grey before "
              "reporting:\n  a drop the floor also produces is occlusion, not "
              "transfer.")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"checkpoint": a.checkpoint, "config": vars(a),
                   "patch_config": pcfg,
                   "merge_rule": {"kind": "per_pixel_argmax_over_prompt_scores",
                                  "instance_reduce": "max(mask_prob * score)",
                                  "score_thresh": a.score_thresh,
                                  "sam3_confidence_threshold": a.sam3_conf,
                                  "sam3_resolution": a.sam3_resolution,
                                  "below_thresh": "UNASSIGNED",
                                  "unassigned_index": UNASSIGNED},
                   "prompts": prompts,
                   "aggregate": agg,
                   "headroom": headroom,
                   "with_headroom_only": {"n_classes": len(keep),
                                          "clean": c_keep, "adv": a_keep,
                                          "drop": a_keep - c_keep},
                   "per_class_iou": {
                       class_name(c): {"clean": float(ciou[c]),
                                       "adv": float(aiou[c])}
                       for c in range(K_GT) if not torch.isnan(ciou[c])},
                   "presence_lost_total": n_lost,
                   "per_image": per_image,
                   "per_prompt": score_rows}, f, indent=2)
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    main()
