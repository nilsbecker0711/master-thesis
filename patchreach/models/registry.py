r"""
Architecture registry — one entry per CHECKPOINT, not per family.

WHY PER-CHECKPOINT: mmseg needs a config that matches the exact variant.
segformer_mit-b0_*.py will not build B5, and upernet_internimage_t_*.py will not
build InternImage-L. Registry keys are therefore free-form (`segformer_b0`,
`internimage_t`, ...) and `bracket` records which ERF class the entry belongs to
so analysis can group them.

RESOLUTION IS *NOT* PART OF THE CONFIG CHOICE. The numbers in an mmseg config
name (512x1024, 1024x1024) are the TRAINING CROP. The config only builds the
architecture; input size comes from the dataloader via --img_h/--img_w. One
config per checkpoint covers every resolution in the transfer matrix.

WHY VULNERABILITY TRACKS THE BRACKET — Yuan et al. (ACM MM 2024) Table 1:

    no attention     (UPerNet/ConvNeXt)  tiny ERF   targeted attack mIoU  5.86
    local attention  (SeMask/Swin)       mid  ERF   targeted attack mIoU 29.25
    global attention (Segmenter/ViT)     huge ERF   targeted attack mIoU 74.57

InternImage uses DCNv3 — deformable convolution, no global self-attention — so
it sits in the small-ERF robust bracket. SegFormer's MiT backbone uses efficient
self-attention => global ERF => vulnerable bracket. Switching arch isolates the
GEOMETRIC factor while holding attack, data, patch size and budget constant.

FILL THESE IN, or pass --cfg_path / --weights on every call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BRACKETS = {
    "none": "no global attention — small ERF, robust bracket",
    "local": "windowed attention — mid ERF, middle bracket",
    "global": "global/efficient self-attention — large ERF, vulnerable bracket",
}


@dataclass
class ArchSpec:
    name: str
    bracket: str                      # none | local | global
    cfg: Optional[str] = None
    weights: Optional[str] = None
    note: str = ""

    @property
    def family(self) -> str:
        return BRACKETS.get(self.bracket, self.bracket)


_W = "/pfs/work9/workspace/scratch/ma_nilbecke-thesis"
_M = f"{_W}/checkpoints"
#/pfs/work9/workspace/scratch/ma_nilbecke-thesis/master-thesis

REGISTRY = {

    # ── DCNv3, no global attention ───────────────────────────────────────────
    "internimage_t": ArchSpec(
        "internimage_t", "none",
        cfg=f"{_M}/upernet_internimage_t_512x1024_160k_cityscapes.py",
        weights=f"{_M}/upernet_internimage_t_512x1024_160k_cityscapes.pth",
        note="UPerNet head. Emits a 150-channel ADE20K-dimensioned head; only "
             "0..18 are active. The channel probe reports this at startup."),
    "internimage_l": ArchSpec("internimage_l", "none"),
    "internimage_xl": ArchSpec("internimage_xl", "none"),

    # ── MiT efficient self-attention ─────────────────────────────────────────
    "segformer_b0": ArchSpec(
        "segformer_b0", "global",
        cfg=f"{_M}/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py",
        weights=f"{_M}/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth",
        note="Weakest MiT variant (~76 dataset mIoU vs B5's ~82). Fine for the "
             "ERF probe; consider B2/B5 for a baseline closer to InternImage-T."),
    "segformer_b5": ArchSpec("segformer_b5", "global",
        cfg=f"{_M}/segformer_b5.py",
        weights=f"{_M}/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth",
        note="Same MiT architecture as b0 at ~22x params. Note its stem is "
            "expansive (3->64 at stride 4) where b0's is compressive (3->32)."),

    "deeplabv3plus_r101": ArchSpec("deeplabv3plus_r101", "none",
        cfg=f"{_M}/deeplabv3plus_r101.py",
        weights=f"{_M}/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth",
        note="Dilated ResNet-101, no attention. Entry stride 8, deepest in the set."),

    "unet_s5d16": ArchSpec("unet_s5d16", "none",
        cfg=f"{_M}/unet_s5d16.py",
        weights=f"{_M}/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth",
        note="Entry stride 1 — the only full-resolution path. Clean mIoU 69.10 is "
            "the lowest in the set; normalise drops by clean score."),

    "setr_pup": ArchSpec("setr_pup", "global",
        cfg=f"{_M}/setr_pup.py",
        weights=f"{_M}/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth",
        note="ViT-L, 16x16 non-overlapping patch embed. Entry stride 16, the other "
            "extreme from UNet. Stem is information-preserving (768->1024), so a "
            "failure here is architectural rather than information loss."),
}

    # ── Swin windowed attention ──────────────────────────────────────────────
    # Stock-mmseg alternative for the no-attention bracket — needs no custom
    # ops. Yuan et al. Table 1 used exactly UPerNet/ConvNeXt for this row
    # (targeted attack mIoU 5.86, their most robust model), so it is arguably
    # the more faithful comparison point anyway.
'''
    "convnext_t": ArchSpec(
        "convnext_t", "none",
        note="UPerNet/ConvNeXt. Stock mmseg, no DCNv3. Check the mmseg zoo for "
             "Cityscapes weights; ADE20K weights are sufficient for "
             "measure_erf.py, which is label- and dataset-independent."),

    "swin_t": ArchSpec(
        "swin_t", "local",
        note="Cityscapes weights are not in the stock mmseg zoo. ADE20K weights "
             "are FINE for measure_erf.py — that probe is label- and "
             "dataset-independent — so the three-bracket ERF comparison is "
             "possible even without Cityscapes training."),
'''
#}

# Convenience aliases so short names keep working.
REGISTRY["internimage"] = REGISTRY["internimage_t"]
REGISTRY["segformer"] = REGISTRY["segformer_b0"]
#REGISTRY["swin"] = REGISTRY["swin_t"]
REGISTRY["unet"] = REGISTRY["unet_s5d16"]
REGISTRY["ocrnet"] = REGISTRY["ocrnet_hr48"]
REGISTRY["deeplab"] = REGISTRY["deeplabv3plus_r101"]


def resolve(name: str, cfg_override=None, weights_override=None):
    """(cfg, weights, spec) with CLI overrides applied. Raises if unresolved."""
    if name not in REGISTRY:
        raise KeyError(f"unknown arch {name!r}. Known: "
                       f"{sorted(k for k in REGISTRY)}")
    spec = REGISTRY[name]
    cfg = cfg_override or spec.cfg
    weights = weights_override or spec.weights
    if not cfg or not weights:
        raise SystemExit(
            f"--arch {name} has no registered paths.\n"
            f"  Either fill cfg/weights into patchreach/models/registry.py,\n"
            f"  or pass --cfg_path <config>.py --weights <ckpt>.pth\n"
            f"  ({spec.note})" if spec.note else "")
    return cfg, weights, spec


def load_segmentor(cfg_path: str, weights_path: str):
    """
    Build + load any mmseg EncoderDecoder. Returns (model, backbone_type).

    Custom backbones are registered from what the CONFIG declares, NOT from a
    command-line flag. Gating on a flag lets --arch and --cfg_path drift apart
    and produces an opaque "InternImage is not in the models registry" KeyError
    three frames deep. A file named segformer_*.py can still declare
    type='InternImage' — that has happened in this project.
    """
    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(cfg_path)
    backbone = cfg.model.get("backbone", {}).get("type", "<unknown>")
    head = cfg.model.get("decode_head", {}).get("type", "<unknown>")

    # Custom backbones are registered ONLY when the config asks for them, so a
    # repo without mmseg_custom / DCNv3 runs SegFormer and Swin perfectly well.
    if backbone == "InternImage":
        try:
            import mmseg_custom.models.backbones.intern_image   # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"This config declares backbone type='InternImage', which needs\n"
                f"the mmseg_custom package AND its compiled DCNv3 CUDA extension\n"
                f"(ops_dcnv3). Neither is part of patch-reach.\n\n"
                f"  config : {cfg_path}\n"
                f"  error  : {e}\n\n"
                f"Options:\n"
                f"  * put mmseg_custom on PYTHONPATH and build ops_dcnv3, or\n"
                f"  * use a stock-mmseg backbone instead. SegFormer (global\n"
                f"    attention) and Swin (windowed) need nothing custom, and\n"
                f"    UPerNet/ConvNeXt covers the no-attention bracket — that is\n"
                f"    exactly the model Yuan et al. Table 1 used for it (5.86).\n"
                f"    All three give a valid ERF comparison without DCNv3."
            ) from e

    print(f"[model] backbone={backbone}  head={head}  "
          f"num_classes={cfg.model.decode_head.get('num_classes')}")

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    ck = load_checkpoint(model, weights_path, map_location="cpu")
    for k in ("CLASSES", "PALETTE"):
        if k in ck.get("meta", {}):
            setattr(model, k, ck["meta"][k])
    return model, backbone
