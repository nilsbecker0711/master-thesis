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

TWO AXES, NOT ONE. `bracket` records the ENCODER's attention scope and is what
the ERF story is told over. It cannot express the second question this lineup
now has to answer: every entry below that predates the factorial varies encoder
AND decoder together, so "CNNs resist this attack" is UNIDENTIFIED — DeepLab
could resist because ResNet never builds the reach, or because ASPP discards
it. `encoder` and `decoder` separate them, and analysis groups on whichever
axis the table is about. `bracket` is kept because it is a property of the
encoder alone and is cross-checked against it at import.

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
from pathlib import Path
from typing import Optional

BRACKETS = {
    "none": "no global attention — small ERF, robust bracket",
    "local": "windowed attention — mid ERF, middle bracket",
    "global": "global/efficient self-attention — large ERF, vulnerable bracket",
}

# Encoder -> attention scope. The bracket is a property of the ENCODER, so this
# is the single place the mapping lives and every entry's `bracket` is checked
# against it at import. A pairing that disagrees is a typo, not a variant.
ENCODERS = {
    "unet":        "none",    # plain conv encoder, entry stride 1
    "resnet50":    "none",    # dilated ResNet-50
    "resnet101":   "none",    # dilated ResNet-101
    "wrn50_2":     "none",    # WideResNet-50-2, 2x width of resnet50
    "convnext_t":  "none",    # ConvNeXt-Tiny, large-kernel conv
    "dcnv3_t":     "none",    # InternImage-T, deformable conv v3
    "swin_t":      "local",   # Swin-Tiny, shifted-window attention
    "vit_s":       "global",  # ViT-Small, full self-attention
    "vit_l":       "global",  # ViT-Large
    "mit_b0":      "global",  # SegFormer MixVisionTransformer, efficient attn
    "mit_b5":      "global",
}

# Decoder -> what it does with the encoder's features. The axis that separates
# "the encoder never built the reach" from "the decoder threw it away".
DECODERS = {
    "unet":       "symmetric expansive path with skip connections",
    "aspp":       "DeepLabV3+ atrous spatial pyramid pooling",
    "upernet":    "UPerNet pyramid pooling + FPN",
    "all_mlp":    "SegFormer all-MLP, concatenates all four stages",
    "pup":        "SETR progressive upsampling",
    "mask_tf":    "Segmenter mask-transformer",
    "fcn":        "plain fully-convolutional head",
}


@dataclass
class ArchSpec:
    name: str
    bracket: str                      # none | local | global — see ENCODERS
    cfg: Optional[str] = None
    weights: Optional[str] = None
    note: str = ""
    encoder: str = ""                 # key into ENCODERS
    decoder: str = ""                 # key into DECODERS

    def __post_init__(self):
        """
        Cross-check the bracket against the encoder, and reject unknown keys.

        RAISES AT IMPORT, deliberately: registry.py is imported by _common, so
        a bad entry fails on the first line of every script instead of three
        frames into mmcv after a checkpoint has already loaded. The failure an
        alias once caused here was a KeyError with no message; these say which
        entry and what is wrong with it.

        An empty encoder/decoder is allowed so a pre-factorial entry is not
        forced to declare an axis it was never filed under -- but every entry
        below does declare both, and a new one should too.
        """
        if self.encoder and self.encoder not in ENCODERS:
            raise ValueError(
                f"ArchSpec {self.name!r}: unknown encoder {self.encoder!r}. "
                f"Known: {sorted(ENCODERS)}")
        if self.decoder and self.decoder not in DECODERS:
            raise ValueError(
                f"ArchSpec {self.name!r}: unknown decoder {self.decoder!r}. "
                f"Known: {sorted(DECODERS)}")
        if self.encoder and ENCODERS[self.encoder] != self.bracket:
            raise ValueError(
                f"ArchSpec {self.name!r}: bracket {self.bracket!r} disagrees "
                f"with encoder {self.encoder!r}, whose attention scope is "
                f"{ENCODERS[self.encoder]!r}. One of the two is a typo.")

    @property
    def family(self) -> str:
        return BRACKETS.get(self.bracket, self.bracket)

    @property
    def available(self) -> bool:
        """
        Both declared files are on disk, i.e. this entry can actually be built.

        A FILESYSTEM QUESTION, NOT A REGISTRY ONE, and deliberately so: every
        entry below names the paths its config and checkpoint belong at, so
        "registered" and "present" are different states and only the second
        one can be run. On a workstation this is False for everything, which
        is correct -- nothing runs there either.
        """
        return bool(self.cfg and self.weights
                    and Path(self.cfg).exists() and Path(self.weights).exists())


_W = "/pfs/work9/workspace/scratch/ma_nilbecke-thesis"
_M = f"{_W}/checkpoints"
#/pfs/work9/workspace/scratch/ma_nilbecke-thesis/master-thesis

REGISTRY = {

    # ═════════════════════════════════════════════════════════════════════════
    #  RUNNABLE — cfg and weights registered
    # ═════════════════════════════════════════════════════════════════════════

    # ── DCNv3, no global attention ───────────────────────────────────────────
    "internimage_t": ArchSpec(
        "internimage_t", "none",
        cfg=f"{_M}/upernet_internimage_t_512x1024_160k_cityscapes.py",
        weights=f"{_M}/upernet_internimage_t_512x1024_160k_cityscapes.pth",
        encoder="dcnv3_t", decoder="upernet",
        note="UPerNet head. Emits a 150-channel ADE20K-dimensioned head; only "
             "0..18 are active. The channel probe reports this at startup."),

    # ── MiT efficient self-attention ─────────────────────────────────────────
    "segformer_b0": ArchSpec(
        "segformer_b0", "global",
        cfg=f"{_M}/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py",
        weights=f"{_M}/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth",
        encoder="mit_b0", decoder="all_mlp",
        note="Weakest MiT variant (~76 dataset mIoU vs B5's ~82). Fine for the "
             "ERF probe; consider B2/B5 for a baseline closer to InternImage-T."),

    "segformer_b5": ArchSpec(
        "segformer_b5", "global",
        cfg=f"{_M}/segformer_b5.py",
        weights=f"{_M}/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth",
        encoder="mit_b5", decoder="all_mlp",
        note="Same MiT architecture as b0 at ~22x params. Note its stem is "
             "expansive (3->64 at stride 4) where b0's is compressive (3->32)."),

    "deeplabv3plus_r101": ArchSpec(
        "deeplabv3plus_r101", "none",
        cfg=f"{_M}/deeplabv3plus_r101.py",
        weights=f"{_M}/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth",
        encoder="resnet101", decoder="aspp",
        note="Dilated ResNet-101, no attention. Entry stride 8, deepest in the set."),

    "unet_s5d16": ArchSpec(
        "unet_s5d16", "none",
        cfg=f"{_M}/unet_s5d16.py",
        weights=f"{_M}/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth",
        encoder="unet", decoder="unet",
        note="Entry stride 1 — the only full-resolution path. Clean mIoU 69.10 is "
             "the lowest in the set; normalise drops by clean score."),

    "setr_pup": ArchSpec(
        "setr_pup", "global",
        cfg=f"{_M}/setr_pup.py",
        weights=f"{_M}/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth",
        encoder="vit_l", decoder="pup",
        note="ViT-L, 16x16 non-overlapping patch embed. Entry stride 16, the other "
             "extreme from UNet. Stem is information-preserving (768->1024), so a "
             "failure here is architectural rather than information loss."),

    # ═════════════════════════════════════════════════════════════════════════
    #  THE ENCODER x DECODER FACTORIAL — stubs, paths not yet registered
    # ═════════════════════════════════════════════════════════════════════════
    #
    # WHY THESE EXIST. Every runnable entry above changes encoder and decoder
    # together, so none of them identifies which one causes the CNN robustness
    # the overfit block measured. These cells hold one axis fixed at a time.
    #
    # HOW TO ACTIVATE ONE: drop the config and checkpoint at the paths named in
    # its cfg=/weights= and it becomes runnable with no other change. Until
    # then resolve() refuses it by name and prints those paths, so a stub is
    # addressable and listable but never silently half-loaded.
    #
    # TIERING — read before budgeting any of this. measure_erf.py and
    # measure_frequency_sensitivity.py are label- and dataset-independent, so
    # the FULL grid runs on ADE20K weights with no Cityscapes training at all.
    # Only the attack tables need Cityscapes checkpoints. Let the label-free
    # grid decide which cells are worth training: two encoders with
    # indistinguishable band profiles do not both need an attack run.

    # ── DeepLabV3+ / ASPP decoder, encoder varies ────────────────────────────
    "deeplabv3plus_r50": ArchSpec(
        "deeplabv3plus_r50", "none",
        cfg=f"{_M}/deeplabv3plus_r50.py",
        weights=f"{_M}/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.pth",
        encoder="resnet50", decoder="aspp",
        note="STUB. The one cell in the factorial that is stock: "
             "deeplabv3plus_r50-d8 Cityscapes is in the mmseg zoo. Paired with "
             "deeplabv3plus_r101 it also gives a depth-only contrast at fixed "
             "encoder family and fixed decoder."),

    "deeplabv3plus_wrn50_2": ArchSpec(
        "deeplabv3plus_wrn50_2", "none",
        cfg=f"{_M}/deeplabv3plus_wrn50_2.py",
        weights=f"{_M}/deeplabv3plus_wrn50_2_512x1024_cityscapes.pth",
        encoder="wrn50_2", decoder="aspp",
        note="STUB. WideResNet-50-2 is NOT an mmseg backbone — it is torchvision "
             "only. Needs either a custom backbone wrapper or "
             "segmentation_models_pytorch, and Cityscapes training either way. "
             "Against deeplabv3plus_r50 it isolates WIDTH at fixed depth, "
             "decoder and family."),

    "deeplabv3plus_convnext_t": ArchSpec(
        "deeplabv3plus_convnext_t", "none",
        cfg=f"{_M}/deeplabv3plus_convnext_t.py",
        weights=f"{_M}/deeplabv3plus_convnext_t_512x1024_cityscapes.pth",
        encoder="convnext_t", decoder="aspp",
        note="STUB. Non-standard pairing: ConvNeXt is normally served by "
             "UPerNet. Worth it only as the modern-conv cell of the ASPP row; "
             "if it is a fight, prefer upernet_convnext_t and accept that the "
             "decoder moves with it."),

    "deeplabv3plus_swin_t": ArchSpec(
        "deeplabv3plus_swin_t", "local",
        cfg=f"{_M}/deeplabv3plus_swin_t.py",
        weights=f"{_M}/deeplabv3plus_swin_t_512x1024_cityscapes.pth",
        encoder="swin_t", decoder="aspp",
        note="STUB. Windowed attention under an ASPP head. Multi-scale output, "
             "so the pairing is mechanically fine; no stock Cityscapes config."),

    "deeplabv3plus_vit_s": ArchSpec(
        "deeplabv3plus_vit_s", "global",
        cfg=f"{_M}/deeplabv3plus_vit_s.py",
        weights=f"{_M}/deeplabv3plus_vit_s_512x1024_cityscapes.pth",
        encoder="vit_s", decoder="aspp",
        note="STUB. ViT is SINGLE-SCALE, so ASPP sees one feature map and the "
             "pyramid degenerates. Buildable, but read this cell as 'ASPP on a "
             "global-attention encoder', not as a like-for-like DeepLab."),

    # ── UNet decoder, encoder varies ─────────────────────────────────────────
    "unet_r50": ArchSpec(
        "unet_r50", "none",
        cfg=f"{_M}/unet_r50.py",
        weights=f"{_M}/unet_r50_512x1024_cityscapes.pth",
        encoder="resnet50", decoder="unet",
        note="STUB. Against deeplabv3plus_r50 this is THE decoder contrast at "
             "fixed encoder — the single most informative cell in the grid, and "
             "the one that answers 'reach never built' vs 'reach discarded'. "
             "mmseg's UNet backbone is not swappable in stock configs, so this "
             "needs segmentation_models_pytorch or a custom config."),

    "unet_wrn50_2": ArchSpec(
        "unet_wrn50_2", "none",
        cfg=f"{_M}/unet_wrn50_2.py",
        weights=f"{_M}/unet_wrn50_2_512x1024_cityscapes.pth",
        encoder="wrn50_2", decoder="unet",
        note="STUB. Width contrast inside the UNet row, mirroring "
             "deeplabv3plus_wrn50_2 in the ASPP row."),

    "unet_convnext_t": ArchSpec(
        "unet_convnext_t", "none",
        cfg=f"{_M}/unet_convnext_t.py",
        weights=f"{_M}/unet_convnext_t_512x1024_cityscapes.pth",
        encoder="convnext_t", decoder="unet",
        note="STUB. Modern-conv encoder with skip connections intact."),

    "unet_swin_t": ArchSpec(
        "unet_swin_t", "local",
        cfg=f"{_M}/unet_swin_t.py",
        weights=f"{_M}/unet_swin_t_512x1024_cityscapes.pth",
        encoder="swin_t", decoder="unet",
        note="STUB. Windowed attention with skips — the cell that asks whether "
             "UNet's skip path can carry reach the encoder does have."),

    "unet_vit_s": ArchSpec(
        "unet_vit_s", "global",
        cfg=f"{_M}/unet_vit_s.py",
        weights=f"{_M}/unet_vit_s_512x1024_cityscapes.pth",
        encoder="vit_s", decoder="unet",
        note="STUB. A plain ViT emits ONE scale, so a UNet decoder has no "
             "natural skips to attach to and needs a feature pyramid faked from "
             "intermediate blocks. Declare that construction if this cell is "
             "reported — it is not a stock UNet."),

    # ── UPerNet decoder — revived from the commented block ───────────────────
    "upernet_convnext_t": ArchSpec(
        "upernet_convnext_t", "none",
        cfg=f"{_M}/upernet_convnext_t.py",
        weights=f"{_M}/upernet_convnext_t_512x1024_cityscapes.pth",
        encoder="convnext_t", decoder="upernet",
        note="STUB, and previously DEAD CODE inside a string literal. Stock "
             "mmseg, no DCNv3 needed. Yuan et al. Table 1 used exactly "
             "UPerNet/ConvNeXt for their most robust row (targeted attack mIoU "
             "5.86), so it is the faithful comparison point. ADE20K weights are "
             "sufficient for the label-free probes."),

    "upernet_swin_t": ArchSpec(
        "upernet_swin_t", "local",
        cfg=f"{_M}/upernet_swin_t.py",
        weights=f"{_M}/upernet_swin_t_512x1024_cityscapes.pth",
        encoder="swin_t", decoder="upernet",
        note="STUB, and previously DEAD CODE inside a string literal. Cityscapes "
             "weights are not in the stock mmseg zoo, but ADE20K weights are "
             "FINE for measure_erf.py and measure_frequency_sensitivity.py, so "
             "the local-attention bracket can enter both label-free grids "
             "without any Cityscapes training. Held against internimage_t it is "
             "also a decoder-matched attention contrast."),

    # ── mask-transformer decoder ─────────────────────────────────────────────
    "segmenter_vit_s": ArchSpec(
        "segmenter_vit_s", "global",
        cfg=f"{_M}/segmenter_vit_s.py",
        weights=f"{_M}/segmenter_vit-s_mask_512x1024_cityscapes.pth",
        encoder="vit_s", decoder="mask_tf",
        note="STUB. The natural home for ViT-S and the small counterpart to "
             "setr_pup's ViT-L, so the pair separates CAPACITY from attention "
             "scope. mmseg ships Segmenter for ADE20K; Cityscapes needs "
             "training."),

    "internimage_t_segmenter": ArchSpec(
        "internimage_t_segmenter", "none",
        cfg=f"{_M}/internimage_t_segmenter.py",
        weights=f"{_M}/internimage_t_segmenter_512x1024_cityscapes.pth",
        encoder="dcnv3_t", decoder="mask_tf",
        note="STUB. THE HYBRID PROBE: a deformable-conv encoder under a "
             "transformer decoder. Against internimage_t (same encoder, UPerNet "
             "head) it asks whether a global decoder can supply reach the "
             "encoder never builds — which is the mechanism question the whole "
             "factorial exists for, in one pair."),
}

# Convenience aliases so short names keep working. EVERY ONE MUST NAME A KEY
# THAT EXISTS: an alias outlived its entry once and raised KeyError at IMPORT
# time, which took down every script, because registry.py is imported by
# _common. The loop below fails with the offending pair rather than a bare
# KeyError, and it is cheap enough to keep.
_ALIASES = {
    "internimage": "internimage_t",
    "segformer": "segformer_b0",
    "unet": "unet_s5d16",
    "deeplab": "deeplabv3plus_r101",
    "swin": "upernet_swin_t",
    "convnext": "upernet_convnext_t",
    "segmenter": "segmenter_vit_s",
}
for _short, _full in _ALIASES.items():
    if _full not in REGISTRY:
        raise KeyError(f"alias {_short!r} points at {_full!r}, which is not in "
                       f"REGISTRY. Known: {sorted(REGISTRY)}")
    REGISTRY[_short] = REGISTRY[_full]


def grid(available_only: bool = False):
    """
    The encoder x decoder factorial as rows, for T03 / T04 / T16.

    Aliases are skipped -- they are the same object under a second key and
    would double-count a cell.

    `available_only` keeps the entries whose files are actually on disk, which
    is what an ATTACK table wants. The label-free probes (measure_erf.py,
    measure_frequency_sensitivity.py) never touch ground truth and so run on
    ADE20K weights, meaning they can cover cells the attack tables cannot --
    which is the whole tiering argument: let the free grid decide which cells
    are worth a Cityscapes training run.
    """
    seen, rows = set(), []
    for name, spec in REGISTRY.items():
        if name in _ALIASES or id(spec) in seen:
            continue
        seen.add(id(spec))
        if available_only and not spec.available:
            continue
        rows.append({"arch": name, "encoder": spec.encoder,
                     "decoder": spec.decoder, "bracket": spec.bracket,
                     "available": spec.available})
    return sorted(rows, key=lambda r: (r["decoder"], r["encoder"]))


def resolve(name: str, cfg_override=None, weights_override=None):
    """(cfg, weights, spec) with CLI overrides applied. Raises if unresolved."""
    if name not in REGISTRY:
        raise KeyError(f"unknown arch {name!r}. Known: "
                       f"{sorted(k for k in REGISTRY)}")
    spec = REGISTRY[name]
    cfg = cfg_override or spec.cfg
    weights = weights_override or spec.weights

    # THE TERNARY USED TO SWALLOW THE WHOLE MESSAGE. `"a" f"b" if note else ""`
    # binds over the entire concatenation, so an entry with no note raised
    # SystemExit("") -- an empty line and a non-zero exit, naming neither the
    # arch nor what to do about it. Every stub added for the factorial would
    # have hit exactly that path.
    if not cfg or not weights:
        raise SystemExit(
            f"--arch {name} has no registered paths.\n"
            f"  Either fill cfg/weights into patchreach/models/registry.py,\n"
            f"  or pass --cfg_path <config>.py --weights <ckpt>.pth\n"
            + (f"  ({spec.note})\n" if spec.note else ""))

    # A STUB NAMES WHERE ITS FILES GO. The factorial entries carry the paths
    # they expect, so the fix is "put the file here" -- but without this check
    # the first sign of a missing one is mmcv's FileNotFoundError several
    # frames deep, after the arch has already been accepted.
    missing = [str(f) for f in (cfg, weights) if not Path(f).exists()]
    if missing:
        raise SystemExit(
            f"--arch {name} is registered but its files are not on disk:\n"
            + "".join(f"    {m}\n" for m in missing)
            + f"  Put them there, or pass --cfg_path / --weights.\n"
            + (f"  ({spec.note})\n" if spec.note else ""))
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
