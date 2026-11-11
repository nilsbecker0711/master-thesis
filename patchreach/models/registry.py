"""Architecture registry — one entry per CHECKPOINT, not per family."""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from pathlib import Path
from typing import Optional

BRACKETS = {
    "none": "no global attention — small ERF, robust bracket",
    "local": "windowed attention — mid ERF, middle bracket",
    "global": "global/efficient self-attention — large ERF, vulnerable bracket",
}


@dataclass
class ArchSpec:
    name: str
    bracket: str
    cfg: Optional[str] = None
    weights: Optional[str] = None
    note: str = ""

    @property
    def family(self) -> str:
        return BRACKETS.get(self.bracket, self.bracket)


_W = "/pfs/work9/workspace/scratch/ma_nilbecke-thesis"
_M = f"{_W}/checkpoints"

_CFG = Path(__file__).resolve().parents[2] / "configs" / "mmseg"

REGISTRY = {

    "internimage_t": ArchSpec(
        "internimage_t", "none",
        cfg=f"{_M}/upernet_internimage_t_512x1024_160k_cityscapes.py",
        weights=f"{_M}/upernet_internimage_t_512x1024_160k_cityscapes.pth",
        note="UPerNet head. Emits a 150-channel ADE20K-dimensioned head; only "
             "0..18 are active. The channel probe reports this at startup."),
    "internimage_l": ArchSpec("internimage_l", "none"),
    "internimage_xl": ArchSpec("internimage_xl", "none"),

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

    "deeplabv3plus_r50": ArchSpec("deeplabv3plus_r50", "none",
        cfg=f"{_M}/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py",
        weights=f"{_M}/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth",
        note="Dilated ResNet-50, no attention. Same head, same entry stride 8 "
            "as r101 — the DEPTH ablation with geometry held constant."),

    "deeplabv3plus_r18": ArchSpec("deeplabv3plus_r18", "none",
        cfg=f"{_M}/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py",
        weights=f"{_M}/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth",
        note="Dilated ResNet-18, BasicBlock, 512-channel output — the head "
            "dims differ from r50/r101, so this is not a pure depth ablation."),

    "pidnet_s": ArchSpec("pidnet_s", "none",
        cfg=str(_CFG / "pidnet_s.py"),
        weights=f"{_M}/pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth",
        note="Smallest variant, clean mIoU 78.74. PAPPM + LightBag."),

    "pidnet_m": ArchSpec("pidnet_m", "none",
        cfg=str(_CFG / "pidnet_m.py"),
        weights=f"{_M}/pidnet-m_2xb6-120k_1024x1024-cityscapes_20230301_143452-f9bcdbf3.pth",
        note="PIDNet-S widened 32->64 channels, clean mIoU 80.22. Same graph "
             "as S (PAPPM + LightBag) — the pure WIDTH ablation within the "
             "family, with depth and module types held constant."),

    "pidnet_l": ArchSpec("pidnet_l", "none",
        cfg=str(_CFG / "pidnet_l.py"),
        weights=f"{_M}/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth",
        note="Clean mIoU 80.89 — the value Maag & Fischer (arXiv:2408.10021) "
             "Table 1 report for 'PIDNet', so this is the variant that makes "
             "docs/experiment_log.md's protocol comparison exact. Not just a "
             "wider M: num_stem_blocks=3 swaps PAPPM/LightBag for DAPPM/Bag."),
}

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

REGISTRY["internimage"] = REGISTRY["internimage_t"]
REGISTRY["segformer"] = REGISTRY["segformer_b0"]
REGISTRY["unet"] = REGISTRY["unet_s5d16"]
REGISTRY["deeplab_101"] = REGISTRY["deeplabv3plus_r101"]
REGISTRY["deeplab_50"] = REGISTRY["deeplabv3plus_r50"]
REGISTRY["deeplab_18"] = REGISTRY["deeplabv3plus_r18"]
REGISTRY["pidnet"] = REGISTRY["pidnet_l"]


def resolve(name: str, cfg_override=None, weights_override=None):
    """(cfg, weights, spec) with CLI overrides applied."""
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


class _StubLoader(Loader):
    """Produces a module whose every attribute is a fresh dict subclass."""

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        def __getattr__(name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (dict,), {"__module__": module.__name__})
            setattr(module, name, cls)
            return cls
        module.__getattr__ = __getattr__


class _StubFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mmengine" or fullname.startswith("mmengine."):
            return spec_from_loader(fullname, _StubLoader())
        return None


@contextmanager
def _mmengine_stub():
    """Let torch.load unpickle an mmseg **1.x** checkpoint inside an mmseg 0.x environment, by standing in a fake `mmengine` package for the duration."""
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        for name in [m for m in sys.modules
                     if m == "mmengine" or m.startswith("mmengine.")]:
            if isinstance(getattr(sys.modules[name], "__loader__", None),
                          _StubLoader):
                del sys.modules[name]


def _is_mmengine_missing(exc: ModuleNotFoundError) -> bool:
    """Is this the 1.x-meta pickle failure, or a genuinely absent dependency?"""
    return str(getattr(exc, "name", "") or "").split(".")[0] == "mmengine"


def _load_weights(model, weights_path: str):
    """mmcv's load_checkpoint, retried under the mmengine stub if the pickle needs it."""
    from mmcv.runner import load_checkpoint

    try:
        return load_checkpoint(model, weights_path, map_location="cpu")
    except ModuleNotFoundError as e:
        if not _is_mmengine_missing(e):
            raise
        print(f"[ckpt] mmseg 1.x-era checkpoint: stubbing out its mmengine "
              f"meta ({e.name}). Weights are unaffected.")
        with _mmengine_stub():
            return load_checkpoint(model, weights_path, map_location="cpu")


def load_segmentor(cfg_path: str, weights_path: str):
    """Build + load any mmseg EncoderDecoder."""
    from mmcv import Config
    from mmcv.runner import wrap_fp16_model
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(cfg_path)
    backbone = cfg.model.get("backbone", {}).get("type", "<unknown>")
    head = cfg.model.get("decode_head", {}).get("type", "<unknown>")

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

    if backbone == "PIDNet":
        import patchreach.models.pidnet            # noqa: F401

    print(f"[model] backbone={backbone}  head={head}  "
          f"num_classes={cfg.model.decode_head.get('num_classes')}")

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    ck = _load_weights(model, weights_path)
    meta = ck.get("meta", {}) or {}
    for k in ("CLASSES", "PALETTE"):
        if k in meta:
            setattr(model, k, meta[k])
    ds = meta.get("dataset_meta") or {}
    if isinstance(ds, dict):
        if not hasattr(model, "CLASSES") and ds.get("classes"):
            model.CLASSES = ds["classes"]
        if not hasattr(model, "PALETTE") and ds.get("palette"):
            model.PALETTE = ds["palette"]
    return model, backbone
