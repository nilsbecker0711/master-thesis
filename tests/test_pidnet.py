r"""
PIDNet: the registry wiring, and the config numbers the checkpoint depends on.

WHAT A SILENT REGRESSION WOULD BREAK HERE. PIDNet is vendored, not installed
(mmseg 0.x has no PIDNet — see patchreach/models/pidnet.py), so nothing
upstream guards the architecture constants. Every one of them is load-bearing
for `load_checkpoint`: get `channels` or `num_stem_blocks` wrong and the build
succeeds, the state_dict reports a page of size mismatches that is easy to
scroll past, and the model runs on partly random weights. A clean score ~10
mIoU below the published one is the only symptom, and it looks like a
resolution or normalisation problem rather than a config problem.

The config tests need neither mmseg nor torch: the configs are plain Python
assignments on purpose. The build test is skipped where mmseg is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patchreach.models.registry import REGISTRY, resolve

# (arch, backbone channels, ppm_channels, num_stem_blocks, num_branch_blocks,
#  head in_channels, head channels) — transcribed from the mmseg v1.2.2 zoo
# configs the published checkpoints were trained with.
VARIANTS = [
    ("pidnet_s", 32, 96, 2, 3, 128, 128),
    ("pidnet_m", 64, 96, 2, 3, 256, 128),
    ("pidnet_l", 64, 112, 3, 4, 256, 256),
]


def _load_cfg(path: Path) -> dict:
    """exec the config as plain Python. No mmcv, no _base_ resolution."""
    ns: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("arch,ch,ppm,stem,branch,head_in,head_ch", VARIANTS)
def test_config_matches_zoo_architecture(arch, ch, ppm, stem, branch, head_in,
                                         head_ch):
    """The seven numbers that decide whether the checkpoint's keys fit."""
    cfg_path, weights, spec = resolve(arch)
    cfg = _load_cfg(Path(cfg_path))["model"]

    b, h = cfg["backbone"], cfg["decode_head"]
    assert (b["type"], h["type"]) == ("PIDNet", "PIDHead")
    assert (b["channels"], b["ppm_channels"]) == (ch, ppm)
    assert (b["num_stem_blocks"], b["num_branch_blocks"]) == (stem, branch)
    assert (h["in_channels"], h["channels"]) == (head_in, head_ch)
    assert h["num_classes"] == 19

    # The backbone fuses to channels*4 and hands exactly that to the head. The
    # two are configured independently, so nothing but this catches a mismatch
    # before load_checkpoint does.
    assert h["in_channels"] == ch * 4

    # align_corners differs BETWEEN the two by design: False inside the
    # backbone's branch upsamples, True on the head — and EncoderDecoder reads
    # the head's for the final resize to input size.
    assert b["align_corners"] is False
    assert h["align_corners"] is True


@pytest.mark.parametrize("arch", [v[0] for v in VARIANTS])
def test_config_is_in_the_repo_and_not_in_the_workspace(arch):
    """
    Unlike every other arch, the config ships WITH the code that implements
    PIDNet — a cluster path here would mean a fresh clone could not build it.
    """
    cfg_path, weights, _ = resolve(arch)
    assert Path(cfg_path).is_file(), cfg_path
    assert Path(cfg_path).resolve().is_relative_to(ROOT.resolve())
    assert weights.endswith(".pth")


@pytest.mark.parametrize("arch", [v[0] for v in VARIANTS])
def test_whole_inference_so_auto_does_not_slide(arch):
    """
    test_cfg declares no crop_size/stride, so `--inference auto` MUST resolve
    to whole — _common.setup_model raises rather than guessing if it does not.
    """
    cfg_path, _, _ = resolve(arch)
    assert _load_cfg(Path(cfg_path))["model"]["test_cfg"]["mode"] == "whole"


def test_bracket_is_none():
    """
    No self-attention anywhere in PIDNet: the receptive field grows by
    convolution, striding and pyramid pooling only. It groups with deeplab and
    internimage, and a 'global' here would put it in the wrong ERF row.
    """
    for arch, *_ in VARIANTS:
        assert REGISTRY[arch].bracket == "none"


def test_pidnet_alias_is_l():
    """
    `--arch pidnet` must mean the 80.89 model, which is what Maag & Fischer
    Table 1 call 'PIDNet'. S (78.74) would quietly break that comparison.
    """
    assert REGISTRY["pidnet"] is REGISTRY["pidnet_l"]


def test_bottleneck_expansion_is_two():
    """
    PIDNet's Bottleneck expands by 2, not ResNet's 4. _make_layer sizes every
    downsample and every follow-on block off it, so a 4 builds a model whose
    checkpoint keys all have the wrong shape.
    """
    pytest.importorskip("mmseg")
    from patchreach.models.pidnet import BasicBlock, Bottleneck
    assert BasicBlock.expansion == 1
    assert Bottleneck.expansion == 2


def test_mmengine_era_checkpoint_unpickles_under_the_stub(tmp_path):
    r"""
    The PIDNet zoo checkpoints were written by mmseg 1.x, so their `meta`
    pickles mmengine objects (ConfigDict, HistoryBuffer). torch.load unpickles
    the WHOLE file before mmcv looks at the weights, so in this mmseg 0.x env
    the load died with `ModuleNotFoundError: No module named 'mmengine'` --
    after printing "load checkpoint from local path", which made it look like
    a corrupt download rather than a version artifact. Cost a cluster job.

    Builds a checkpoint carrying exactly that kind of object, so the test
    needs no 170 MB download: the stub is used to MINT the pickle, then the
    real assertion is that a plain load fails and a stubbed one does not.
    """
    torch = pytest.importorskip("torch")
    from patchreach.models.registry import _mmengine_stub

    with _mmengine_stub():
        import mmengine.logging.history_buffer as hb
        obj = hb.HistoryBuffer()
        obj["count"] = 7
        payload = {"state_dict": {"w": torch.ones(2, 2)},
                   "meta": {"mmengine_version": "0.10.3",
                            "message_hub": obj,
                            "dataset_meta": {"classes": ("road", "car"),
                                             "palette": [[1, 2, 3], [4, 5, 6]]}}}
        torch.save(payload, tmp_path / "ck.pth")

    # Exactly the cluster's failure.
    with pytest.raises(ModuleNotFoundError) as ei:
        torch.load(tmp_path / "ck.pth", map_location="cpu")
    assert ei.value.name.split(".")[0] == "mmengine"

    with _mmengine_stub():
        ck = torch.load(tmp_path / "ck.pth", map_location="cpu")
    assert torch.equal(ck["state_dict"]["w"], torch.ones(2, 2))
    assert ck["meta"]["dataset_meta"]["classes"] == ("road", "car")

    # Scoped: no stub module and no finder may outlive the context, or a real
    # mmengine installed beside us would be permanently shadowed.
    assert not [m for m in sys.modules if m.startswith("mmengine")]


def test_only_mmengine_triggers_the_retry():
    """
    A missing mmseg_custom or ops_dcnv3 is a real install problem and must
    keep propagating -- the stub must not become a catch-all for ImportError.
    """
    from patchreach.models.registry import _is_mmengine_missing
    assert _is_mmengine_missing(ModuleNotFoundError(name="mmengine"))
    assert _is_mmengine_missing(
        ModuleNotFoundError(name="mmengine.logging.history_buffer"))
    for other in ("mmseg_custom", "ops_dcnv3", "mmengine_extra", "", None):
        assert not _is_mmengine_missing(ModuleNotFoundError(name=other))


@pytest.mark.parametrize("arch", [v[0] for v in VARIANTS])
def test_builds_and_emits_19_channels_at_input_resolution(arch):
    """
    Build each variant from its config and run the eval path end to end.

    Three things at once: that the vendored modules register into mmseg 0.x,
    that the backbone's single-tensor eval return reaches PIDHead without a
    _transform_inputs index error, and that encode_decode resizes back to the
    INPUT size — not the stride-8 grid — which every metric and diagnostic
    assumes. No checkpoint is loaded; this is shape only.
    """
    pytest.importorskip("mmseg")
    torch = pytest.importorskip("torch")
    from mmcv import Config
    from mmseg.models import build_segmentor

    import patchreach.models.pidnet            # noqa: F401  (registers)

    cfg_path, _, _ = resolve(arch)
    cfg = Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.model.get("test_cfg"))
    model.eval()

    h, w = 128, 256
    metas = [{"ori_shape": (h, w, 3), "img_shape": (h, w, 3),
              "pad_shape": (h, w, 3), "scale_factor": 1.0, "flip": False,
              "flip_direction": None}]
    with torch.no_grad():
        out = model.encode_decode(torch.zeros(1, 3, h, w), metas)
    assert out.shape == (1, 19, h, w)
