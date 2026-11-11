"""The defence wrapper, SAC's shape completion, and the localisation metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from patchreach.data.cityscapes import norm_tensors
from defences import DEFENCES, build_defence
from defences.base import PatchDefence, Purified
from defences.sac.defence import (SAC_BREAK_ITS, SAC_INIT_L1_THRESH,
                                     SAC_SQUARE_SIZES, SAC_THRESH_DECAY,
                                     completion_pass, shape_completion)
from defences.sac.unet import UNet, load_sac_unet
from defences.wrapper import DefendedSegModel
from defences.detection import (DETECT_COVERAGE, aggregate_detection,
                                          mask_detection)

DEV = torch.device("cpu")
MEAN, STD = norm_tensors(DEV)


def square(top, left, side, h=64, w=64):
    m = torch.zeros(1, 1, h, w)
    m[..., top:top + side, left:left + side] = 1.0
    return m


class _Echo(nn.Module):
    """Inner 'model' that returns the input it was given, as 2-class logits."""

    mode = "whole"

    def forward(self, x):
        return torch.stack([x.mean(1), -x.mean(1)], dim=1)


class _FixedMask(PatchDefence):
    """Erases a known region, so the wrapper can be tested without weights."""

    name = "fixed"

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("m", mask)

    def forward(self, x01):
        return Purified(x01=x01 * (1.0 - self.m), mask=self.m, raw_mask=self.m)


def test_perfect_square_completes_to_itself():
    s = square(20, 20, 16)
    assert torch.equal(shape_completion(s, (16,)), s)


@pytest.mark.parametrize("drop", [0.1, 0.3, 0.5])
def test_eroded_square_still_fully_covers_the_patch(drop):
    """THE GUARANTEE THE PAPER SELLS."""
    true = square(20, 20, 16)
    raw = true.clone().flatten()
    idx = true.flatten().nonzero().squeeze(1)
    g = torch.Generator().manual_seed(0)
    kill = idx[torch.randperm(len(idx), generator=g)[:int(len(idx) * drop)]]
    raw[kill] = 0.0
    out = shape_completion(raw.view_as(true), (16,))
    assert float((out.bool() & true.bool()).sum() / true.sum()) == 1.0


def test_empty_raw_mask_completes_to_nothing():
    """A clean image the segmenter did not fire on must come back untouched. total = inside = 0 gives normalised hamming 1.0, and acceptance needs strictly below 1 - l1_thresh, so no threshold decay can ever accept."""
    out = shape_completion(torch.zeros(1, 1, 64, 64), SAC_SQUARE_SIZES)
    assert float(out.sum()) == 0.0


def test_scattered_false_positives_complete_to_nothing():
    m = torch.zeros(1, 1, 64, 64)
    for y, x in ((5, 5), (40, 50), (20, 60)):
        m[0, 0, y, x] = 1.0
    assert float(shape_completion(m, (16,)).sum()) == 0.0


def test_the_false_positive_trigger_matches_its_closed_form():
    """A square is accepted iff l1_thresh < (2*inside - total)/size^2, so for n CLUSTERED false positives acceptance needs l1_thresh < n/size^2."""
    thr_min = SAC_INIT_L1_THRESH * SAC_THRESH_DECAY ** (SAC_BREAK_ITS - 1)
    trigger = thr_min * min(SAC_SQUARE_SIZES) ** 2
    assert 3.0 < trigger < 4.0

    def erased(n, h, w):
        m = torch.zeros(1, 1, h, w)
        m[0, 0, h // 3:h // 3 + n, w // 3] = 1.0
        return float(shape_completion(m, SAC_SQUARE_SIZES).sum())

    assert erased(3, 256, 512) == 0.0
    assert erased(4, 256, 512) > 0.0
    assert erased(4, 256, 512) == erased(4, 512, 1024)


def test_a_smaller_square_makes_the_false_positive_much_worse():
    """The argument against tuning --sac_square_sizes downward."""
    m = torch.zeros(1, 1, 64, 64)
    m[0, 0, 30:32, 30] = 1.0
    assert float(shape_completion(m, (16,)).mean()) > 0.10
    assert float(shape_completion(m, SAC_SQUARE_SIZES).mean()) == 0.0


def test_a_saturated_raw_mask_completes_to_nothing_at_eval_resolution():
    """An untrained or broken segmenter fires everywhere."""
    assert float(shape_completion(torch.ones(1, 1, 256, 512),
                                  SAC_SQUARE_SIZES).sum()) == 0.0
    assert float(shape_completion(torch.ones(1, 1, 128, 128),
                                  SAC_SQUARE_SIZES).sum()) > 0.0


def test_completion_is_per_image_not_per_batch():
    """The reference loops over images, so the threshold-decay count is per image."""
    batch = torch.cat([square(20, 20, 16),
                       torch.zeros(1, 1, 64, 64),
                       square(5, 40, 16)], dim=0)
    batched = shape_completion(batch, (16,))
    singles = torch.cat([shape_completion(batch[i:i + 1], (16,))
                         for i in range(batch.shape[0])], dim=0)
    assert torch.equal(batched, singles)


def test_completion_pass_declines_a_square_larger_than_the_frame():
    assert completion_pass(torch.zeros(1, 1, 32, 32), 64, 0.9) is None


def test_every_square_larger_than_the_frame_is_refused_loudly():
    with pytest.raises(SystemExit, match="larger than the"):
        shape_completion(square(2, 2, 4, h=32, w=32), (64, 128))


def test_unet_round_trips_its_own_state_dict_strictly(tmp_path):
    """load_sac_unet uses strict=True on purpose: a non-strictly loaded U-Net still runs and still emits a mask, and that mask is noise."""
    ck = tmp_path / "w.pth"
    torch.save(UNet(3, 1, base_filter=4).state_dict(), ck)
    net = load_sac_unet(str(ck), base_filter=4, log=lambda *a: None)
    out = net(torch.rand(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128)


def test_unet_loader_strips_a_dataparallel_prefix(tmp_path):
    ck = tmp_path / "w.pth"
    sd = {f"module.{k}": v
          for k, v in UNet(3, 1, base_filter=4).state_dict().items()}
    torch.save(sd, ck)
    load_sac_unet(str(ck), base_filter=4, log=lambda *a: None)


def test_unet_loader_refuses_a_wrong_base_filter(tmp_path):
    ck = tmp_path / "w.pth"
    torch.save(UNet(3, 1, base_filter=4).state_dict(), ck)
    with pytest.raises(RuntimeError):
        load_sac_unet(str(ck), base_filter=8, log=lambda *a: None)


def test_missing_checkpoint_names_the_download(tmp_path):
    with pytest.raises(SystemExit, match="SegmentAndComplete"):
        load_sac_unet(str(tmp_path / "absent.pth"), log=lambda *a: None)


def test_normalisation_round_trip_is_exact():
    """The defences take [0,1] RGB and the pipeline carries ImageNet-normalised tensors."""
    w = DefendedSegModel(_Echo(), _FixedMask(torch.zeros(1, 1, 8, 8)),
                         MEAN, STD)
    x01 = torch.rand(2, 3, 8, 8)
    assert torch.allclose(w.to_pixels(w.to_normed(x01)), x01, atol=1e-6)


def test_bypassed_reproduces_the_undefended_forward_exactly():
    """THE 2x2 RESTS ON THIS."""
    inner = _Echo()
    w = DefendedSegModel(inner, _FixedMask(torch.ones(1, 1, 8, 8)), MEAN, STD)
    x = torch.randn(1, 3, 8, 8)
    with w.bypassed():
        got = w(x)
    assert torch.equal(got, inner(x))
    assert w.last is None


def test_defence_is_restored_after_bypass_even_on_error():
    w = DefendedSegModel(_Echo(), _FixedMask(torch.zeros(1, 1, 8, 8)),
                         MEAN, STD)
    with pytest.raises(ValueError):
        with w.bypassed():
            raise ValueError("boom")
    assert w.enabled is True


def test_defended_forward_feeds_the_model_the_purified_image():
    """The composition under test: denormalise -> purify in [0,1] -> renormalise -> inner model."""
    m = torch.zeros(1, 1, 16, 16)
    m[..., 4:8, 4:8] = 1.0
    inner = _Echo()
    w = DefendedSegModel(inner, _FixedMask(m), MEAN, STD)
    x01 = torch.rand(1, 3, 16, 16)
    got = w(w.to_normed(x01))

    expected = inner(w.to_normed(x01 * (1.0 - m)))
    assert torch.allclose(got, expected, atol=1e-6)
    assert w.last is not None
    assert torch.equal(w.last.mask, m)
    assert float(w.last.x01[..., 4:8, 4:8].abs().max()) == 0.0
    assert float((w.last.x01 - x01)[..., 8:, 8:].abs().max()) < 1e-6


def test_grad_carrying_input_is_refused():
    """The guard that stops an attack training through a dead gradient."""
    w = DefendedSegModel(_Echo(), _FixedMask(torch.zeros(1, 1, 8, 8)),
                         MEAN, STD)
    x = torch.randn(1, 3, 8, 8, requires_grad=True)
    with pytest.raises(RuntimeError, match="ZERO"):
        w(x)
    with pytest.raises(RuntimeError, match="BPDA"):
        w(torch.randn(1, 3, 8, 8) * torch.ones(1, requires_grad=True))


def test_bypassed_still_allows_gradients():
    """Reach diagnostics on the undefended model must keep working."""
    inner = _Echo()
    w = DefendedSegModel(inner, _FixedMask(torch.zeros(1, 1, 8, 8)), MEAN, STD)
    x = torch.randn(1, 3, 8, 8, requires_grad=True)
    with w.bypassed():
        w(x).sum().backward()
    assert x.grad is not None


def test_wrapper_reports_the_inner_inference_mode():
    class Slide(_Echo):
        mode = "slide"
    w = DefendedSegModel(Slide(), _FixedMask(torch.zeros(1, 1, 8, 8)),
                         MEAN, STD)
    assert w.mode == "slide"


def test_unimplemented_defences_are_named_and_fail_informatively():
    """ram is registered with a None builder so a sweep over the full list fails on the cell rather than silently skipping it, and so the error says 'not implemented' rather than 'invalid choice'. sac and jedi are implemented; see test_jedi.py for jedi's own suite."""
    assert {"none", "sac", "jedi", "ram"} <= set(DEFENCES)
    with pytest.raises(SystemExit, match="not implemented yet"):
        build_defence("ram")


def test_unknown_defence_lists_the_known_ones():
    with pytest.raises(SystemExit, match="unknown --defence"):
        build_defence("lgs")


def test_none_is_not_a_buildable_defence():
    """--defence none must add NO wrapper, so building one is a caller bug."""
    with pytest.raises(ValueError):
        build_defence("none")


def test_coverage_and_precision_split_the_two_failure_modes():
    true = square(8, 8, 8, h=32, w=32).bool()
    exact = mask_detection(true, true)
    assert exact["coverage"] == 1.0 and exact["precision"] == 1.0

    half = torch.zeros_like(true)
    half[..., 8:12, 8:16] = True
    assert mask_detection(half, true)["coverage"] == pytest.approx(0.5)
    assert mask_detection(half, true)["precision"] == 1.0

    big = square(4, 4, 16, h=32, w=32).bool()
    over = mask_detection(big, true)
    assert over["coverage"] == 1.0
    assert over["precision"] < 0.3
    assert over["fp_area_frac"] > 0.1


def test_clean_image_has_no_footprint_so_ratios_are_nan():
    fired = square(4, 4, 16, h=32, w=32).bool()
    got = mask_detection(fired, None)
    assert got["coverage"] != got["coverage"]
    assert got["fp_area_frac"] == pytest.approx(0.25)


def test_empty_prediction_scores_zero_coverage_not_nan():
    true = square(8, 8, 8, h=32, w=32).bool()
    got = mask_detection(torch.zeros_like(true), true)
    assert got["coverage"] == 0.0
    assert got["detected"] == 0.0


def test_detected_thresholds_coverage_at_the_module_constant():
    true = square(0, 0, 10, h=32, w=32).bool()
    just_over = torch.zeros_like(true)
    just_over[..., :6, :10] = True
    assert mask_detection(just_over, true)["coverage"] > DETECT_COVERAGE
    assert mask_detection(just_over, true)["detected"] == 1.0
    just_under = torch.zeros_like(true)
    just_under[..., :4, :10] = True
    assert mask_detection(just_under, true)["detected"] == 0.0


def test_aggregate_skips_nan_and_reports_its_own_n():
    """Clean images contribute NaN coverage by construction, so the aggregate has to carry how many images each field was actually averaged over — otherwise a coverage number silently describes a subset."""
    true = square(8, 8, 8, h=32, w=32).bool()
    rows = [mask_detection(true, true), mask_detection(true, None)]
    agg = aggregate_detection(rows)
    assert agg["coverage"] == 1.0
    assert agg["coverage_n"] == 1
    assert agg["fp_area_frac_n"] == 2


ROOT_DIR = Path(__file__).resolve().parents[1]
SEQUENCE = ("resolve_placement", "set_reference_from_image", "apply")


def _call_sequence(path):
    """The ordered patch-application calls in a file, duplicates collapsed."""
    src = path.read_text(encoding="utf-8")
    hits = []
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        for name in SEQUENCE:
            if f".{name}(" in code and (not hits or hits[-1] != name):
                hits.append(name)
    return hits


def test_the_per_image_sequence_matches_evaluate_py():
    """THE ONE INVARIANT THIS LAYOUT CANNOT ENFORCE BY CONSTRUCTION."""
    mine = _call_sequence(Path(__file__).with_name("evaluate_defended.py"))
    theirs = _call_sequence(ROOT_DIR / "scripts" / "evaluate.py")
    assert mine == list(SEQUENCE), mine
    assert theirs == list(SEQUENCE), theirs


def test_scripts_is_not_modified_to_know_about_defences():
    """The whole point of this folder."""
    for sub in ("scripts", "patchreach"):
        for f in (ROOT_DIR / sub).rglob("*.py"):
            src = f.read_text(encoding="utf-8")
            assert "defence" not in src.lower(), f"{f} mentions a defence"
            assert "defenses" not in src, f"{f} mentions defenses"


def test_the_entry_point_only_borrows_read_only_plumbing():
    """Documents the coupling surface deliberately."""
    src = (Path(__file__).with_name("evaluate_defended.py")
           .read_text(encoding="utf-8"))
    line = next(l for l in src.splitlines() if l.startswith("from _common"))
    assert line == ("from _common import add_model_args, setup_model, "
                    "make_dataset, image_indices"), line
