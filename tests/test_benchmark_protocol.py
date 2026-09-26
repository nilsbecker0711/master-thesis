"""
The Nesti/Rossolini benchmark protocol: mAcc, the random-patch baseline, and
the area-matched patch geometry benchmark.sh launches.

What a silent regression would break here is not a number but a COMPARISON.
mAcc that quietly averaged over the wrong class set, a "random patch" row
that was actually a grey square, or a --patch_size that disagreed with
int(img_h * --patch_scale) would each produce a table that looks fine and is
not comparable to the paper it claims to sit beside — or, for the last one,
a run that refuses to start at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from patchreach.data.cityscapes import norm_tensors
from patchreach.metrics.miou import SegMetric, compare
from patchreach.patch.placement import footprint_side
from patchreach.patch.spec import Patch, PatchConfig

DEV = torch.device("cpu")
MEAN, STD = norm_tensors(DEV)


# ── mAcc ─────────────────────────────────────────────────────────────────────

def _hand_case():
    """class 0: 8 px, 6 correct, 2 -> class 1.  class 1: 4 px, all correct.
    class 2: absent from the labels entirely."""
    lbl = torch.tensor([[0] * 8 + [1] * 4])
    prd = torch.tensor([[0] * 6 + [1] * 2 + [1] * 4])
    m = SegMetric(3)
    m.update(prd, lbl)
    return m, lbl


def test_macc_is_per_class_recall():
    m, _ = _hand_case()
    acc = m.per_class_acc()
    assert acc[0].item() == pytest.approx(75.0)      # 6/8
    assert acc[1].item() == pytest.approx(100.0)     # 4/4
    assert torch.isnan(acc[2])                       # absent from GT
    assert m.compute_acc() == pytest.approx(87.5)


def test_macc_is_not_miou():
    """They coincide only when nothing is over-predicted, so a test that
    passed on an accidental alias would be worthless."""
    m, _ = _hand_case()
    assert m.compute() == pytest.approx(70.8333, abs=1e-3)
    assert m.compute_acc() == pytest.approx(87.5)


def test_macc_ignores_classes_absent_from_ground_truth():
    """A class that is only ever PREDICTED has gt == 0, so its recall is 0/0.
    per_class_acc must report nan, not 0 — a zero would drag mAcc down and
    read as a failure to segment something that was never there."""
    lbl = torch.tensor([[0, 0, 0, 0]])
    prd = torch.tensor([[0, 0, 2, 2]])
    m = SegMetric(3)
    m.update(prd, lbl)
    acc = m.per_class_acc()
    assert acc[0].item() == pytest.approx(50.0)
    assert torch.isnan(acc[1]) and torch.isnan(acc[2])
    assert m.compute_acc() == pytest.approx(50.0)


def test_compare_carries_acc_without_disturbing_the_existing_keys():
    m, lbl = _hand_case()
    adv = SegMetric(3)
    adv.update(torch.tensor([[0] * 12]), lbl)
    d = compare(m, adv)

    assert d["clean_acc"] == pytest.approx(87.5)
    assert d["adv_acc"] == pytest.approx(50.0)       # c0 8/8, c1 0/4
    assert d["drop_acc"] == pytest.approx(37.5)
    # No union variant: accuracy is normalised by the GT count, so the union
    # class set has no defined value for a GT-absent class.
    assert not any(k.startswith("clean_acc") and k.endswith("_union")
                   for k in d)
    # Every pre-existing key still present and unmoved.
    for k in ("clean", "adv", "drop", "clean_union", "adv_union",
              "drop_union", "n_classes_gt", "n_classes_clean_union",
              "n_classes_adv_union"):
        assert k in d


def test_compare_prefix_applies_to_the_acc_keys_too():
    m, lbl = _hand_case()
    d = compare(m, m, prefix="rem_")
    assert "rem_clean_acc" in d and "rem_drop_acc" in d


# ── the random-patch baseline ────────────────────────────────────────────────

def _patch(**kw):
    return Patch(PatchConfig(mode="raw", size=16, **kw), DEV, MEAN, STD)


def test_default_init_is_still_grey():
    """Every run recorded before --raw_init existed started at uniform 0.5.
    If this moves, no past raw number is restatable."""
    assert PatchConfig().raw_init == "grey"
    r = _patch().render()
    assert torch.allclose(r, torch.full_like(r, 0.5), atol=1e-6)


def test_random_init_is_actually_random():
    r = _patch(raw_init="random").render()
    assert r.std() > 0.2                 # uniform [0,1] has sd ~0.289
    assert r.min() < 0.05 and r.max() > 0.95
    assert not torch.allclose(r, torch.full_like(r, 0.5), atol=0.1)


def test_random_init_round_trips_through_the_sigmoid():
    """logit_seed is the one init code path; the rendered patch at step 0 must
    BE the uniform draw, not some squashed version of it."""
    torch.manual_seed(0)
    r = _patch(raw_init="random").render()
    torch.manual_seed(0)
    expected = torch.rand(3, 16, 16)
    assert (r - expected).abs().max() < 1e-5


def test_random_init_works_under_direct_parameterisation():
    r = _patch(raw_init="random", pixel_param="direct").render()
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert r.std() > 0.2


@pytest.mark.parametrize("mode", ["csf", "universal_csf", "gan"])
def test_random_init_is_refused_where_it_would_do_nothing(mode):
    """Those modes override the pixel init, so accepting the flag would file a
    run as a random-init row that is not one. Checked through validate(),
    which is the entry point Patch.__init__ calls — a gan config cannot be
    constructed here without a generator."""
    with pytest.raises(ValueError, match="raw_init"):
        PatchConfig(mode=mode, raw_init="random").validate()


def test_raw_init_is_validated():
    with pytest.raises(ValueError, match="raw_init"):
        PatchConfig(raw_init="noise").validate()


def test_the_guard_fires_through_patch_construction_too():
    """validate() is only useful if the real path reaches it."""
    with pytest.raises(ValueError, match="raw_init"):
        Patch(PatchConfig(mode="csf", size=16, raw_init="random"),
              DEV, MEAN, STD)


def test_grey_and_random_are_different_controls():
    """The point of the flag: --lr 0 on the default measures OCCLUSION, and
    the benchmark's baseline column is not that."""
    g, r = _patch().render(), _patch(raw_init="random").render()
    assert (g - r).abs().mean() > 0.1


# ── the fixed geometry benchmark_slice.sh launches ───────────────────────────

BENCH_SCALE = 0.25
BENCH_SIZE = 256          # --patch_size in benchmark_slice.sh, at H=1024


def test_bench_size_survives_the_universal_csf_size_guard():
    """train.py refuses universal_csf unless --patch_size == int(img_h*scale),
    because resampling a residual resamples its spectrum. A benchmark whose
    two numbers disagree dies at startup, so this pairing is load-bearing."""
    assert int(1024 * BENCH_SCALE) == BENCH_SIZE


def test_the_benchmark_patch_covers_what_the_rest_of_the_thesis_does():
    """The reason for holding 0.25 rather than adopting Nesti's size ladder:
    scale_ref='height' fixes coverage by aspect ratio alone, so the native
    benchmark row covers the same fraction of the frame as every 512x1024
    run. Lose that and the row is comparable to their paper and to nothing
    of ours."""
    small = footprint_side(512, 1024, BENCH_SCALE) ** 2 / (512 * 1024)
    native = footprint_side(1024, 2048, BENCH_SCALE) ** 2 / (1024 * 2048)
    assert small == pytest.approx(native, rel=1e-9)
    assert native == pytest.approx(0.03125, rel=1e-9)


def test_our_operating_point_is_not_one_of_their_rungs():
    """Guard on the write-up, not the code. Their ladder is 2.14 / 3.82 /
    8.57% of the frame and ours is 3.125%; it sits BETWEEN the first two and
    matches none, so no row may be quoted against a specific rung as though
    the sizes agreed."""
    ours = footprint_side(1024, 2048, BENCH_SCALE) ** 2 / (1024 * 2048)
    theirs = [(150 * 300), (200 * 400), (300 * 600)]
    fracs = [a / (1024 * 2048) for a in theirs]
    assert all(abs(ours - f) / f > 0.10 for f in fracs)
    assert fracs[0] < ours < fracs[1]
