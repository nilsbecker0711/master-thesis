"""
The defence wrapper, SAC's shape completion, and the localisation metrics.

What a silent regression breaks here is a COMPARISON, as everywhere else in this
suite. `--defence none` that no longer reproduces the undefended forward makes
every defended/undefended pair incomparable. A shape completion that stopped
satisfying SAC's removal guarantee would quietly turn a coverage number into a
statement about erosion luck. A grad guard that stopped firing would let an
attack train through a zero-gradient block and report a robust model.
"""
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


# ── shape completion: SAC's removal guarantee ────────────────────────────────

def test_perfect_square_completes_to_itself():
    s = square(20, 20, 16)
    assert torch.equal(shape_completion(s, (16,)), s)


@pytest.mark.parametrize("drop", [0.1, 0.3, 0.5])
def test_eroded_square_still_fully_covers_the_patch(drop):
    """
    THE GUARANTEE THE PAPER SELLS. A raw mask within a bounded Hamming distance
    of the true square completes to cover that square, so the patch is removed
    even though the segmenter only found part of it. Coverage, not IoU, is the
    quantity with teeth — the completion is allowed to overshoot.
    """
    true = square(20, 20, 16)
    raw = true.clone().flatten()
    idx = true.flatten().nonzero().squeeze(1)
    g = torch.Generator().manual_seed(0)
    kill = idx[torch.randperm(len(idx), generator=g)[:int(len(idx) * drop)]]
    raw[kill] = 0.0
    out = shape_completion(raw.view_as(true), (16,))
    assert float((out.bool() & true.bool()).sum() / true.sum()) == 1.0


def test_empty_raw_mask_completes_to_nothing():
    """
    A clean image the segmenter did not fire on must come back untouched.
    total = inside = 0 gives normalised hamming 1.0, and acceptance needs
    strictly below 1 - l1_thresh, so no threshold decay can ever accept.
    """
    out = shape_completion(torch.zeros(1, 1, 64, 64), SAC_SQUARE_SIZES)
    assert float(out.sum()) == 0.0


def test_scattered_false_positives_complete_to_nothing():
    m = torch.zeros(1, 1, 64, 64)
    for y, x in ((5, 5), (40, 50), (20, 60)):
        m[0, 0, y, x] = 1.0
    assert float(shape_completion(m, (16,)).sum()) == 0.0


def test_the_false_positive_trigger_matches_its_closed_form():
    """
    A square is accepted iff l1_thresh < (2*inside - total)/size^2, so for n
    CLUSTERED false positives acceptance needs l1_thresh < n/size^2. The loop's
    loosest threshold is 0.9*0.7^14, hence completion fires iff
    n > 0.0061 * min(square_sizes)^2 — n >= 4 for SAC's smallest default square
    of 25, and independent of the frame size.

    Pinned because this is the mechanism behind the defence's entire clean-image
    cost, and because the threshold is implicit in four constants that a future
    tweak could move without anyone noticing which way.
    """
    thr_min = SAC_INIT_L1_THRESH * SAC_THRESH_DECAY ** (SAC_BREAK_ITS - 1)
    trigger = thr_min * min(SAC_SQUARE_SIZES) ** 2
    assert 3.0 < trigger < 4.0                       # so n=3 misses, n=4 fires

    def erased(n, h, w):
        m = torch.zeros(1, 1, h, w)
        m[0, 0, h // 3:h // 3 + n, w // 3] = 1.0
        return float(shape_completion(m, SAC_SQUARE_SIZES).sum())

    assert erased(3, 256, 512) == 0.0
    assert erased(4, 256, 512) > 0.0
    # the hole is a fixed ABSOLUTE size — the frame only changes the fraction
    assert erased(4, 256, 512) == erased(4, 512, 1024)


def test_a_smaller_square_makes_the_false_positive_much_worse():
    """The argument against tuning --sac_square_sizes downward."""
    m = torch.zeros(1, 1, 64, 64)
    m[0, 0, 30:32, 30] = 1.0
    assert float(shape_completion(m, (16,)).mean()) > 0.10
    assert float(shape_completion(m, SAC_SQUARE_SIZES).mean()) == 0.0


def test_a_saturated_raw_mask_completes_to_nothing_at_eval_resolution():
    """
    An untrained or broken segmenter fires everywhere. Acceptance needs
    l1_thresh < (2*inside - total)/size^2, and a full-frame mask makes that
    numerator NEGATIVE as soon as the frame is more than twice the area of the
    largest candidate square — so completion rejects it and fails safe.

    THE CONDITION IS NOT UNIVERSAL. On a frame small enough that a candidate
    square covers over half of it, a saturated mask IS accepted: 100x100 inside
    128x128 gives (20000-16384)/10000 = 0.36, which the fourth threshold decay
    clears. Every resolution in this project is far above that, but the guard is
    geometric rather than absolute, so it is pinned at both ends.
    """
    assert float(shape_completion(torch.ones(1, 1, 256, 512),
                                  SAC_SQUARE_SIZES).sum()) == 0.0
    # and the small-frame counter-case, so the bound is documented, not assumed
    assert float(shape_completion(torch.ones(1, 1, 128, 128),
                                  SAC_SQUARE_SIZES).sum()) > 0.0


def test_completion_is_per_image_not_per_batch():
    """
    The reference loops over images, so the threshold-decay count is per image.
    A batched loop without masking would let one image that completed on the
    first pass stop the search for the rest of the batch.
    """
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


# ── the U-Net loader ────────────────────────────────────────────────────────

def test_unet_round_trips_its_own_state_dict_strictly(tmp_path):
    """
    load_sac_unet uses strict=True on purpose: a non-strictly loaded U-Net still
    runs and still emits a mask, and that mask is noise. The only symptom is a
    defence that mysteriously does nothing, which reads exactly like a
    successful attack.
    """
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


# ── the wrapper ─────────────────────────────────────────────────────────────

def test_normalisation_round_trip_is_exact():
    """
    The defences take [0,1] RGB and the pipeline carries ImageNet-normalised
    tensors. A defence that silently received normalised input would still run
    and still produce a plausible mask, so this conversion gets a test of its
    own.
    """
    w = DefendedSegModel(_Echo(), _FixedMask(torch.zeros(1, 1, 8, 8)),
                         MEAN, STD)
    x01 = torch.rand(2, 3, 8, 8)
    assert torch.allclose(w.to_pixels(w.to_normed(x01)), x01, atol=1e-6)


def test_bypassed_reproduces_the_undefended_forward_exactly():
    """
    THE 2x2 RESTS ON THIS. Undefended and defended columns come from one model
    instance in one process; if bypassed() were not bit-exact against the raw
    inner model, the comparison would carry the wrapper's own arithmetic.
    """
    inner = _Echo()
    w = DefendedSegModel(inner, _FixedMask(torch.ones(1, 1, 8, 8)), MEAN, STD)
    x = torch.randn(1, 3, 8, 8)
    with w.bypassed():
        got = w(x)
    assert torch.equal(got, inner(x))
    assert w.last is None                      # no stale mask to mis-score


def test_defence_is_restored_after_bypass_even_on_error():
    w = DefendedSegModel(_Echo(), _FixedMask(torch.zeros(1, 1, 8, 8)),
                         MEAN, STD)
    with pytest.raises(ValueError):
        with w.bypassed():
            raise ValueError("boom")
    assert w.enabled is True


def test_defended_forward_feeds_the_model_the_purified_image():
    """
    The composition under test: denormalise -> purify in [0,1] -> renormalise ->
    inner model. Checked against the same chain computed by hand, because the
    failure mode is a defence that ran on the wrong range and still returned a
    plausible-looking mask.
    """
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
    # the erased block really is pixel-space black, not merely "different"
    assert float(w.last.x01[..., 4:8, 4:8].abs().max()) == 0.0
    # Pixels the mask does not claim survive, but only to float32 — the
    # denormalise/renormalise round trip moves them by ~1e-7. Harmless, and the
    # reason --defence none adds NO wrapper instead of wrapping an empty mask:
    # the undefended column has to be bit-exact, not nearly.
    assert float((w.last.x01 - x01)[..., 8:, 8:].abs().max()) < 1e-6


def test_grad_carrying_input_is_refused():
    """
    The guard that stops an attack training through a dead gradient. Without it
    the run completes, the patch never moves, and the numbers look like a robust
    model with nothing in the logs to say otherwise.
    """
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


# ── the registry ────────────────────────────────────────────────────────────

def test_unimplemented_defences_are_named_and_fail_informatively():
    """
    ram is registered with a None builder so a sweep over the full list fails on
    the cell rather than silently skipping it, and so the error says 'not
    implemented' rather than 'invalid choice'. sac and jedi are implemented;
    see test_jedi.py for jedi's own suite.
    """
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


# ── localisation metrics ────────────────────────────────────────────────────

def test_coverage_and_precision_split_the_two_failure_modes():
    true = square(8, 8, 8, h=32, w=32).bool()
    exact = mask_detection(true, true)
    assert exact["coverage"] == 1.0 and exact["precision"] == 1.0

    half = torch.zeros_like(true)
    half[..., 8:12, 8:16] = True
    assert mask_detection(half, true)["coverage"] == pytest.approx(0.5)
    assert mask_detection(half, true)["precision"] == 1.0

    # over-covering keeps coverage at 1 and is charged on precision and on the
    # false-positive area, which is the asymmetry the module docstring argues for
    big = square(4, 4, 16, h=32, w=32).bool()
    over = mask_detection(big, true)
    assert over["coverage"] == 1.0
    assert over["precision"] < 0.3
    assert over["fp_area_frac"] > 0.1


def test_clean_image_has_no_footprint_so_ratios_are_nan():
    fired = square(4, 4, 16, h=32, w=32).bool()
    got = mask_detection(fired, None)
    assert got["coverage"] != got["coverage"]            # NaN
    assert got["fp_area_frac"] == pytest.approx(0.25)


def test_empty_prediction_scores_zero_coverage_not_nan():
    true = square(8, 8, 8, h=32, w=32).bool()
    got = mask_detection(torch.zeros_like(true), true)
    assert got["coverage"] == 0.0
    assert got["detected"] == 0.0


def test_detected_thresholds_coverage_at_the_module_constant():
    true = square(0, 0, 10, h=32, w=32).bool()
    just_over = torch.zeros_like(true)
    just_over[..., :6, :10] = True                       # coverage 0.6
    assert mask_detection(just_over, true)["coverage"] > DETECT_COVERAGE
    assert mask_detection(just_over, true)["detected"] == 1.0
    just_under = torch.zeros_like(true)
    just_under[..., :4, :10] = True                      # coverage 0.4
    assert mask_detection(just_under, true)["detected"] == 0.0


def test_aggregate_skips_nan_and_reports_its_own_n():
    """
    Clean images contribute NaN coverage by construction, so the aggregate has
    to carry how many images each field was actually averaged over — otherwise
    a coverage number silently describes a subset.
    """
    true = square(8, 8, 8, h=32, w=32).bool()
    rows = [mask_detection(true, true), mask_detection(true, None)]
    agg = aggregate_detection(rows)
    assert agg["coverage"] == 1.0
    assert agg["coverage_n"] == 1
    assert agg["fp_area_frac_n"] == 2


# ── the coupling to scripts/, which nothing enforces at runtime ──────────────

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
    """
    THE ONE INVARIANT THIS LAYOUT CANNOT ENFORCE BY CONSTRUCTION.

    defences/ was kept self-contained so that scripts/ needs no --defence flag,
    and the price is that evaluate_defended.py REPEATS evaluate.py's per-image
    sequence: resolve_placement against the undefended clean prediction, then
    set_reference_from_image at the resolved placement, then apply.

    Get that order wrong and nothing raises. Semantic and gradcam placement read
    the clean prediction, and set_reference_from_image copies the region at the
    RESOLVED placement, so a swap silently evaluates a different patch — and the
    defended numbers would then be measured against an attack the undefended run
    never saw.

    This pins the two files against each other. It CANNOT notice a change made
    only in evaluate.py that leaves its own order self-consistent, which is why
    run_sac.sh also cross-checks the two undefended drop_remote values at
    runtime.
    """
    mine = _call_sequence(Path(__file__).with_name("evaluate_defended.py"))
    theirs = _call_sequence(ROOT_DIR / "scripts" / "evaluate.py")
    assert mine == list(SEQUENCE), mine
    assert theirs == list(SEQUENCE), theirs


def test_scripts_is_not_modified_to_know_about_defences():
    """
    The whole point of this folder. If a --defence flag ever appears in scripts/
    or patchreach/, this layout has stopped being self-contained and the claim
    "every existing run produces exactly the numbers it produced before" needs
    re-checking rather than assuming.
    """
    for sub in ("scripts", "patchreach"):
        for f in (ROOT_DIR / sub).rglob("*.py"):
            src = f.read_text(encoding="utf-8")
            assert "defence" not in src.lower(), f"{f} mentions a defence"
            assert "defenses" not in src, f"{f} mentions defenses"


def test_the_entry_point_only_borrows_read_only_plumbing():
    """
    Documents the coupling surface deliberately. These four helpers build a
    model and a dataset and resolve image indices; none of them is modified and
    none of them knows a defence exists. Anything else imported from _common
    widens the contract and should be a conscious decision.
    """
    src = (Path(__file__).with_name("evaluate_defended.py")
           .read_text(encoding="utf-8"))
    line = next(l for l in src.splitlines() if l.startswith("from _common"))
    assert line == ("from _common import add_model_args, setup_model, "
                    "make_dataset, image_indices"), line
