r"""
Cheap invariant tests. Run with: pytest -q

Each of these encodes a bug that actually cost a run in the previous codebase.
They take seconds; the bugs took hours to find.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from patchreach.patch.lap import logit_seed, ade, tv_loss
from patchreach.patch.shape import largest_component
from patchreach.patch.placement import find_semantic_placement


def test_logit_seed_roundtrip():
    """sigmoid(logit_seed(p)) == p. LAP and raw_ganinit init both depend on it."""
    p = torch.rand(3, 32, 32)
    assert torch.allclose(torch.sigmoid(logit_seed(p)), p, atol=1e-4)


def test_logit_seed_finite_at_bounds():
    """logit(0) = -inf saturates the sigmoid and kills the gradient."""
    assert torch.isfinite(logit_seed(torch.zeros(3, 8, 8))).all()
    assert torch.isfinite(logit_seed(torch.ones(3, 8, 8))).all()


def test_ade_extremes():
    """ADE: uniform -> 1.0 (max regularity), noise -> ~1/levels (spread)."""
    assert ade(torch.full((3, 32, 32), 0.5), levels=4) == pytest.approx(1.0, abs=1e-6)
    g = torch.rand(1, 32, 32).repeat(3, 1, 1)
    assert 0.1 < ade(g, levels=4) < 0.5


def test_tv_zero_on_constant():
    assert tv_loss(torch.full((3, 16, 16), 0.3)).item() == pytest.approx(0.0, abs=1e-4)


def test_largest_component_drops_specks():
    m = torch.zeros(32, 32, dtype=torch.bool)
    m[8:24, 8:24] = True          # object
    m[2, 2] = True                # speck
    m[29:31, 28:31] = True        # speck
    out = largest_component(m)
    assert out.sum() == 16 * 16
    assert out[8:24, 8:24].all()


def test_semantic_placement_finds_class():
    """Road in the lower half -> the window must land in the lower half."""
    pred = torch.zeros(40, 80, dtype=torch.long)
    pred[:18, :] = 2              # building on top
    top, _ = find_semantic_placement(pred, cls=0, p=12)
    assert top >= 18


def test_semantic_placement_absent_class_falls_back_to_centre():
    pred = torch.zeros(40, 80, dtype=torch.long)
    assert find_semantic_placement(pred, cls=17, p=12) == ((40 - 12) // 2,
                                                           (80 - 12) // 2)


# ═════════════════════════════════════════════════════════════════════════════
#  mIoU class set — the bug that made an inert attack look beneficial
# ═════════════════════════════════════════════════════════════════════════════

def test_union_class_set_lets_a_rare_class_swing_miou():
    """
    THE FAILURE THIS PINS. Under 'union' a class with no ground truth that the
    model nevertheless predicts scores IoU=0 and drags the nanmean down.
    Suppress that prediction and the class leaves the denominator entirely and
    the mean JUMPS — which made a 150-epoch run report drop_remote = -3.63,
    i.e. the attack apparently improving the model, on 0.86% of pixels flipped.
    """
    from patchreach.metrics.miou import SegMetric

    labels = torch.zeros(1, 100, 100, dtype=torch.long)
    labels[:, 50:] = 1                       # only classes 0 and 1 in GT

    clean = SegMetric(19)
    pred = labels.clone()
    pred[:, 0, :5] = 16                      # 5 spurious px of a GT-absent class
    clean.update(pred, labels)

    adv = SegMetric(19)
    adv.update(labels.clone(), labels)       # attack removed them; else perfect

    # union: the "attack" looks hugely beneficial
    assert clean.compute("union") < adv.compute("union") - 20
    # gt: the class is never counted in either pass, so nothing moves
    assert clean.compute("gt") == pytest.approx(adv.compute("gt"), abs=0.5)


def test_gt_class_set_is_identical_for_clean_and_adv():
    """The whole point: the denominator must not depend on the predictions."""
    from patchreach.metrics.miou import SegMetric
    labels = torch.randint(0, 4, (1, 64, 64))
    a, b = SegMetric(19), SegMetric(19)
    a.update(labels.clone(), labels)
    b.update(torch.randint(0, 19, (1, 64, 64)), labels)   # wildly different
    assert a.n_counted("gt") == b.n_counted("gt")
    assert a.n_counted("union") != b.n_counted("union")


def test_default_class_set_is_gt():
    from patchreach.metrics.miou import SegMetric
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    m = SegMetric(19)
    p = labels.clone(); p[:, 0, 0] = 5
    m.update(p, labels)
    assert m.compute() == pytest.approx(m.compute("gt"))
    assert m.compute("gt") != pytest.approx(m.compute("union"))
