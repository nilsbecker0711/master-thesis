r"""
The IAP port, pinned against a LITERAL TRANSCRIPTION of the reference.

WHY THIS FILE IS SHAPED THIS WAY. Every one of IAP's four parts is a short
arithmetic expression that can be wrong in a way that produces a plausible
patch and no error:

  the sensitivity map     a centred window instead of an anchored one, or the
                          biased std instead of the unbiased one, moves the map
                          by a pixel and changes its scale — and the map is
                          what gates the whole update.
  the window search       the reference maximises a SUM and this repository's
                          helper maximises a MEAN. They agree only because the
                          window size is constant, which is an argument, not a
                          fact, until it is tested.
  the regulariser         sum vs mean is a factor of p**2 * 3, which at 256px
                          is 196608. A run with the wrong one converges to the
                          clean image and reports a null result.
  the update rule         the three factors broadcast against each other in
                          four dimensions. Any one of them silently reducing
                          over the wrong axis still runs.

So the reference loops are transcribed here, verbatim in structure, and the
vectorised port is held equal to them. They are deliberately slow and
deliberately ugly: that is what makes them a reference rather than a second
implementation that could drift the same way.

Run with: pytest -q tests/test_iap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patchreach.losses import adversarial
from patchreach.metrics.miou import attack_rates
from patchreach.patch import iap
from patchreach.patch.placement import find_max_response_placement

K = 19


# ═════════════════════════════════════════════════════════════════════════════
#  the reference, transcribed from github.com/subratkishoredutta/IAP/src/utils.py
# ═════════════════════════════════════════════════════════════════════════════

def ref_get_std(input_tensor, subset_size_y, subset_size_x):
    """
    get_std_x (1x3) and get_std_y (3x1) in one body — they differ only in the
    window. Structure, loop bounds and the zero initialisation are the
    reference's; note that the window is anchored at (i, j) and that the loop
    stops short of the right/bottom edge, leaving those entries at zero.
    """
    image = input_tensor.clone()
    stride = 1
    stdten = torch.zeros(input_tensor.shape[2], input_tensor.shape[3])
    for i in range(0, image.size(2) - subset_size_y + 1, stride):
        for j in range(0, image.size(3) - subset_size_x + 1, stride):
            subset = image[0][:, i:i + subset_size_y, j:j + subset_size_x]
            std = (subset[0].std() + subset[1].std() + subset[2].std()) / 3
            stdten[i][j] = std
    return stdten


def ref_sens(image, eps=1e-3):
    """get_final_std() then `sens = 1/(stdimg+0.001)`."""
    stdx = ref_get_std(image, 1, 3)
    stdy = ref_get_std(image, 3, 1)
    return 1.0 / (torch.sqrt(torch.min(stdx, stdy)) + eps)


def ref_find_location(mask, window_size):
    """find_location(), which maximises the window SUM with a strict `>`."""
    height, width = mask.size()
    max_sum = 0
    max_sum_location = (0, 0)
    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            window = mask[i:i + window_size, j:j + window_size]
            window_sum = torch.sum(window).item()
            if window_sum > max_sum:
                max_sum = window_sum
                max_sum_location = (i, j)
    return max_sum_location


def ref_step(delta, grad, sens, lr):
    """
    The released update, as one expression:

        delta.data - learning_rate*delta.grad.mean(dim=1)*delta.data*(1/sens)

    delta is [1,3,p,p] here (the reference's layout) and sens is [p,p].
    """
    return delta - lr * grad.mean(dim=1) * delta * (1.0 / sens)


# ═════════════════════════════════════════════════════════════════════════════
#  1. the sensitivity map
# ═════════════════════════════════════════════════════════════════════════════

def test_sensitivity_matches_the_reference_loops():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 11, 14)
    assert torch.allclose(iap.sensitivity(x), ref_sens(x)[None], atol=1e-5)


def test_sensitivity_matches_on_near_constant_content():
    """
    THE CASE THE SUM-OF-SQUARES SHORTCUT GETS WRONG. A near-flat window makes
    S2 and S^2/n agree to six digits, so var = (S2 - S^2/n)/(n-1) is fp32
    noise there — and a flat region is exactly where sens is large and the
    whole invisibility mechanism lives. window_std() stacks the samples
    instead, so it stays exact.
    """
    x = torch.full((1, 3, 9, 9), 0.4)
    x[0, :, 4, 4] += 1e-4
    got, want = iap.sensitivity(x), ref_sens(x)[None]
    assert torch.allclose(got, want, rtol=1e-4)
    # and it must actually reach the ceiling, not merely agree on a wrong value
    assert float(got.max()) == pytest.approx(1.0 / iap.SENS_EPS, rel=1e-3)


def test_sensitivity_border_band_is_the_reference_artefact():
    """
    D7, pinned. The reference's loops leave a two-pixel band at the right and
    bottom at zero std, so sens = 1/eps = 1000 there and the update's 1/sens
    factor freezes those pixels. This is reproduced, not fixed: the published
    numbers were produced with it.
    """
    torch.manual_seed(1)
    s = iap.sensitivity(torch.rand(1, 3, 10, 12))[0]
    ceiling = 1.0 / iap.SENS_EPS
    assert torch.allclose(s[:, -2:], torch.full_like(s[:, -2:], ceiling))
    assert torch.allclose(s[-2:, :], torch.full_like(s[-2:, :], ceiling))
    assert float(s[:-2, :-2].max()) < ceiling


def test_window_std_is_the_unbiased_estimator():
    """n-1, matching torch's default and therefore the reference's `.std()`."""
    x = torch.tensor([[0.0, 0.5, 1.0]]).view(1, 1, 1, 3).expand(1, 3, 1, 3)
    got = float(iap.window_std(x, 1, 3)[0, 0, 0])
    assert got == pytest.approx(float(torch.tensor([0.0, 0.5, 1.0]).std()))
    assert got == pytest.approx(0.5)            # biased would be 0.408


def test_sensitivity_refuses_a_non_batched_image():
    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        iap.sensitivity(torch.rand(3, 8, 8))


# ═════════════════════════════════════════════════════════════════════════════
#  2. localisation
# ═════════════════════════════════════════════════════════════════════════════

def test_localise_matches_the_reference_window_search():
    """
    The reference maximises the window SUM; find_max_response_placement
    maximises the window MEAN. Equal argmax only because the window size is
    constant over the search — which is the claim this test exists to hold.
    """
    torch.manual_seed(2)
    for _ in range(5):
        cam = torch.rand(20, 28)
        sens = torch.rand(20, 28) + 0.5
        for p in (4, 7, 11):
            assert iap.localise(cam, sens, p, score="cam") == \
                ref_find_location(cam, p)


def test_localise_delegates_to_the_shared_placement_helper():
    """
    D1/D2 plumbing: IAP localisation and --placement gradcam must resolve the
    same window from the same map, or the per-image ablation and the generator
    ablation stop measuring the same thing.
    """
    torch.manual_seed(3)
    cam, sens = torch.rand(16, 24), torch.rand(16, 24) + 0.5
    assert iap.localise(cam, sens, 6, score="cam") == \
        find_max_response_placement(cam, 6, centre_if_constant=True)


def test_the_two_localisation_scores_disagree():
    """
    D1 is a real fork in the procedure, not a cosmetic one: the paper's J/Sens
    and the released code's J put the patch in different places. If this ever
    passes trivially the flag has stopped doing anything.
    """
    torch.manual_seed(4)
    cam = torch.rand(24, 24)
    sens = torch.rand(24, 24) * 50 + 1.0
    a = iap.localise(cam, sens, 6, score="cam")
    b = iap.localise(cam, sens, 6, score="cam_over_sens")
    assert a != b


def test_cam_over_sens_prefers_texture_at_equal_cam():
    """
    The mechanism, isolated: with a FLAT Grad-CAM the paper's score is decided
    by 1/sens alone, so it must land on the textured half (low sens) and not on
    the smooth one.
    """
    cam = torch.ones(16, 16)
    sens = torch.full((16, 16), 100.0)      # smooth: perturbation shows
    sens[:, 8:] = 1.0                       # textured: it hides
    top, left = iap.localise(cam, sens, 8, score="cam_over_sens")
    assert left == 8 and top == 0


def test_localise_centre_ablation_needs_no_cam():
    assert iap.localise(None, torch.rand(20, 30), 10, score="center") == (5, 10)


def test_localise_diverges_from_the_reference_on_a_degenerate_map():
    """
    DELIBERATE, and declared. The reference's `max_sum = 0` with a strict `>`
    returns the TOP-LEFT corner for an all-zero map, which reads as a placement
    and is not one; centre_if_constant=True falls back to the documented
    default position instead. Pinned so the divergence stays visible.
    """
    cam = torch.zeros(16, 16)
    assert ref_find_location(cam, 8) == (0, 0)
    assert iap.localise(cam, torch.ones(16, 16), 8, score="cam") == (4, 4)


def test_localise_rejects_an_unknown_score():
    with pytest.raises(ValueError, match="score must be one of"):
        iap.localise(torch.rand(8, 8), torch.rand(8, 8), 4, score="gradcam")


# ═════════════════════════════════════════════════════════════════════════════
#  3. the regulariser
# ═════════════════════════════════════════════════════════════════════════════

def test_regulariser_matches_the_reference_expression():
    torch.manual_seed(5)
    d, b = torch.rand(3, 9, 9), torch.rand(3, 9, 9)
    s = torch.rand(9, 9) * 20 + 1
    want = (s * torch.abs(d - b)).sum()
    assert float(iap.regulariser(d, b, s, "sum")) == pytest.approx(float(want))


def test_regulariser_sum_is_the_one_that_scales_with_area():
    """
    D3, pinned as arithmetic. 'sum' grows with p**2 and that is why w3 = 3 is
    not the same operating point at 256px that it was at 84px; 'sum84' undoes
    exactly that factor; 'mean' is free of it and is not IAP.
    """
    p = 42
    d = torch.full((3, p, p), 0.6)
    b = torch.full((3, p, p), 0.5)
    s = torch.ones(p, p)
    n = 3 * p * p
    assert float(iap.regulariser(d, b, s, "sum")) == pytest.approx(0.1 * n)
    assert float(iap.regulariser(d, b, s, "mean")) == pytest.approx(0.1)
    assert float(iap.regulariser(d, b, s, "sum84")) == pytest.approx(
        0.1 * n * (iap.REF_PATCH_SIDE / p) ** 2)


def test_regulariser_carries_a_gradient_to_delta_only():
    d = torch.rand(3, 8, 8, requires_grad=True)
    b = torch.rand(3, 8, 8)
    s = (torch.rand(8, 8) + 1).requires_grad_(True)
    iap.regulariser(d, b, s, "sum").backward()
    assert d.grad is not None and float(d.grad.abs().sum()) > 0
    # sens is a property of the CLEAN image; a gradient through it would be a
    # gradient through a constant that happens to be computed from a tensor.
    assert s.grad is None


def test_the_released_regulariser_has_no_gradient():
    r"""
    D9, AND THE LARGEST DISCREPANCY IN THE PORT. The reference builds its
    reference patch as

        delta = torch.nn.Parameter(delta)
        temp  = delta.clone()

    and clone() on a grad-requiring leaf is DIFFERENTIABLE, so autograd walks
    two paths into delta — `+sens*sign(d-t)` through delta and `-sens*sign(d-t)`
    through temp — which cancel exactly. The term is reported (it is the
    paper's "human distance") and optimises nothing.

    This test holds the finding on the REFERENCE'S OWN EXPRESSION, so it stays
    true independently of anything in iap.py and cannot be explained away as a
    porting artefact.
    """
    torch.manual_seed(11)
    base = torch.rand(1, 3, 8, 8)
    sens = torch.rand(8, 8) * 20 + 1

    delta = torch.nn.Parameter(base.clone())
    temp = delta.clone()                       # the reference, verbatim
    assert temp.requires_grad and temp.grad_fn is not None
    delta.data.add_(0.1)                       # so sign(d - t) = +1, not 0
    r = 3 * (sens.detach() * torch.abs(delta - temp)).sum()
    r.backward()

    assert float(r) > 0                        # the term is worth something
    assert float(delta.grad.abs().sum()) == 0.0    # and none of it is gradient


def test_the_ported_regulariser_does_have_a_gradient():
    """The other side of D9: a DETACHED base leaves the paper's gradient."""
    torch.manual_seed(11)
    base = torch.rand(3, 8, 8)
    sens = torch.rand(8, 8) * 20 + 1
    delta = (base + 0.1).clone().requires_grad_(True)
    (3 * iap.regulariser(delta, base, sens, "sum")).backward()
    assert torch.allclose(delta.grad, 3 * sens * torch.ones(3, 8, 8),
                          atol=1e-5)


def test_reg_grad_released_reports_the_value_but_not_the_gradient():
    """
    The reproduction must match the released code on BOTH counts: identical
    loss value, zero gradient. A mode that simply dropped the term would match
    the gradient and not the value, and the "human distance" column would stop
    being comparable between the two arms.
    """
    logits, label, fp = _scene()
    sup = ~fp
    # The adversarial terms need a graph of their own, or detaching the
    # regulariser leaves the total with no grad_fn and backward() raises. In
    # the real loop the patch IS upstream of the logits; here the two are
    # independent on purpose, so `d.grad` isolates the regulariser's path.
    logits = logits.clone().requires_grad_(True)
    b = torch.rand(3, 6, 6)
    s = torch.rand(6, 6) * 10 + 1

    val, total, grad = {}, {}, {}
    for mode in ("live", "released"):
        d = (b + 0.1).clone().requires_grad_(True)
        o = iap.Objective(tar="ipatch_cospgd", untar="ce", target_class=16,
                          reg_grad=mode)
        parts = o(logits, label, fp, sup, d, b, s)
        parts["total"].backward()
        val[mode], total[mode], grad[mode] = (parts["reg"],
                                              float(parts["total"]), d.grad)

    assert val["live"] == pytest.approx(val["released"])      # same value
    assert total["live"] == pytest.approx(total["released"])  # same total
    # live: the paper's gradient. released: NOTHING reached the patch.
    assert torch.allclose(grad["live"],
                          iap.IAP_W3 * s * torch.ones(3, 6, 6), atol=1e-4)
    assert grad["released"] is None


def test_reg_grad_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="reg_grad must be one of"):
        iap.Objective(reg_grad="detached")


def test_regulariser_rejects_an_unknown_reduction():
    with pytest.raises(ValueError, match="reduce must be one of"):
        iap.regulariser(torch.rand(3, 4, 4), torch.rand(3, 4, 4),
                        torch.ones(4, 4), "l2")


# ═════════════════════════════════════════════════════════════════════════════
#  4. the colour-constant update
# ═════════════════════════════════════════════════════════════════════════════

def _param_with_grad(p=7, seed=6):
    torch.manual_seed(seed)
    param = (torch.rand(3, p, p) * 0.8 + 0.1).requires_grad_(True)
    param.grad = torch.randn(3, p, p) * 0.01
    sens = torch.rand(p, p) * 30 + 1.0
    return param, sens


def test_step_matches_the_reference_expression():
    """
    The reference works on [1,3,p,p] and this repository's Patch parameter is
    [3,p,p], so the channel axis is 1 there and -3 here. A mean over the wrong
    axis still runs and still produces a patch.
    """
    param, sens = _param_with_grad()
    want = ref_step(param.detach()[None], param.grad[None], sens, 0.09)[0]
    iap.colour_constant_step_(param, sens, 0.09)
    assert torch.allclose(param.detach(), want, atol=1e-7)


def test_step_is_colour_constant():
    """
    THE PAPER'S TITLE CLAIM, as an invariant. Every channel of a pixel is
    scaled by the SAME relative amount, so the pixel moves along its own colour
    direction and its RGB ratios survive. Compare with Adam, which updates each
    channel independently and recolours the patch.
    """
    param, sens = _param_with_grad()
    before = param.detach().clone()
    iap.colour_constant_step_(param, sens, 0.05)
    ratio = param.detach() / before                 # step / value, per channel
    assert torch.allclose(ratio[0], ratio[1], atol=1e-6)
    assert torch.allclose(ratio[1], ratio[2], atol=1e-6)
    # and it is not a no-op, which would satisfy the above vacuously
    assert float((param.detach() - before).abs().max()) > 1e-6


def test_step_freezes_smooth_regions():
    """
    THE INVISIBILITY MECHANISM. 1/sens = sqrt(min std) + eps, so a perfectly
    flat neighbourhood (sens at its 1/eps ceiling) moves by ~eps times what a
    textured one moves, under the SAME gradient. Invisibility is not bought by
    penalising harder where it would show; it is bought by not listening to the
    attack there.
    """
    param = torch.full((3, 2, 2), 0.5).requires_grad_(True)
    param.grad = torch.ones(3, 2, 2)
    sens = torch.tensor([[1.0 / iap.SENS_EPS, 1.0 / iap.SENS_EPS],
                         [2.0, 2.0]])
    before = param.detach().clone()
    iap.colour_constant_step_(param, sens, 0.1)
    moved = (before - param.detach())
    smooth, textured = float(moved[0, 0, 0]), float(moved[0, 1, 0])
    assert textured / smooth == pytest.approx(0.5 / iap.SENS_EPS, rel=1e-3)


def test_step_leaves_a_zero_pixel_frozen():
    """
    The documented corner of the multiplicative update: `* param` makes 0 an
    absorbing state. It is why the parameter must be initialised from image
    content and never from zeros.
    """
    param = torch.zeros(3, 2, 2).requires_grad_(True)
    param.grad = torch.ones(3, 2, 2)
    iap.colour_constant_step_(param, torch.ones(2, 2), 0.2)
    assert float(param.detach().abs().max()) == 0.0


def test_step_refuses_a_missing_gradient():
    param = torch.rand(3, 4, 4).requires_grad_(True)
    with pytest.raises(RuntimeError, match="grad = None"):
        iap.colour_constant_step_(param, torch.ones(4, 4), 0.1)


def test_step_refuses_a_non_finite_gradient():
    param = torch.rand(3, 4, 4).requires_grad_(True)
    param.grad = torch.full((3, 4, 4), float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        iap.colour_constant_step_(param, torch.ones(4, 4), 0.1)


# ═════════════════════════════════════════════════════════════════════════════
#  the loss
# ═════════════════════════════════════════════════════════════════════════════

def _scene(h=8, w=12, seed=7):
    torch.manual_seed(seed)
    logits = torch.randn(1, K, h, w)
    label = torch.randint(0, K, (1, h, w))
    fp = torch.zeros(1, h, w, dtype=torch.bool)
    fp[:, :2, :2] = True
    return logits, label, fp


def test_objective_decomposes_exactly_into_its_three_terms():
    """
    The port is a SUBSTITUTION, so the total has to equal the weighted sum of
    this repository's own losses plus the regulariser — with no sign flip
    anywhere, which is the claim Objective's docstring makes.
    """
    logits, label, fp = _scene()
    sup = ~fp
    d = torch.rand(3, 5, 5)
    b = torch.rand(3, 5, 5)
    s = torch.rand(5, 5) + 1
    obj = iap.Objective(tar="ipatch_cospgd", untar="ce", target_class=16,
                        w1=1.5, w2=1.0, w3=3.0)
    out = obj(logits, label, fp, sup, d, b, s)

    want = (1.5 * adversarial.ipatch_cospgd_loss(logits, 16, fp, sup)
            + 1.0 * adversarial.ce_loss(logits, label, sup)
            + 3.0 * iap.regulariser(d, b, s, "sum"))
    assert float(out["total"]) == pytest.approx(float(want), rel=1e-6)
    assert out["tar"] == pytest.approx(
        float(adversarial.ipatch_cospgd_loss(logits, 16, fp, sup)), rel=1e-6)
    assert out["untar"] == pytest.approx(
        float(adversarial.ce_loss(logits, label, sup)), rel=1e-6)


def test_objective_untargeted_term_is_the_reference_minus_ce():
    """
    IAP's second term is `- criterion(pred, class_id)` — an explicit minus on a
    cross-entropy. adversarial.ce_loss already returns -CE, so the port must
    ADD it with a positive weight. A second negation would make the attack
    optimise toward the ground truth and still converge.
    """
    logits, label, fp = _scene()
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, label, reduction="none", ignore_index=255)
    assert float(adversarial.ce_loss(logits, label, None)) == pytest.approx(
        -float(ce.mean()), rel=1e-6)


def test_objective_drops_a_term_on_none():
    logits, label, fp = _scene()
    d = b = torch.zeros(3, 4, 4)
    s = torch.ones(4, 4)
    out = iap.Objective(tar="none", untar="ce")(logits, label, fp, ~fp,
                                                d, b, s)
    assert "tar" not in out and "untar" in out


def test_objective_refuses_to_drop_both_adversarial_terms():
    with pytest.raises(ValueError, match="regulariser alone"):
        iap.Objective(tar="none", untar="none")


def test_objective_refuses_a_margin_loss_without_the_clean_prediction():
    with pytest.raises(ValueError, match="CLEAN PREDICTION"):
        iap.Objective(untar="margin")


# ═════════════════════════════════════════════════════════════════════════════
#  the early stop
# ═════════════════════════════════════════════════════════════════════════════

def _onehot(pred, k=K):
    lg = torch.zeros(1, k, *pred.shape[-2:])
    lg.scatter_(1, pred.unsqueeze(1), 8.0)
    return lg


def test_progress_rate_is_the_reported_metric_divided_by_100():
    """
    The stop criterion must be a quantity that appears in the results table, or
    a run that stopped early stopped on something nobody can check.
    """
    label = torch.zeros(1, 6, 6, dtype=torch.long)
    pc = torch.zeros(1, 6, 6, dtype=torch.long)
    pa = torch.zeros(1, 6, 6, dtype=torch.long)
    pa[:, :3, :] = 16                      # half the frame driven to class 16
    fp = torch.zeros(1, 6, 6, dtype=torch.bool)
    lc, la = _onehot(pc), _onehot(pa)

    rates = attack_rates(lc, la, label, fp, 16)
    assert iap.progress_rate(lc, la, label, fp, "flip") == pytest.approx(
        rates["any_flip_rate"] / 100.0)
    assert iap.progress_rate(lc, la, label, fp, "target",
                             target_class=16) == pytest.approx(
        rates["target_hit_rate"] / 100.0)
    assert iap.progress_rate(lc, la, label, fp, "flip") == pytest.approx(0.5)


def test_progress_rate_none_never_stops():
    label = torch.zeros(1, 4, 4, dtype=torch.long)
    fp = torch.zeros(1, 4, 4, dtype=torch.bool)
    lg = _onehot(label)
    assert iap.progress_rate(lg, lg, label, fp, "none") == 0.0


def test_progress_rate_target_needs_a_target_class():
    label = torch.zeros(1, 4, 4, dtype=torch.long)
    fp = torch.zeros(1, 4, 4, dtype=torch.bool)
    lg = _onehot(label)
    with pytest.raises(ValueError, match="needs a target_class"):
        iap.progress_rate(lg, lg, label, fp, "target")


def test_progress_rate_drop_needs_the_clean_baseline():
    label = torch.zeros(1, 4, 4, dtype=torch.long)
    fp = torch.zeros(1, 4, 4, dtype=torch.bool)
    lg = _onehot(label)
    with pytest.raises(ValueError, match="clean remote mIoU"):
        iap.progress_rate(lg, lg, label, fp, "drop")


# ═════════════════════════════════════════════════════════════════════════════
#  the refusals that keep a run from silently being a different procedure
# ═════════════════════════════════════════════════════════════════════════════

def _cfg(**kw):
    from patchreach.patch.spec import PatchConfig
    base = dict(mode="raw", pixel_param="direct", size=16, scale=0.25,
                placement="center")
    base.update(kw)
    return PatchConfig(**base)


def _patch(cfg):
    from patchreach.data.cityscapes import norm_tensors
    from patchreach.patch.spec import Patch
    dev = torch.device("cpu")
    m, s = norm_tensors(dev)
    return Patch(cfg, dev, m, s)


def test_prepare_refuses_a_resampled_patch():
    """
    THE PASTE-SIDE RULE, as a refusal. side(H,W) = int(H*scale) and the
    parameter is size x size; when they differ, Patch.apply interpolates, and
    three of IAP's four parts are then computed on a signal that is not the one
    pasted. Both checks run before the model is touched, which is why this test
    needs no segmentor.
    """
    patch = _patch(_cfg(size=16, scale=0.25))         # 16 vs int(64*0.25)=16
    img = torch.zeros(1, 3, 100, 100)                 # int(100*0.25) = 25
    with pytest.raises(ValueError, match="parameter resolution"):
        iap.prepare(None, img, patch, mean_t=0.0, std_t=1.0,
                    label=torch.zeros(1, 100, 100, dtype=torch.long), cam=None)


def test_prepare_refuses_a_sigmoid_parameterisation():
    """
    Not a preference. The update multiplies by `delta`; under a sigmoid
    parameterisation that factor is a LOGIT, negative for any pixel below 0.5,
    so the step reverses sign for half the patch and the run still completes.
    """
    patch = _patch(_cfg(size=16, scale=0.25, pixel_param="sigmoid"))
    img = torch.zeros(1, 3, 64, 64)
    with pytest.raises(ValueError, match="pixel_param direct"):
        iap.prepare(None, img, patch, mean_t=0.0, std_t=1.0,
                    label=torch.zeros(1, 64, 64, dtype=torch.long), cam=None)


def test_prepare_refuses_to_centre_silently_without_a_cam():
    patch = _patch(_cfg(size=16, scale=0.25))
    img = torch.zeros(1, 3, 64, 64)
    with pytest.raises(ValueError, match="needs a SegmentationCAM"):
        iap.prepare(None, img, patch, mean_t=0.0, std_t=1.0,
                    label=torch.zeros(1, 64, 64, dtype=torch.long), cam=None,
                    score="cam")


def test_attack_image_rejects_an_empty_ladder():
    patch = _patch(_cfg(size=16, scale=0.25))
    with pytest.raises(ValueError, match="ladder is empty"):
        iap.attack_image(None, torch.zeros(1, 3, 64, 64),
                         torch.zeros(1, 64, 64, dtype=torch.long), patch,
                         mean_t=0.0, std_t=1.0, cam=None, lrs=[])


# ═════════════════════════════════════════════════════════════════════════════
#  invisibility
# ═════════════════════════════════════════════════════════════════════════════

def test_invisibility_separates_footprint_from_frame():
    """
    BOTH REGIONS OR NEITHER. These metrics are spatial means, so a patch
    covering 3% of the frame is diluted ~32x in the global figure: a change
    that is obvious inside the footprint still scores as near-perfect
    globally. The paper reports local and global for every metric for exactly
    this reason, and a table quoting only the global column would say an attack
    is invisible when it is not.
    """
    pytest.importorskip("torchmetrics")
    torch.manual_seed(8)
    clean = torch.rand(1, 3, 64, 64)
    adv = clean.clone()
    adv[:, :, 8:24, 8:24] = torch.rand(1, 3, 16, 16)      # 6.25% of the frame
    v = iap.invisibility(clean, adv, (8, 8, 16))
    assert v["ssim_patch"] < v["ssim_image"]
    assert v["psnr_patch"] < v["psnr_image"]


def test_invisibility_is_perfect_on_an_unchanged_frame():
    pytest.importorskip("torchmetrics")
    clean = torch.rand(1, 3, 32, 32)
    v = iap.invisibility(clean, clean.clone(), (4, 4, 16))
    assert v["ssim_patch"] == pytest.approx(1.0, abs=1e-5)
    assert v["ssim_image"] == pytest.approx(1.0, abs=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
#  slide inference — what IAP can and cannot do under a tiled model
# ═════════════════════════════════════════════════════════════════════════════

def _stub_segmentor():
    """
    An mmseg-shaped segmentor: a .backbone returning a TUPLE of feature maps,
    and encode_decode(x, metas). Enough for WrappedSegModel and
    SlidingWindowSegModel to wrap, and for the CAM to try to hook.
    """
    import torch.nn as nn

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 16, 3, stride=4, padding=1)
            self.n_calls = 0

        def forward(self, x):
            self.n_calls += 1
            return (self.c(x),)

    class Seg(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.head = nn.Conv2d(16, K, 1)
            self.test_cfg = {"mode": "slide", "crop_size": (64, 64),
                             "stride": (48, 48)}

        def encode_decode(self, x, metas):
            return self.head(self.backbone(x)[0])

    return Seg()


def test_the_cam_refuses_a_tiled_model():
    r"""
    THE FAILURE THIS REPLACES was not a refusal but a crash three frames deep.
    SlidingWindowSegModel calls the segmentor once per window, so the CAM's
    forward hook fires n_windows times and keeps only the LAST window's feature
    map, while the gradient and the upsample target are frame-shaped. Measured
    on this stub before the guard: module='backbone' raised AttributeError (the
    wrapper shifts the dotted path), and module='model.backbone' resolved and
    then died with "The size of tensor a (128) must match the size of tensor b
    (80)".

    This also covers overfit.py's `--placement gradcam --inference auto`, which
    had the same latent bug and no flag refusing it.
    """
    from patchreach.models.wrapper import (SlidingWindowSegModel,
                                           WrappedSegModel)
    from patchreach.patch import segmentation_cam

    whole = WrappedSegModel(_stub_segmentor())
    tiled = SlidingWindowSegModel(whole, (64, 64), (48, 48), K)

    with pytest.raises(ValueError, match="cannot run on a SlidingWindowSegModel"):
        segmentation_cam.build(tiled, "ce")

    # ... and the un-tiled model is still fine, so the guard is not a blanket.
    cam = segmentation_cam.build(whole, "ce")
    m, logits = cam(torch.rand(1, 3, 64, 128),
                    torch.randint(0, K, (1, 64, 128)))
    assert m.shape == (1, 1, 64, 128)
    cam.close()


def test_slide_checkpoint_is_refused_only_when_a_cam_is_built():
    """
    The first version of this guard refused --slide_checkpoint outright, which
    blocked the ONE combination that works at native resolution: centre
    placement builds no CAM, so nothing calls autograd.grad and the reentrant
    checkpoint is free to do its job. A guard that over-refuses is as wrong as
    one that under-refuses — it sends the run to a mode the baseline is not in.
    """
    _scripts_on_path()
    import iap_attack as S

    p = S.build_parser()
    base = ["--cityscapes_root", "X", "--slide_checkpoint"]

    with pytest.raises(SystemExit):
        S.check_args(p.parse_args(base + ["--iap_locate", "cam"]), p)

    a = S.check_args(p.parse_args(base + ["--iap_locate", "center"]), p)
    assert a.slide_checkpoint and a.iap_locate == "center"


def _scripts_on_path():
    import sys
    from pathlib import Path
    d = str(Path(__file__).resolve().parents[1] / "scripts")
    if d not in sys.path:
        sys.path.insert(0, d)
