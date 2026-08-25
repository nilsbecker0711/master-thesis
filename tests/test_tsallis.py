r"""
Tsallis cross-entropy attack objective. Run with: pytest -q tests/test_tsallis.py

test_existing_losses_unchanged is the guard on the whole integration: the
golden numbers in it were captured from ce_loss/cospgd_loss/ipatch_cospgd_loss
BEFORE any edit was made, so a regression in the shared _reduce() or in the
build() dispatch shows up here rather than three weeks later in a results CSV.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from patchreach.losses.adversarial import (build, ce_loss, cospgd_loss,
                                           ipatch_cospgd_loss)
from patchreach.losses.tsallis import (CE_LIMIT_EPS, TsallisCELoss, schedule_q,
                                       tsallis_from_log_py, tsallis_per_pixel,
                                       validate_q)

QS = (-3.0, -2.0, -1.0, 0.0, 0.5)


def _batch(K=19, B=2, H=12, W=20, seed=1234, dtype=torch.float32):
    """Fixed-seed synthetic batch with void pixels and a support mask."""
    torch.manual_seed(seed)
    logits = torch.randn(B, K, H, W, dtype=torch.float64).to(dtype)
    labels = torch.randint(0, K, (B, H, W))
    labels[0, :3, :] = 255
    labels[1, 5, 7] = 255
    support = torch.zeros(B, H, W, dtype=torch.bool)
    support[:, 2:, 1:] = True
    return logits, labels, support


# ═════════════════════════════════════════════════════════════════════════════
#  1. the CE limit
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("K", [2, 5, 19, 40])
def test_ce_limit(K):
    """q -> 1 IS cross-entropy: same reduction, same ignore mask, same value."""
    loss = TsallisCELoss(q=1.0 - 1e-7)
    for support in (None, _batch(K)[2]):
        logits, labels, _ = _batch(K)
        ref = ce_loss(logits, labels, support)
        got = loss(logits, labels, None, support)
        assert got.item() == pytest.approx(ref.item(), abs=1e-5)


def test_ce_limit_gradients_match_and_point_the_same_way():
    """
    The sign check Step 1 asks for: one Adam step from Tsallis at q -> 1 must
    move the parameter the SAME direction as one step from the existing ce.
    """
    logits, labels, support = _batch()

    a = logits.clone().requires_grad_(True)
    ce_loss(a, labels, support).backward()

    b = logits.clone().requires_grad_(True)
    TsallisCELoss(q=1.0 - 1e-7)(b, labels, None, support).backward()

    assert torch.allclose(a.grad, b.grad, atol=1e-6)
    # not merely close in norm — every coordinate agrees in sign
    nz = a.grad.abs() > 1e-9
    assert torch.equal(torch.sign(a.grad[nz]), torch.sign(b.grad[nz]))


def test_closed_form_agrees_with_ce_just_outside_the_fallback():
    """
    The fallback makes q = 1 - 1e-7 exact by construction. This checks the
    expm1 branch itself converges to CE, at a q the fallback does NOT catch.
    """
    logits, labels, support = _batch(dtype=torch.float64)
    q = 1.0 - 1e-4
    assert abs(1.0 - q) > CE_LIMIT_EPS            # the closed form really runs
    got = TsallisCELoss(q=q)(logits, labels, None, support)
    ref = ce_loss(logits, labels, support)
    assert got.item() == pytest.approx(ref.item(), abs=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
#  2. the gradient re-weighting — the mechanism the whole method rests on
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("p_val", [0.05, 0.2, 0.5, 0.8, 0.99])
def test_gradient_reweighting(q, p_val):
    """dL_q/dp == p**(1-q) * dL_CE/dp, by autograd on a scalar p."""
    p = torch.tensor(p_val, dtype=torch.float64, requires_grad=True)
    tsallis_from_log_py(torch.log(p), q).backward()
    d_q = p.grad.clone()

    p2 = torch.tensor(p_val, dtype=torch.float64, requires_grad=True)
    (-torch.log(p2)).backward()
    d_ce = p2.grad.clone()

    assert d_q.item() == pytest.approx((p_val ** (1.0 - q)) * d_ce.item(),
                                       rel=1e-10)


@pytest.mark.parametrize("q,expected", [(0.0, 0.5), (-1.0, 2.0 / 3.0),
                                        (-2.0, 0.75), (-3.0, 0.8)])
def test_gradient_peak(q, expected):
    """
    Grid-search the gradient-norm lower-bound proxy p^(2(1-q)) (1-p)^2 and
    confirm its argmax is p* = (1-q)/(2-q).
    """
    p = torch.linspace(1e-4, 1.0 - 1e-4, 200001, dtype=torch.float64)
    proxy = p ** (2.0 * (1.0 - q)) * (1.0 - p) ** 2
    p_star = float(p[int(proxy.argmax())])
    assert p_star == pytest.approx((1.0 - q) / (2.0 - q), abs=1e-4)
    assert p_star == pytest.approx(expected, abs=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
#  3. masking — 255 must be inert, not merely small
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("q", QS)
def test_ignore_index(q):
    """
    Void pixels contribute EXACTLY zero gradient, and the loss equals the loss
    of the masked subset computed on its own.
    """
    logits, labels, _ = _batch()
    void = labels == 255
    assert void.any()

    a = logits.clone().requires_grad_(True)
    val = TsallisCELoss(q=q)(a, labels, None, None)
    val.backward()
    # [B,K,H,W] gradient, [B,H,W] mask -> broadcast over the class axis
    assert torch.count_nonzero(a.grad[void.unsqueeze(1).expand_as(a.grad)]) == 0

    # same number from the valid subset alone
    per_pixel = tsallis_per_pixel(logits, labels, q)
    manual = -per_pixel[~void].mean()
    assert val.item() == pytest.approx(manual.item(), rel=1e-6)


@pytest.mark.parametrize("q", QS)
def test_support_mask_is_the_ce_masks_intersection(q):
    """
    The spatial mask (reach restriction, footprint exclusion) composes exactly
    as it does for ce_loss — one shared expression, not a second copy.
    """
    logits, labels, support = _batch()
    val = TsallisCELoss(q=q)(logits, labels, None, support)
    valid = (labels != 255) & support
    manual = -tsallis_per_pixel(logits, labels, q)[valid].mean()
    assert val.item() == pytest.approx(manual.item(), rel=1e-6)


def test_footprint_argument_is_accepted_and_ignored():
    """Untargeted, so it takes the 4-arg convention but scores no differently."""
    logits, labels, support = _batch()
    fp = torch.zeros_like(labels, dtype=torch.bool)
    fp[:, 4:8, 4:10] = True
    loss = TsallisCELoss(q=0.0)
    assert loss(logits, labels, fp, support).item() == pytest.approx(
        loss(logits, labels, None, support).item())


# ═════════════════════════════════════════════════════════════════════════════
#  4. numerical stability
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("q", list(QS) + [1.0 - CE_LIMIT_EPS,
                                          1.0 - CE_LIMIT_EPS / 2.0, 1.0])
def test_numerical_stability(q):
    """
    Logits scaled to +/-50. p_y bottoms out around 1e-22 there, which is enough
    that the naive route p_y**(1-q) flushes to EXACTLY 0 at q = -3 — the
    gradient weight is destroyed before it is applied. Loss and gradients must
    be finite for every q regardless.
    """
    logits, labels, support = _batch()
    logits = (logits / logits.abs().max() * 50.0).requires_grad_(True)

    p_y = F.softmax(logits.detach(), 1).gather(
        1, labels.clamp(max=18).unsqueeze(1)).squeeze(1)
    assert (p_y ** 4 == 0).any(), "the naive route does not actually underflow"

    val = TsallisCELoss(q=q)(logits, labels, None, support)
    val.backward()
    assert torch.isfinite(val), f"non-finite loss at q={q}"
    assert torch.isfinite(logits.grad).all(), f"non-finite gradient at q={q}"


@pytest.mark.parametrize("q", list(QS) + [1.0 - CE_LIMIT_EPS / 2.0])
def test_survives_probabilities_that_underflow_to_exactly_zero(q):
    """
    The hard case: p_y == 0 in float32, so log(softmax(.)) is -inf and any
    probability-space implementation returns inf or nan. log_softmax keeps a
    finite ~-123 and L_q lands on its true supremum 1/(1-q).
    """
    K = 19
    logits = torch.full((1, K, 4, 4), 60.0)
    logits[:, 0] = -60.0                      # ground-truth class, 120 below
    labels = torch.zeros(1, 4, 4, dtype=torch.long)

    p_y = F.softmax(logits, 1).gather(1, labels.unsqueeze(1)).squeeze(1)
    assert (p_y == 0).all(), "p_y did not underflow to exactly zero"
    assert torch.isinf(torch.log(p_y)).all(), "naive log(p) is not -inf"

    logits = logits.requires_grad_(True)
    val = TsallisCELoss(q=q)(logits, labels, None, None)
    val.backward()
    assert torch.isfinite(val)
    assert torch.isfinite(logits.grad).all()
    if abs(1.0 - q) >= CE_LIMIT_EPS:
        assert -val.item() == pytest.approx(1.0 / (1.0 - q), rel=1e-5)


def test_fallback_boundary_is_continuous():
    """No step change in the loss as q crosses the CE-fallback threshold."""
    logits, labels, support = _batch(dtype=torch.float64)
    below = TsallisCELoss(q=1.0 - CE_LIMIT_EPS / 2.0)(
        logits, labels, None, support).item()
    above = TsallisCELoss(q=1.0 - CE_LIMIT_EPS * 2.0)(
        logits, labels, None, support).item()
    assert below == pytest.approx(above, abs=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
#  5. the q schedule
# ═════════════════════════════════════════════════════════════════════════════

def test_schedule_linear_hits_both_endpoints_and_is_monotone():
    T = 50
    loss = TsallisCELoss(schedule="linear", q_start=-2.0, q_end=1.0,
                         total_steps=T)
    seen = []
    for s in range(T):
        loss.on_step_begin(s, T)
        seen.append(loss.q)
    assert seen[0] == pytest.approx(-2.0)
    assert seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[T // 2] == pytest.approx(-2.0 + 3.0 * (T // 2) / (T - 1))


def test_schedule_const_is_invariant():
    loss = TsallisCELoss(q=-1.5, schedule="const", total_steps=10)
    for s in range(10):
        loss.on_step_begin(s, 10)
        assert loss.q == pytest.approx(-1.5)


def test_schedule_single_step_does_not_divide_by_zero():
    assert schedule_q(0, 1, schedule="linear", q_start=-2.0,
                      q_end=1.0) == pytest.approx(-2.0)
    loss = TsallisCELoss(schedule="linear", total_steps=1)
    loss.on_step_begin(0, 1)
    assert loss.q == pytest.approx(-2.0)


def test_schedule_clamps_rather_than_overshooting_q_end():
    """
    A 1-based loop, or one that overruns its announced length, must not walk
    past q_end — at q_end = 1 that would enter the rejected q > 1 regime.
    """
    assert schedule_q(99, 10, schedule="linear", q_start=-2.0,
                      q_end=1.0) == pytest.approx(1.0)
    assert schedule_q(-5, 10, schedule="linear", q_start=-2.0,
                      q_end=1.0) == pytest.approx(-2.0)


def test_q_never_advances_without_the_hook():
    """A loop that does not call on_step_begin leaves q pinned at q_start."""
    loss = TsallisCELoss(schedule="linear", q_start=-2.0, q_end=1.0,
                         total_steps=100)
    assert loss.q == pytest.approx(-2.0)
    assert loss.step == 0


def test_schedule_changes_the_loss_value():
    """The schedule is wired to the loss, not merely to a logged number."""
    logits, labels, support = _batch()
    loss = TsallisCELoss(schedule="linear", q_start=-3.0, q_end=0.0,
                         total_steps=10)
    loss.on_step_begin(0, 10)
    first = loss(logits, labels, None, support).item()
    loss.on_step_begin(9, 10)
    last = loss(logits, labels, None, support).item()
    assert first != pytest.approx(last)


# ═════════════════════════════════════════════════════════════════════════════
#  6. q > 1 is rejected
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("q", [1.0 + 1e-6, 1.5, 2.0, 10.0])
def test_rejects_q_above_one(q):
    with pytest.raises(ValueError, match="q <= 1"):
        TsallisCELoss(q=q)
    with pytest.raises(ValueError, match="q <= 1"):
        TsallisCELoss(schedule="linear", q_start=q)
    with pytest.raises(ValueError, match="q <= 1"):
        TsallisCELoss(schedule="linear", q_end=q)


def test_q_exactly_one_is_allowed():
    """q = 1 is the CE limit, which is in the family and is q_end's default."""
    assert validate_q(1.0)
    TsallisCELoss(q=1.0)
    TsallisCELoss(schedule="linear", q_start=-2.0, q_end=1.0)


def test_inert_q_is_not_rejected():
    """q_start/q_end are ignored under const; q is ignored under linear."""
    TsallisCELoss(q=0.0, schedule="const", q_start=5.0, q_end=5.0)
    TsallisCELoss(q=5.0, schedule="linear", q_start=-2.0, q_end=1.0)


def test_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="schedule"):
        TsallisCELoss(schedule="cosine")


def test_patch_config_rejects_q_above_one_at_config_time():
    """The CLI pathway must fail before the first forward pass, not after."""
    pytest.importorskip("torchvision")
    from patchreach.patch.spec import PatchConfig
    with pytest.raises(ValueError, match="q <= 1"):
        PatchConfig(tsallis_q=2.0).validate()
    PatchConfig(tsallis_q=0.0).validate()               # the default is fine


# ═════════════════════════════════════════════════════════════════════════════
#  7. dispatch and config reachability
# ═════════════════════════════════════════════════════════════════════════════

def test_build_returns_a_stateful_callable():
    obj = build("tsallis", 8, tsallis_q=-1.0, tsallis_schedule="linear",
                tsallis_q_start=-2.0, tsallis_q_end=1.0,
                tsallis_total_steps=20)
    assert hasattr(obj, "on_step_begin")
    logits, labels, support = _batch()
    assert torch.isfinite(obj(logits, labels, None, support))


def test_existing_losses_do_not_grow_a_step_hook():
    """
    The hasattr guard in the loops is only a proof of non-interference if the
    pre-existing objectives really have no on_step_begin.
    """
    for name in ("ce", "cospgd", "ipatch_cospgd"):
        assert not hasattr(build(name), "on_step_begin")


def test_tsallis_fields_are_reachable_from_the_cli():
    """
    PatchConfig fields with no argparse pathway are dead config. This walks the
    real parser rather than trusting the mapping by eye — the repo has shipped
    a missing pathway before.
    """
    pytest.importorskip("torchvision")
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from _common import add_patch_args

    p = add_patch_args(argparse.ArgumentParser())
    a = p.parse_args(["--tsallis_q", "-1.5", "--tsallis_schedule", "linear",
                      "--tsallis_q_start", "-3", "--tsallis_q_end", "0.5"])
    assert (a.tsallis_q, a.tsallis_schedule, a.tsallis_q_start,
            a.tsallis_q_end) == (-1.5, "linear", -3.0, 0.5)

    from patchreach.patch.spec import PatchConfig
    cfg = PatchConfig(tsallis_q=a.tsallis_q,
                      tsallis_schedule=a.tsallis_schedule,
                      tsallis_q_start=a.tsallis_q_start,
                      tsallis_q_end=a.tsallis_q_end).validate()
    assert cfg.tsallis_schedule == "linear"


def test_tsallis_defaults_match_between_cli_and_config():
    pytest.importorskip("torchvision")
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from _common import add_patch_args
    from patchreach.patch.spec import PatchConfig

    a = add_patch_args(argparse.ArgumentParser()).parse_args([])
    d = PatchConfig()
    for k in ("tsallis_q", "tsallis_schedule", "tsallis_q_start",
              "tsallis_q_end"):
        assert getattr(a, k) == getattr(d, k), k


# ═════════════════════════════════════════════════════════════════════════════
#  8. THE GUARD — golden values captured before any edit was made
# ═════════════════════════════════════════════════════════════════════════════

# ce_loss / cospgd_loss / ipatch_cospgd_loss on _batch() at seed 1234, float32
# CPU. Captured on the pre-change tree; any drift in _reduce(), in the masking
# expression, or in build()'s dispatch breaks this and nothing else has to.
GOLDEN = {
    ("ce", False): (-3.3223705291748047, 0.048807259649038315),
    ("ce", True): (-3.3132686614990234, 0.052603162825107574),
    ("cospgd", False): (-0.43564239144325256, 0.010645800270140171),
    ("cospgd", True): (-0.4388832449913025, 0.011609832756221294),
}


@pytest.mark.parametrize("name,use_support", sorted(GOLDEN))
def test_existing_losses_unchanged(name, use_support):
    fn = {"ce": ce_loss, "cospgd": cospgd_loss}[name]
    logits, labels, support = _batch()
    logits = logits.requires_grad_(True)
    val = fn(logits, labels, support if use_support else None)
    val.backward()

    want_v, want_g = GOLDEN[(name, use_support)]
    assert val.item() == pytest.approx(want_v, rel=1e-6)
    assert logits.grad.norm().item() == pytest.approx(want_g, rel=1e-6)


def test_existing_ipatch_unchanged():
    logits, labels, _ = _batch()
    logits = logits.requires_grad_(True)
    fp = torch.zeros(2, 12, 20, dtype=torch.bool)
    fp[:, 4:8, 4:10] = True
    val = ipatch_cospgd_loss(logits, 8, fp, None)
    val.backward()
    assert val.item() == pytest.approx(2.935673952102661, rel=1e-6)
    assert logits.grad.norm().item() == pytest.approx(0.0417482852935791,
                                                      rel=1e-6)


@pytest.mark.parametrize("name", ["ce", "cospgd", "ipatch_cospgd"])
def test_build_still_returns_the_same_numbers(name):
    """The dispatch gained a branch; the branches it already had did not move."""
    logits, labels, support = _batch()
    fp = torch.zeros_like(labels, dtype=torch.bool)
    fp[:, 4:8, 4:10] = True
    direct = {"ce": lambda: ce_loss(logits, labels, support),
              "cospgd": lambda: cospgd_loss(logits, labels, support),
              "ipatch_cospgd": lambda: ipatch_cospgd_loss(logits, 8, fp,
                                                          support)}[name]()
    assert build(name, 8)(logits, labels, fp, support).item() == \
        pytest.approx(direct.item(), rel=1e-9)


# ═════════════════════════════════════════════════════════════════════════════
#  9. every entry point — the wiring, not the maths
# ═════════════════════════════════════════════════════════════════════════════

TSALLIS_FLAGS = ("--tsallis_q", "--tsallis_schedule", "--tsallis_q_start",
                 "--tsallis_q_end")


def _scripts_on_path():
    import sys
    from pathlib import Path
    d = str(Path(__file__).resolve().parents[1] / "scripts")
    if d not in sys.path:
        sys.path.insert(0, d)


def _option_strings(parser):
    return {s for a in parser._actions for s in a.option_strings}


@pytest.mark.parametrize("group", ["add_patch_args", "add_generator_args"])
def test_every_argument_group_exposes_tsallis(group):
    """
    add_patch_args feeds train.py / overfit.py / overfit_population.py;
    add_generator_args feeds train_conditional_generator.py. Between them they
    cover every parser that builds an attack objective.
    """
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import argparse
    import _common
    opts = _option_strings(getattr(_common, group)(argparse.ArgumentParser()))
    for f in TSALLIS_FLAGS:
        assert f in opts, f"{group} is missing {f}"


@pytest.mark.parametrize("module", ["train", "train_conditional_generator",
                                    "overfit_population"])
def test_training_parsers_accept_tsallis(module):
    """--loss_fn tsallis must be a legal choice, not just a legal string."""
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import importlib
    p = importlib.import_module(module).build_parser()
    loss = next(a for a in p._actions if "--loss_fn" in a.option_strings)
    assert "tsallis" in loss.choices, f"{module} rejects --loss_fn tsallis"


def test_cam_objective_accepts_tsallis():
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import argparse
    from _common import add_cam_args
    p = add_cam_args(argparse.ArgumentParser())
    cam = next(a for a in p._actions if "--cam_objective" in a.option_strings)
    assert "tsallis" in cam.choices


def test_tsallis_kwargs_maps_namespace_and_dict_identically():
    """
    export_conditional_patches.py resolves against a saved config DICT while
    every other caller passes an argparse namespace. One mapping, both shapes.
    """
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import argparse
    from _common import tsallis_kwargs

    ns = argparse.Namespace(tsallis_q=-1.5, tsallis_schedule="linear",
                            tsallis_q_start=-3.0, tsallis_q_end=0.5)
    as_dict = vars(ns)
    assert tsallis_kwargs(ns, 42) == tsallis_kwargs(as_dict, 42)
    assert tsallis_kwargs(ns, 42)["tsallis_total_steps"] == 42
    assert tsallis_kwargs(ns)["tsallis_q_start"] == -3.0


def test_tsallis_kwargs_defaults_on_a_legacy_config():
    """A checkpoint written before these flags existed must still resolve."""
    pytest.importorskip("torchvision")
    _scripts_on_path()
    from _common import tsallis_kwargs
    kw = tsallis_kwargs({}, 1)
    assert kw == {"tsallis_q": 0.0, "tsallis_schedule": "const",
                  "tsallis_q_start": -2.0, "tsallis_q_end": 1.0,
                  "tsallis_total_steps": 1}
    assert build("ce", 8, **kw) is not None          # and it is build()-shaped


def test_tsallis_kwargs_round_trips_into_a_scheduled_loss():
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import argparse
    from _common import tsallis_kwargs
    ns = argparse.Namespace(tsallis_q=0.0, tsallis_schedule="linear",
                            tsallis_q_start=-2.0, tsallis_q_end=1.0)
    obj = build("tsallis", 8, **tsallis_kwargs(ns, 10))
    obj.on_step_begin(0, 10)
    assert obj.q == pytest.approx(-2.0)
    obj.on_step_begin(9, 10)
    assert obj.q == pytest.approx(1.0)


def test_segmentation_cam_build_forwards_tsallis():
    """Signature check: the CAM must be able to receive the q configuration."""
    import inspect
    from patchreach.patch import segmentation_cam
    assert "tsallis" in inspect.signature(segmentation_cam.build).parameters


def test_every_optimisation_loop_installs_the_step_hook():
    """
    THE INVARIANT THIS PINS. A module that runs an optimisation loop calls
    .backward(); a scheduled loss is only scheduled if that loop calls
    on_step_begin(). Add a fourth training entry point and forget the hook, and
    --tsallis_schedule linear silently pins q at q_start for the whole run —
    a wrong result with no error. This fails instead.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    scanned = 0
    for f in list((root / "scripts").glob("*.py")) + \
             list((root / "patchreach").rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        if ".backward()" not in src:
            continue
        scanned += 1
        assert "on_step_begin" in src, (
            f"{f.relative_to(root)} runs an optimisation loop but never calls "
            "on_step_begin — a scheduled loss would not advance in it")
    assert scanned >= 3, f"expected >=3 optimisation loops, scanned {scanned}"


def test_tsallis_tag_is_empty_for_every_other_loss():
    """A non-tsallis run's directory name must not move by a single character."""
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import argparse
    from _common import tsallis_tag
    for name in ("ce", "cospgd", "ipatch_cospgd"):
        ns = argparse.Namespace(loss_fn=name, tsallis_q=-2.0,
                                tsallis_schedule="linear",
                                tsallis_q_start=-2.0, tsallis_q_end=1.0)
        assert tsallis_tag(ns) == ""


@pytest.mark.parametrize("kw,expected", [
    ({"tsallis_schedule": "const", "tsallis_q": 0.0}, "q0"),
    ({"tsallis_schedule": "const", "tsallis_q": -2.0}, "q-2"),
    ({"tsallis_schedule": "const", "tsallis_q": 0.5}, "q0.5"),
    ({"tsallis_schedule": "linear", "tsallis_q_start": -2.0,
      "tsallis_q_end": 1.0}, "q-2to1"),
    ({"tsallis_schedule": "linear", "tsallis_q_start": -3.0,
      "tsallis_q_end": 0.5}, "q-3to0.5"),
])
def test_tsallis_tag_encodes_the_active_schedule(kw, expected):
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import argparse
    from _common import tsallis_tag
    base = {"loss_fn": "tsallis", "tsallis_q": 0.0,
            "tsallis_schedule": "const", "tsallis_q_start": -2.0,
            "tsallis_q_end": 1.0}
    base.update(kw)
    assert tsallis_tag(argparse.Namespace(**base)) == expected


def test_a_q_sweep_produces_distinct_run_directories():
    """
    THE FAILURE THIS PINS. Without a q bit every arm of a sweep resolves to the
    same name and increment_path disambiguates with _2, _3 — the data survives
    but the path stops saying which q produced it.
    """
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import argparse
    from _common import tsallis_tag
    names = set()
    for q in (1.0, 0.0, -1.0, -2.0, -3.0):
        ns = argparse.Namespace(loss_fn="tsallis", tsallis_q=q,
                                tsallis_schedule="const",
                                tsallis_q_start=-2.0, tsallis_q_end=1.0)
        names.add("_".join(x for x in ["segformer", "csf", "tsallis",
                                       "img420", tsallis_tag(ns), ""] if x))
    assert len(names) == 5, names


@pytest.mark.parametrize("module", ["train", "train_conditional_generator"])
def test_run_id_carries_q_only_for_tsallis(module):
    pytest.importorskip("torchvision")
    _scripts_on_path()
    import importlib
    m = importlib.import_module(module)
    a = m.build_parser().parse_args(
        ["--arch", "segformer", "--cityscapes_root", "/x",
         "--loss_fn", "tsallis", "--tsallis_schedule", "linear",
         "--tsallis_q_start", "-2", "--tsallis_q_end", "1"])
    assert "q-2to1" in m.run_id(a)

    b = m.build_parser().parse_args(
        ["--arch", "segformer", "--cityscapes_root", "/x",
         "--loss_fn", "cospgd", "--tsallis_q", "-2"])
    assert "q-2" not in m.run_id(b)
