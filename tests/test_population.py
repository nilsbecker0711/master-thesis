r"""
Invariants for population-level per-image attacks.

Three things here can be wrong in a way that produces a plausible number and no
error: the pooled dataset mIoU being quietly reported as the mean of per-image
mIoUs (they are different quantities and miou.py says so), a resumed run
pooling only the images since the restart, and the panel selection silently
showing only the strongest attacks. One test each, plus the anti-drift property
that overfit.py and overfit_population.py run the SAME attack.

Run with: pytest -q tests/test_population.py
"""
import json

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from patchreach.metrics import population as P
from patchreach.metrics.miou import SegMetric
from patchreach.patch import optimise
from patchreach.patch.spec import Patch, PatchConfig

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  Distribution summary
# ═════════════════════════════════════════════════════════════════════════════

def test_describe_reports_a_distribution_not_just_a_mean():
    """semantic.py: 'Report it as a distribution, never a single number.'"""
    d = P.describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert d["n"] == 5
    assert d["mean"] == pytest.approx(3.0)
    assert d["median"] == pytest.approx(3.0)
    assert d["min"] == pytest.approx(1.0) and d["max"] == pytest.approx(5.0)
    assert d["q1"] == pytest.approx(2.0) and d["q3"] == pytest.approx(4.0)
    # sample std (ddof=1), not population std: these images are a SAMPLE
    assert d["std"] == pytest.approx(1.5811, abs=1e-3)


def test_success_rates_count_images_over_each_threshold():
    d = P.describe([0.0, 2.0, 7.0, 20.0])
    assert d["success_rate"]["1.0"] == pytest.approx(75.0)
    assert d["success_rate"]["5.0"] == pytest.approx(50.0)
    assert d["success_rate"]["10.0"] == pytest.approx(25.0)


def test_bootstrap_ci_brackets_the_mean_and_narrows_with_n():
    g = torch.Generator().manual_seed(0)
    small = (torch.randn(20, generator=g) * 5 + 10).tolist()
    large = (torch.randn(400, generator=g) * 5 + 10).tolist()
    ls, hs = P.bootstrap_ci(small)
    ll, hl = P.bootstrap_ci(large)
    assert ls < sum(small) / len(small) < hs
    assert (hl - ll) < (hs - ls)


def test_describe_survives_degenerate_inputs():
    assert P.describe([])["n"] == 0
    d = P.describe([4.0])
    assert d["n"] == 1 and d["std"] == 0.0
    assert d["ci95_boot"] == [4.0, 4.0]


def test_images_needed_matches_the_formula():
    """n = (1.96 sigma / half_width)^2 — the number that lets the sample size
    defend itself with the variance actually measured."""
    assert P.images_needed(10.0, 1.0) == 385         # ceil((19.6)^2)
    assert P.images_needed(10.0, 2.0) == 97
    assert P.images_needed(0.0, 1.0) == 1


# ═════════════════════════════════════════════════════════════════════════════
#  Pooled vs per-image
# ═════════════════════════════════════════════════════════════════════════════

def _fake_logits(pred, K=4):
    """One-hot logits that argmax to `pred`."""
    return torch.nn.functional.one_hot(pred, K).permute(0, 3, 1, 2).float() * 10


def _pop_with(preds_clean, preds_adv, labels, K=4):
    pop = P.Population(K, device="cpu")
    for pc, pa, lb in zip(preds_clean, preds_adv, labels):
        fp = torch.zeros_like(lb, dtype=torch.bool)
        pop.update(_fake_logits(pc, K), _fake_logits(pa, K), lb, fp,
                   {"image": len(pop.records), "drop_remote": 0.0})
    return pop


def test_pooled_miou_is_one_confusion_matrix_over_every_image():
    """
    miou.py: 'DATASET mIoU accumulates one confusion matrix across many
    images. Published numbers are always dataset mIoU.' The pooled figure must
    equal a SegMetric fed every image, not an average of per-image mIoUs.
    """
    K = 4
    g = torch.Generator().manual_seed(0)
    labels = [torch.randint(0, K, (1, 8, 8), generator=g) for _ in range(5)]
    clean = [lb.clone() for lb in labels]
    adv = [torch.randint(0, K, (1, 8, 8), generator=g) for _ in range(5)]

    pop = _pop_with(clean, adv, labels, K)
    ref = SegMetric(K)
    for pa, lb in zip(adv, labels):
        ref.update(pa, lb)
    assert pop.pooled()["adv_all"] == pytest.approx(ref.compute(), abs=1e-6)


def test_pooled_is_not_the_mean_of_per_image_mious():
    """
    The error this module exists to prevent. A rare class present in one image
    drags that image's per-image mIoU to near zero while barely moving the
    pooled confusion matrix, so the two numbers genuinely differ — and
    reporting one as the other is wrong in a way no error message catches.
    """
    K = 4
    from patchreach.metrics.miou import single_image_miou
    g = torch.Generator().manual_seed(3)
    labels = [torch.randint(0, 2, (1, 8, 8), generator=g) for _ in range(3)]
    labels[0][0, 0, 0] = 3                       # a rare class, 1 pixel
    adv = [torch.randint(0, 2, (1, 8, 8), generator=g) for _ in range(3)]

    pop = _pop_with([lb.clone() for lb in labels], adv, labels, K)
    per_image = [single_image_miou(_fake_logits(pa, K), lb, K)
                 for pa, lb in zip(adv, labels)]
    mean_per_image = sum(per_image) / len(per_image)
    assert abs(pop.pooled()["adv_all"] - mean_per_image) > 1e-3


def test_pooled_reports_both_class_sets_and_flags_a_moved_denominator():
    r"""
    The artefact that made a 150-epoch run report drop_remote = -3.63, i.e. the
    attack apparently IMPROVING the model. A class with no ground truth that the
    ADVERSARIAL pass predicts somewhere enters the 'union' denominator at IoU 0
    and drags that mean down; under 'gt' it is not evaluated in either pass and
    cannot move. Pooling must inherit the fix, not re-introduce it.
    """
    K = 4
    lb = torch.zeros(1, 16, 16, dtype=torch.long)
    lb[0, :8, :] = 1                                # GT has classes 0 and 1
    clean = lb.clone()
    adv = lb.clone()
    adv[0, 15, 15] = 3                              # one spurious GT-absent px
    fp = torch.zeros(1, 16, 16, dtype=torch.bool)

    pop = P.Population(K, device="cpu")
    pop.update(_fake_logits(clean, K), _fake_logits(adv, K), lb, fp,
               {"image": 0, "drop_remote": 0.0})
    out = pop.pooled("gt")

    assert out["class_set_moved"] is True
    assert out["n_classes_clean_union"] != out["n_classes_adv_union"]
    # under 'gt' the single wrong pixel is a tiny real drop; under 'union' the
    # class set moves and the drop is dominated by the artefact
    assert out["drop_remote"] < 1.0
    assert out["drop_remote_union"] > out["drop_remote"]


def test_pooled_classes_argument_selects_which_number_is_headline():
    K = 4
    g = torch.Generator().manual_seed(5)
    lb = torch.randint(0, 2, (1, 16, 16), generator=g)
    adv = torch.randint(0, 2, (1, 16, 16), generator=g)
    fp = torch.zeros_like(lb, dtype=torch.bool)
    pop = P.Population(K, device="cpu")
    pop.update(_fake_logits(lb.clone(), K), _fake_logits(adv, K), lb, fp,
               {"image": 0, "drop_remote": 0.0})
    assert pop.pooled("gt")["classes"] == "gt"
    assert pop.pooled("union")["classes"] == "union"
    assert pop.pooled("union")["drop_remote"] == \
        pytest.approx(pop.pooled("gt")["drop_remote_union"])


def test_record_carries_end_of_run_patch_stats():
    r"""
    These previously lived only in `history`, which the population scripts strip
    before writing summary.json — so a population run could not report realised
    visibility, the one number the CSF family's claim rests on, nor
    frac_at_clip, the early warning for saturation collapse. analysis/pick_lr.py
    reads them to select a learning rate at equal perceptual cost, so losing
    them silently breaks that selection rather than erroring.
    """
    model, img, label, patch = _setup()
    clean = optimise.prepare(model, img, patch, 64, 64)
    rec = optimise.attack_image(model, img, label, patch, steps=4, log_every=2,
                                num_classes=4, clean_logits=clean,
                                verbose=False)
    # raw mode reports spread and saturation
    assert "final_frac_at_clip" in rec
    assert "final_pixel_std" in rec
    assert all(k.startswith("final_") for k in rec if k.startswith("final_"))


def test_csf_record_carries_realised_visibility():
    """tau is an INTENT; realised visibility is the OUTCOME, and only the
    outcome is reportable."""
    img = (torch.rand(1, 3, 64, 64) - MEAN) / STD
    label = torch.randint(0, 4, (1, 64, 64))
    torch.manual_seed(0)
    model = _TinySeg(4)
    patch = Patch(PatchConfig(mode="csf", size=16, scale=0.25),
                  torch.device("cpu"), MEAN, STD)
    clean = optimise.prepare(model, img, patch, 64, 64, from_image=True,
                             mean_t=MEAN, std_t=STD)
    rec = optimise.attack_image(model, img, label, patch, steps=4, log_every=2,
                                num_classes=4, clean_logits=clean,
                                verbose=False)
    assert rec["final_visibility"] > 0
    assert "final_resid_rms" in rec


def test_attack_image_records_both_class_sets():
    model, img, label, patch = _setup()
    clean = optimise.prepare(model, img, patch, 64, 64)
    rec = optimise.attack_image(model, img, label, patch, steps=4, log_every=2,
                                num_classes=4, clean_logits=clean,
                                verbose=False)
    for k in ("classes", "drop_remote_union", "drop_all_union",
              "n_classes_gt", "class_set_moved"):
        assert k in rec, k
    assert rec["classes"] == "gt"


def test_remote_pooling_excludes_each_images_own_footprint():
    """Every image has its OWN patch and placement here, so one shared
    exclusion mask would be wrong."""
    K = 4
    lb = torch.zeros(1, 8, 8, dtype=torch.long)
    pa = torch.zeros(1, 8, 8, dtype=torch.long)
    pa[0, :4, :] = 1                              # wrong in the top half
    fp = torch.zeros(1, 8, 8, dtype=torch.bool)
    fp[0, :4, :] = True                           # ...which is the footprint

    pop = P.Population(K, device="cpu")
    pop.update(_fake_logits(lb, K), _fake_logits(pa, K), lb, fp,
               {"image": 0, "drop_remote": 0.0})
    # Every remaining remote pixel is correct, so remote mIoU is perfect
    # while 'all' is not — the whole point of the remote/all distinction.
    assert pop.pooled()["adv_remote"] == pytest.approx(100.0)
    assert pop.pooled()["adv_all"] < 100.0


def test_pooled_flip_rate_uses_pixel_counts_not_an_image_average():
    K = 4
    lb = torch.zeros(1, 10, 10, dtype=torch.long)
    fp = torch.zeros(1, 10, 10, dtype=torch.bool)
    a = torch.zeros(1, 10, 10, dtype=torch.long)
    a[0, :2, :] = 1                                # 20 of 100 flipped
    b = torch.zeros(1, 10, 10, dtype=torch.long)   # 0 of 100 flipped

    pop = P.Population(K, device="cpu")
    for pa in (a, b):
        pop.update(_fake_logits(lb, K), _fake_logits(pa, K), lb, fp,
                   {"image": len(pop.records), "drop_remote": 0.0})
    assert pop.pooled()["any_flip_rate"] == pytest.approx(10.0)


# ═════════════════════════════════════════════════════════════════════════════
#  Resume
# ═════════════════════════════════════════════════════════════════════════════

def test_state_round_trip_preserves_the_pooled_matrices():
    """
    A confusion matrix cannot be rebuilt from a scalar mIoU, so a run resumed
    from the records alone would pool only the images since the restart and
    report it as if it covered all of them.
    """
    K = 4
    g = torch.Generator().manual_seed(1)
    labels = [torch.randint(0, K, (1, 8, 8), generator=g) for _ in range(4)]
    adv = [torch.randint(0, K, (1, 8, 8), generator=g) for _ in range(4)]
    pop = _pop_with([lb.clone() for lb in labels], adv, labels, K)
    before = pop.pooled()

    fresh = P.Population(K, device="cpu").load_state_dict(pop.state_dict())
    assert fresh.pooled() == pytest.approx(before)
    assert fresh.done_images == pop.done_images


def test_resume_rejects_an_incompatible_checkpoint():
    """Pooling matrices of different widths, or hit rates with different
    denominators, would silently mix two definitions."""
    st = P.Population(4, device="cpu").state_dict()
    with pytest.raises(ValueError):
        P.Population(19, device="cpu").load_state_dict(st)
    with pytest.raises(ValueError):
        P.Population(4, device="cpu", target_class=8).load_state_dict(st)


# ═════════════════════════════════════════════════════════════════════════════
#  Panel selection
# ═════════════════════════════════════════════════════════════════════════════

def _recs(vals):
    return [{"image": i, "drop_remote": v} for i, v in enumerate(vals)]


def test_best_picks_the_strongest():
    got = P.select(_recs([1, 9, 5, 7, 3]), 2, "best")
    assert [r["drop_remote"] for r in got] == [9, 7]


def test_worst_picks_the_weakest():
    got = P.select(_recs([1, 9, 5, 7, 3]), 2, "worst")
    assert [r["drop_remote"] for r in got] == [1, 3]


def test_spread_is_the_default_and_covers_the_range():
    """
    The default is `spread`, not `best`, because the population summary reports
    a distribution and a panel figure showing only the strongest attacks
    contradicts it — which is exactly what a reviewer calls cherry-picking.
    """
    vals = list(range(1, 22))                     # 1..21
    got = [r["drop_remote"] for r in P.select(_recs(vals), 3, "spread")]
    assert max(vals) in got
    assert min(vals) in got
    assert len(set(got)) == 3
    assert min(got) < sorted(got)[1] < max(got)   # a genuine middle


def test_selection_is_capped_by_the_population_and_never_duplicates():
    got = P.select(_recs([4, 8]), 5, "spread")
    assert len(got) == 2
    assert len({r["image"] for r in got}) == 2
    assert P.select(_recs([1, 2, 3]), 0, "best") == []
    assert P.select([], 3, "best") == []


def test_selection_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        P.select(_recs([1, 2]), 1, "cherry")


# ═════════════════════════════════════════════════════════════════════════════
#  The shared attack procedure
# ═════════════════════════════════════════════════════════════════════════════

class _TinySeg(nn.Module):
    def __init__(self, K=4):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.head = nn.Conv2d(8, K, 1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.head(torch.relu(self.c1(x)))


def _setup(H=64, W=64, K=4):
    torch.manual_seed(0)
    model = _TinySeg(K)
    img = (torch.rand(1, 3, H, W) - MEAN) / STD
    label = torch.randint(0, K, (1, H, W))
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25),
                  torch.device("cpu"), MEAN, STD)
    return model, img, label, patch


def test_prepare_resolves_placement_before_the_attack():
    """The ordering bug that silently falls back to centre: placement must be
    resolved from the CLEAN prediction, so it cannot be None afterwards."""
    model, img, label, patch = _setup()
    assert patch.placement is None
    clean = optimise.prepare(model, img, patch, 64, 64)
    assert patch.placement is not None
    assert clean.shape[-2:] == (64, 64)


def test_attack_image_returns_the_expected_record(tmp_path):
    model, img, label, patch = _setup()
    clean = optimise.prepare(model, img, patch, 64, 64)
    before = patch.param.detach().clone()

    rec = optimise.attack_image(model, img, label, patch, steps=6, log_every=3,
                                num_classes=4, clean_logits=clean,
                                out_dir=tmp_path, verbose=False)

    for k in ("clean_all", "clean_remote", "final_all", "final_remote",
              "drop_all", "drop_remote", "best_drop_remote", "any_flip_rate",
              "degraded_after_peak", "history"):
        assert k in rec, k
    assert not torch.equal(before, patch.param.detach())   # it optimised
    assert (tmp_path / "best.pt").exists()
    assert json.dumps(  # the record must survive json.dump for results.json
        {k: v for k, v in rec.items() if k != "history"})


def test_attack_image_is_deterministic_for_a_fixed_seed():
    """
    The anti-drift property. overfit.py and overfit_population.py call this
    same function, so a per-image result must not depend on which script
    invoked it — otherwise the population numbers and the single-image numbers
    diverge and nothing says why.
    """
    outs = []
    for _ in range(2):
        model, img, label, patch = _setup()
        clean = optimise.prepare(model, img, patch, 64, 64)
        outs.append(optimise.attack_image(
            model, img, label, patch, steps=6, log_every=3, num_classes=4,
            clean_logits=clean, verbose=False)["drop_remote"])
    assert outs[0] == pytest.approx(outs[1])


def test_attack_image_does_not_write_step_images_by_default(tmp_path):
    """Right for one interactive run, catastrophic across hundreds of images."""
    model, img, label, patch = _setup()
    clean = optimise.prepare(model, img, patch, 64, 64)
    optimise.attack_image(model, img, label, patch, steps=6, log_every=3,
                          num_classes=4, clean_logits=clean, out_dir=tmp_path,
                          verbose=False)
    assert not list(tmp_path.glob("patch_step*.png"))

    optimise.attack_image(model, img, label, patch, steps=6, log_every=3,
                          num_classes=4, clean_logits=clean, out_dir=tmp_path,
                          save_step_images=True, verbose=False)
    assert list(tmp_path.glob("patch_step*.png"))


def test_non_finite_loss_raises_rather_than_poisoning_adam():
    """adversarial.py records the cost of NOT doing this: a NaN propagates
    through backward(), poisons Adam's moments permanently and freezes the
    patch for the rest of training with no error raised."""
    model, img, label, patch = _setup()
    clean = optimise.prepare(model, img, patch, 64, 64)

    def explode(*_, **__):
        return torch.tensor(float("nan"), requires_grad=True)

    import patchreach.losses.adversarial as adv
    orig = adv.build
    adv.build = lambda *a, **k: explode
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            optimise.attack_image(model, img, label, patch, steps=3,
                                  num_classes=4, clean_logits=clean,
                                  verbose=False)
    finally:
        adv.build = orig


# ═════════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════════

def test_summarise_reports_both_numbers_and_flags_underpowering():
    K = 4
    g = torch.Generator().manual_seed(2)
    pop = P.Population(K, device="cpu")
    for i in range(6):
        lb = torch.randint(0, K, (1, 8, 8), generator=g)
        pa = torch.randint(0, K, (1, 8, 8), generator=g)
        fp = torch.zeros_like(lb, dtype=torch.bool)
        pop.update(_fake_logits(lb, K), _fake_logits(pa, K), lb, fp,
                   {"image": i, "drop_remote": float(i * 4),
                    "degraded_after_peak": False})

    out = pop.summarise(log=lambda *a, **k: None)
    assert out["n"] == 6
    assert "distribution" in out and "pooled" in out
    # spread is large and n is tiny, so this must declare itself underpowered
    assert out["underpowered_pm2"] is True
    assert out["images_needed_pm1"] > out["images_needed_pm2"]


def test_summarise_handles_an_empty_population():
    out = P.Population(4, device="cpu").summarise(log=lambda *a, **k: None)
    assert out["n"] == 0
