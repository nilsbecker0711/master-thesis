r"""
Invariants for the POOLED diagnostic suite.

Everything here can be wrong in a way that produces a plausible figure and no
error, which is exactly the failure mode that makes an aggregate dangerous:

  pooling by averaging per-image rates    a class covering 300px in one image
                                          would then count as much as the same
                                          class covering 200k in another.
  per-image distance binning              under a moving footprint bin k would
                                          mean a different physical ring in
                                          every image and the curve would be a
                                          smear of incomparable numbers.
  a resumed run losing the accumulator    the pooled tensors are not
                                          recoverable from the per-image
                                          records, so a silent restart would
                                          report the tail of the run as if it
                                          were the whole of it.
  the run-length verdict                  "converged" is the claim that --steps
                                          did not set the number, and it has to
                                          be false when the tail is climbing.

Run with: pytest -q tests/test_aggregate.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from patchreach.diagnostics import aggregate as A

K = 19
H, W = 128, 256


def _logits(pred, k=K):
    """One-hot-ish logits whose argmax is exactly `pred`."""
    lg = torch.zeros(1, k, *pred.shape[-2:])
    lg.scatter_(1, pred.unsqueeze(1), 8.0)
    return lg


def _footprint(top=32, left=64, side=32):
    fp = torch.zeros(1, H, W, dtype=torch.bool)
    fp[0, top:top + side, left:left + side] = True
    return fp


# ═════════════════════════════════════════════════════════════════════════════
#  Pooling
# ═════════════════════════════════════════════════════════════════════════════

def test_flip_rate_is_pooled_not_averaged_over_images():
    r"""
    Two images: one where class 0 covers a lot and never flips, one where it
    covers a little and always flips. The POOLED rate is close to zero; a mean
    of the two per-image rates would be ~50%.
    """
    d = A.PopulationDiagnostics(K, n_bins=4)
    fp = _footprint()

    # image A: class 0 everywhere, nothing flips
    lab = torch.zeros(1, H, W, dtype=torch.long)
    pred = torch.zeros(1, H, W, dtype=torch.long)
    d.update(_logits(pred), _logits(pred), lab, fp)

    # image B: class 0 on a thin strip only, all of it flips to class 1
    lab_b = torch.full((1, H, W), 255, dtype=torch.long)
    lab_b[0, :2, :] = 0
    pc = torch.zeros(1, H, W, dtype=torch.long)
    pa = torch.ones(1, H, W, dtype=torch.long)
    d.update(_logits(pc), _logits(pa), lab_b, fp)

    rates = d.flip_rates(min_px=1)
    r = rates["road"]["rate"]
    assert r < 5.0, f"pooled flip rate {r:.1f}% looks like a mean of rates"
    # and the denominator really is the sum, not the count of images
    assert rates["road"]["total"] == int(d.pred_total[0])


def test_distance_bins_are_absolute_so_two_placements_stay_comparable():
    """Bin k must be the same physical ring whatever the footprint does."""
    lab = torch.zeros(1, H, W, dtype=torch.long)
    pred = torch.zeros(1, H, W, dtype=torch.long)

    a = A.PopulationDiagnostics(K, n_bins=6)
    a.update(_logits(pred), _logits(pred), lab, _footprint(0, 0))
    b = A.PopulationDiagnostics(K, n_bins=6)
    b.update(_logits(pred), _logits(pred), lab, _footprint(90, 200))

    # different placements populate different bins, but every populated bin
    # covers the SAME [lo, hi) in pixels — which is what makes them addable.
    assert A.BIN_EDGES[0] == (0, A.BIN_W)
    assert len(A.BIN_CENTRES) == A.N_BINS
    assert int(a.reach_total.sum()) > 0 and int(b.reach_total.sum()) > 0


def test_reach_denominator_excludes_unlabelled_pixels():
    r"""
    A flip is only counted on labelled pixels, so an unlabelled pixel must not
    sit in the denominator either — otherwise ignore-heavy images would depress
    the pooled reach curve without ever being able to raise it.
    """
    lab = torch.full((1, H, W), 255, dtype=torch.long)
    lab[0, :, :W // 2] = 3
    pc = torch.zeros(1, H, W, dtype=torch.long)
    d = A.PopulationDiagnostics(K, n_bins=8)
    d.update(_logits(pc), _logits(pc), lab, _footprint())
    assert int(d.reach_total.sum()) == int(((lab != 255) & ~_footprint()).sum())


def test_pooled_flows_find_a_single_dominant_channel():
    """The implicit-class-selector claim, at population scale."""
    d = A.PopulationDiagnostics(K, n_bins=4)
    lab = torch.zeros(1, H, W, dtype=torch.long)
    pc = torch.zeros(1, H, W, dtype=torch.long)
    pa = torch.zeros(1, H, W, dtype=torch.long)
    pa[0, :, :W // 2] = 13                      # road -> car, everywhere
    for _ in range(3):
        d.update(_logits(pc), _logits(pa), lab, _footprint())

    fl = d.flows()
    assert fl[0]["src"] == "road" and fl[0]["dst"] == "car"
    assert fl[0]["pct_of_flips"] > 99.0


# ═════════════════════════════════════════════════════════════════════════════
#  Resume
# ═════════════════════════════════════════════════════════════════════════════

def test_state_round_trip_preserves_every_pooled_tensor():
    d = A.PopulationDiagnostics(K, n_bins=6)
    lab = torch.zeros(1, H, W, dtype=torch.long)
    pc = torch.zeros(1, H, W, dtype=torch.long)
    pa = torch.zeros(1, H, W, dtype=torch.long)
    pa[0, 40:60, 80:120] = 13
    for i in range(2):
        d.update(_logits(pc), _logits(pa), lab, _footprint(),
                 {"image": i, "clean_remote": 50.0,
                  "history": [{"step": 1, "miou_remote": 50.0},
                              {"step": 2, "miou_remote": 45.0}]})

    r = A.PopulationDiagnostics(K, n_bins=6).load_state_dict(d.state_dict())
    assert torch.equal(r.flip_cm, d.flip_cm)
    assert torch.equal(r.reach_total, d.reach_total)
    assert torch.equal(r.pred_total, d.pred_total)
    assert r.n_images == d.n_images == 2
    assert r.curves == d.curves
    assert r.done_images == {0, 1}


def test_incompatible_checkpoint_is_refused_rather_than_added():
    """Adding a K=4 matrix into a K=19 one is silent nonsense, so: raise."""
    small = A.PopulationDiagnostics(4, n_bins=6)
    with pytest.raises(ValueError, match="not compatible"):
        A.PopulationDiagnostics(K, n_bins=6).load_state_dict(
            small.state_dict())


# ═════════════════════════════════════════════════════════════════════════════
#  Verdicts
# ═════════════════════════════════════════════════════════════════════════════

def _with_curves(drops_per_step):
    d = A.PopulationDiagnostics(K, n_bins=4)
    lab = torch.zeros(1, H, W, dtype=torch.long)
    pc = torch.zeros(1, H, W, dtype=torch.long)
    for i in range(4):
        hist = [{"step": s, "miou_remote": 50.0 - v}
                for s, v in drops_per_step]
        d.update(_logits(pc), _logits(pc), lab, _footprint(),
                 {"image": i, "clean_remote": 50.0, "history": hist})
    return d


def test_a_climbing_tail_is_reported_as_not_converged():
    steps = [(s, s * 0.05) for s in range(100, 1100, 100)]   # still rising
    out = _with_curves(steps)._report_convergence(lambda *a, **k: None)
    assert out["converged"] is False
    assert out["tail_gain"] > 1.0


def test_a_flat_tail_is_reported_as_converged():
    steps = [(s, min(s, 500) * 0.05) for s in range(100, 1100, 100)]
    out = _with_curves(steps)._report_convergence(lambda *a, **k: None)
    assert out["converged"] is True


def test_realised_visibility_is_reported_against_tau_not_instead_of_it():
    r"""
    tau is the INTENT and the realised index is the OUTCOME. The ratio between
    them is the whole point of the figure, so it has to be in the dict.
    """
    d = A.PopulationDiagnostics(K, n_bins=4)
    recs = [{"final_visibility": 0.7, "final_visibility_local": 1.4}
            for _ in range(5)]
    out = d._report_visibility(0.25, recs, lambda *a, **k: None)
    assert out["tau"] == 0.25
    assert out["final_visibility"]["median_over_tau"] == pytest.approx(2.8)
    assert out["final_visibility"]["frac_over_tau"] == pytest.approx(100.0)
    # a non-csf run carries no visibility keys and must simply say nothing
    assert d._report_visibility(None, [{"drop_remote": 3.0}],
                                lambda *a, **k: None) == {}


def test_empty_accumulator_summarises_and_draws_nothing(tmp_path):
    d = A.PopulationDiagnostics(K)
    out = d.summarise(tau=0.25, records=[], log=lambda *a, **k: None)
    assert out["n_images"] == 0
    assert d.write_figures(tmp_path, tau=0.25, records=[]) == []


def test_figures_are_written_and_are_standalone_files(tmp_path):
    pytest.importorskip("matplotlib")
    d = A.PopulationDiagnostics(K, n_bins=8)
    lab = torch.zeros(1, H, W, dtype=torch.long)
    pc = torch.zeros(1, H, W, dtype=torch.long)
    pa = torch.zeros(1, H, W, dtype=torch.long)
    pa[0, 40:70, 70:130] = 13
    recs = []
    for i in range(4):
        rec = {"image": i, "clean_remote": 50.0, "final_visibility": 0.4,
               "final_visibility_local": 0.8,
               "history": [{"step": s, "miou_remote": 50.0 - s * 0.01}
                           for s in (10, 20, 30, 40, 50)]}
        recs.append(rec)
        d.update(_logits(pc), _logits(pa), lab, _footprint(), rec)

    figs = d.write_figures(tmp_path, tau=0.25, records=recs, title="test")
    assert "reach.png" in figs and "visibility.png" in figs
    assert "convergence.png" in figs and "flip_flows.png" in figs
    for f in figs:
        assert (tmp_path / f).stat().st_size > 1000
