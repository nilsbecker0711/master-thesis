r"""
Invariants for the trade-off panel — analysis/tradeoff_panel.py.

The figure this script draws is read as an argument: "the constrained attack
reaches X% of the unconstrained one at a budget the eye cannot see". Every
failure below produces a plausible-looking panel and no error, which is the
only kind worth a test here:

  a curve assembled at mixed learning rates   a picture of the lr grid, filed
                                              as a picture of tau. The sweep's
                                              stage overlap makes this the
                                              DEFAULT outcome of globbing
                                              cells/, so from_sweep() reads the
                                              manifest instead.
  the knee missed                             the flat tail then reads as "the
                                              attack saturates" when what
                                              saturated was the budget, and
                                              every tau above it gets quoted as
                                              an operating point.
  a seed spread drawn as a confidence
  interval                                    +/- 1 sd over three repeats of
                                              ONE image is not an interval over
                                              scenes, and the two must not
                                              share a label.
  a raw arm folded into the curve             the ceiling would become a point
                                              on the thing it is meant to bound.

Run with: pytest -q tests/test_tradeoff_panel.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "tradeoff_panel", REPO / "analysis" / "tradeoff_panel.py")
TP = importlib.util.module_from_spec(_spec)
sys.modules["tradeoff_panel"] = TP
_spec.loader.exec_module(TP)


# ─────────────────────────────────────────────────────────────────────────────
#  fixtures
# ─────────────────────────────────────────────────────────────────────────────
def cfg(arch="segformer_b0", mode="csf", tau=0.25, lr=0.01, steps=1000,
        enforce="realised", **kw):
    d = {"arch": arch, "patch_mode": mode, "csf_threshold": tau,
         "csf_enforce": enforce, "csf_param": "pgd", "loss_fn": "cospgd",
         "steps": steps, "lr": lr, "lr_schedule": "cosine",
         "placement": "center", "img_h": 512, "img_w": 1024}
    d.update(kw)
    return d


def write_population(root, name, c, n=12, drop=10.0, vis=0.25):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(c))
    recs = [{"image": i, "drop_remote": drop + 0.1 * i,
             "any_flip_rate": 40.0 + i, "final_visibility": vis}
            for i in range(n)]
    (d / "summary.json").write_text(json.dumps({"config": c, "records": recs}))
    return d


def write_seeds(root, name, c, k=3, drop=10.0, vis=0.25):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(c))
    rows = [{"drop_remote": drop + i, "any_flip_rate": 40.0 + i,
             "final_visibility": vis} for i in range(k)]
    (d / "summary.json").write_text(json.dumps({"n_seeds": k,
                                                "per_seed": rows}))
    return d


def write_single(root, name, c, drop=10.0, vis=0.25):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(c))
    (d / "results.json").write_text(json.dumps(
        {"config": c, "drop_remote": drop, "any_flip_rate": 40.0,
         "final_visibility": vis}))
    return d


METRICS = ["any_flip_rate", "drop_remote"]


# ─────────────────────────────────────────────────────────────────────────────
#  the three layouts, and the interval each one earns
# ─────────────────────────────────────────────────────────────────────────────
def test_population_gets_a_bootstrap_interval(tmp_path):
    d = write_population(tmp_path, "pop", cfg())
    run = TP.load_run(d)
    assert run["kind"] == "population"
    s = TP.stat(run["rows"], "drop_remote")
    assert s["n"] == 12 and s["band"] == "bootstrap 95% CI"
    assert s["lo"] < s["mean"] < s["hi"]


def test_seed_repeats_are_labelled_as_sd_not_as_a_ci(tmp_path):
    """A spread over repeats of ONE image is not an interval over scenes."""
    d = write_seeds(tmp_path, "seeded", cfg())
    s = TP.stat(TP.load_run(d)["rows"], "drop_remote")
    assert s["n"] == 3
    assert "sd" in s["band"] and "CI" not in s["band"]


def test_a_single_run_gets_no_interval_at_all(tmp_path):
    """Not a zero-width one — that would read as a precise measurement."""
    d = write_single(tmp_path, "one", cfg())
    s = TP.stat(TP.load_run(d)["rows"], "drop_remote")
    assert s["n"] == 1 and s["lo"] is None and s["hi"] is None


def test_a_directory_with_no_results_is_not_a_run(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg()))
    assert TP.load_run(d) is None


# ─────────────────────────────────────────────────────────────────────────────
#  classification
# ─────────────────────────────────────────────────────────────────────────────
def test_raw_becomes_the_ceiling_and_lap_is_skipped(tmp_path):
    write_population(tmp_path, "csf", cfg(tau=0.5))
    write_population(tmp_path, "raw", cfg(mode="raw"), drop=40.0)
    write_population(tmp_path, "lap", cfg(mode="lap"))
    dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())

    curve, ceiling, skipped = TP.collect(dirs, METRICS, 0)
    assert [p["tau"] for p in curve] == [0.5]
    assert len(ceiling) == 1 and ceiling[0]["mode"] == "raw"
    assert len(skipped) == 1 and "lap" in skipped[0][1]


# ─────────────────────────────────────────────────────────────────────────────
#  the knee
# ─────────────────────────────────────────────────────────────────────────────
def _pt(tau, vis, enforce="realised"):
    return {"tau": tau, "enforce": enforce,
            "final_visibility": {"mean": vis, "lo": None, "hi": None, "n": 1,
                                 "band": ""}}


def test_knee_under_realised_is_where_the_request_stops_being_met():
    # realised tracks tau to 2.0, then the range fit pins it there
    pts = [_pt(0.25, 0.25), _pt(0.5, 0.5), _pt(1.0, 1.0), _pt(2.0, 2.0),
           _pt(4.0, 2.05), _pt(8.0, 2.05)]
    k = TP.knee(pts, 0.05)
    assert k["tau_max_tracking"] == 2.0
    assert "realised" in k["rule"]


def test_knee_under_nominal_falls_back_to_saturation():
    """A ratio rule would flag every rung; the question is still the same."""
    pts = [_pt(0.25, 0.10, "nominal"), _pt(0.5, 0.20, "nominal"),
           _pt(1.0, 0.38, "nominal"), _pt(2.0, 0.39, "nominal"),
           _pt(4.0, 0.39, "nominal")]
    k = TP.knee(pts, 0.05)
    assert k["tau_max_tracking"] == 1.0
    assert "grew" in k["rule"]


def test_knee_is_none_when_nothing_tracked():
    k = TP.knee([_pt(1.0, 0.2), _pt(2.0, 0.2)], 0.05)
    assert k["tau_max_tracking"] is None


def test_knee_reports_rather_than_guesses_without_visibility():
    k = TP.knee([{"tau": 1.0, "enforce": "realised"}], 0.05)
    assert k["tau_max_tracking"] is None and "no realised" in k["rule"]


# ─────────────────────────────────────────────────────────────────────────────
#  duplicates and drift
# ─────────────────────────────────────────────────────────────────────────────
def test_identical_cells_collapse_but_differing_ones_do_not(tmp_path):
    write_population(tmp_path, "a", cfg(tau=0.5), n=12)
    write_population(tmp_path, "b", cfg(tau=0.5), n=9)      # same config
    write_population(tmp_path, "c", cfg(tau=0.5, lr=0.2))   # a real difference
    dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    curve, _, _ = TP.collect(dirs, METRICS, 0)

    kept, collapsed = TP.dedup_identical(curve)
    assert collapsed == 1                       # a and b were one cell
    assert len(kept) == 2                       # c is a second condition
    assert max(p["n_rows"] for p in kept) == 12  # the fuller run won
    assert "lr" in TP.axis_drift(kept)


def test_axis_drift_is_silent_on_a_clean_ladder(tmp_path):
    for t in (0.1, 0.25, 0.5):
        write_population(tmp_path, f"t{t}", cfg(tau=t))
    dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    curve, _, _ = TP.collect(dirs, METRICS, 0)
    assert TP.axis_drift(curve) == {}


# ─────────────────────────────────────────────────────────────────────────────
#  reading a sweep at its operating point
# ─────────────────────────────────────────────────────────────────────────────
def test_sweep_is_read_at_the_operating_point_not_globbed(tmp_path):
    r"""
    The sweep's stage overlap means the incumbent-tau rung lives under the lr
    stage's tag and the lr ladder shares its tau. Globbing cells/ therefore
    yields a curve at four learning rates; the manifest is what disambiguates.
    """
    root = tmp_path / "sweep"
    cells = root / "cells"
    cells.mkdir(parents=True)
    # the lr ladder, all at tau 0.25 — only lr 0.01 is the operating point
    for lr in (0.003, 0.01, 0.03):
        write_seeds(cells, f"lr{lr}_tau0.25", cfg(tau=0.25, lr=lr))
    # the tau ladder at the chosen lr, with NO tau 0.25 cell (deduped away)
    for t in (0.5, 1.0, 2.0):
        write_seeds(cells, f"tau{t}", cfg(tau=t, lr=0.01))
    # a cell at another run length, which must not enter either
    write_seeds(cells, "steps200", cfg(tau=0.25, lr=0.01, steps=200))
    write_population(cells, "ceiling", cfg(mode="raw"), drop=40.0)
    (root / "sweep.json").write_text(json.dumps({
        "config": {"arch": "segformer_b0"},
        "decisions": {"cospgd": {"operating_point": {
            "tau": 0.25, "lr": 0.01, "steps": 1000,
            "enforce": "realised", "loss": "cospgd"}}}}))

    curve, ceiling, _ = TP.from_sweep(root, METRICS, 0)
    assert sorted(p["tau"] for p in curve) == [0.25, 0.5, 1.0, 2.0]
    assert {p["axes"]["lr"] for p in curve} == {0.01}
    assert {p["axes"]["steps"] for p in curve} == {1000}
    assert len(ceiling) == 1


def test_sweep_without_a_decision_keeps_everything_rather_than_nothing(
        tmp_path):
    """An aborted sweep still has cells; it must not silently plot an empty
    panel."""
    root = tmp_path / "sweep"
    cells = root / "cells"
    cells.mkdir(parents=True)
    for t in (0.25, 0.5):
        write_seeds(cells, f"tau{t}", cfg(tau=t))
    (root / "sweep.json").write_text(json.dumps(
        {"config": {"arch": "segformer_b0"}, "decisions": {}}))

    curve, _, _ = TP.from_sweep(root, METRICS, 0)
    assert sorted(p["tau"] for p in curve) == [0.25, 0.5]


def test_a_directory_that_is_not_a_sweep_says_so(tmp_path):
    curve, ceiling, skipped = TP.from_sweep(tmp_path, METRICS, 0)
    assert not curve and not ceiling
    assert "sweep.json" in skipped[0][1]


# ─────────────────────────────────────────────────────────────────────────────
#  the figure itself
# ─────────────────────────────────────────────────────────────────────────────
def test_draw_writes_a_figure_and_reports_the_fraction_of_the_ceiling(
        tmp_path):
    pytest.importorskip("matplotlib")
    ladder = []
    for t, vis, drop in [(0.25, 0.25, 10.0), (0.5, 0.5, 20.0),
                         (1.0, 1.0, 30.0), (2.0, 1.05, 31.0)]:
        d = write_population(tmp_path / "runs", f"t{t}", cfg(tau=t),
                             drop=drop, vis=vis)
        ladder.append(d)
    write_population(tmp_path / "runs", "raw", cfg(mode="raw"), drop=60.0)
    dirs = sorted(p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    curve, ceiling, _ = TP.collect(dirs, METRICS, 0)

    out = tmp_path / "fig.png"
    notes = TP.draw({"segformer_b0": {"curve": curve, "ceiling": ceiling}},
                    ["drop_remote"], "tau", False, 0.05, out, None, "best",
                    True)
    assert out.exists() and out.with_suffix(".pdf").exists()

    n = notes["segformer_b0"]
    assert n["knee"]["tau_max_tracking"] == 1.0        # 2.0 did not track
    frac = n["frac_of_ceiling"]["drop_remote"]
    assert frac["tau"] == 1.0                          # quoted at the knee...
    assert 45 < frac["percent"] < 60                   # ...not at tau 2.0
