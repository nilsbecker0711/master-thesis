#!/usr/bin/env python
r"""
The ATTACK-SUCCESS / VISIBILITY TRADE-OFF, one panel per architecture.

    python analysis/tradeoff_panel.py results/tradeoff/*/ \
        --out figures/tradeoff_panel.png

    python analysis/tradeoff_panel.py results/tradeoff/*/ \
        --metrics any_flip_rate,drop_remote,final_visibility \
        --x realised --normalise --out figures/tradeoff_realised.png

WHAT IT DRAWS
-------------
Rows are metrics (flip rate, mIoU drop, ...), columns are architectures. In
each cell: the csf attack's effect against the perceptual budget it was given,
with the UNCONSTRAINED arm (--patch_mode raw, same images, same optimiser) as a
horizontal ceiling. The question the figure answers is not "does the attack
work" but "how much of the unconstrained attack survives at a budget a human
cannot see" -- and, for the architectures where the constrained attack
currently does nothing, whether the curve is flat because the budget binds or
flat all the way up to the ceiling because something else does.

The ceiling is what makes the panel readable. A csf curve alone cannot
distinguish "this architecture resists the attack" from "this architecture
resists the CONSTRAINT", because both look like a small number. Run the raw arm
on the same images or the panel is one curve and no claim.

TAU IS A REQUEST, NOT AN OUTCOME
--------------------------------
Above some tau the dynamic-range fit binds: the residual cannot grow further
inside [0,1] on the content it covers, realised visibility stops tracking the
request, and every larger tau produces the same residual as the last feasible
one. This script finds that knee and SHADES EVERYTHING ABOVE IT, by the same
rule scripts/sweep_operating_point.py:decide_tau_range() uses, so the flat
right-hand end of a curve is never read as "the attack saturates" when what
saturated was the budget. Nothing in the shaded region is a valid operating
point.

    --x tau        the requested budget. What you asked for.
    --x realised   the visibility actually measured (final_visibility).
                   The honest axis: it is monotone by construction and the
                   knee cannot hide in it. Use it for the thesis figure and
                   keep --x tau as the diagnostic.

WHAT IT READS
-------------
Any run directory this repository writes, in any of three layouts, told apart
by which files are present:

    summary.json with "records"    overfit_population.py  -> per-image spread,
                                   mean with a bootstrap 95% CI
    summary.json with "per_seed"   overfit.py --seeds N   -> mean +/- 1 sd
    results.json                   overfit.py --seeds 1   -> a single sample,
                                   drawn without an interval and labelled n=1

Directories are classified by their config: --patch_mode csf/universal_csf
becomes a point on the curve at its --csf_threshold, --patch_mode raw becomes
the ceiling. Anything else (lap, gan) is skipped with a message rather than
silently folded into one of the two.

WHAT IT REFUSES TO DO QUIETLY
-----------------------------
A trade-off curve is only a curve if every point differs in tau ALONE. The
loader records the other axes (steps, lr, loss, placement, image set,
csf_param, enforcement) per point and warns, by name, about any that moves
within an architecture. It still draws -- you may have a reason -- but the
warning goes to the console and into the sidecar JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.metrics.population import describe

# The two families this figure is about. lap/gan runs are a different
# constraint and are skipped rather than plotted as though they were csf.
CSF_MODES = ("csf", "universal_csf")
CEILING_MODES = ("raw",)

METRIC_LABEL = {
    "any_flip_rate": "remote pixels flipped (%)",
    "drop_remote": "remote mIoU drop (points)",
    "best_drop_remote": "best remote mIoU drop (points)",
    "target_hit_rate": "remote pixels hit target (%)",
    "final_visibility": "realised visibility (JND)",
    "final_visibility_local": "realised visibility, local contrast (JND)",
}

# The same metrics without their units, for --normalise, where the numbers are
# no longer in those units. Carrying "(%)" onto an axis running 0 to 1 is a
# mislabelled figure, not a cosmetic slip.
METRIC_SHORT = {
    "any_flip_rate": "remote flip rate",
    "drop_remote": "remote mIoU drop",
    "best_drop_remote": "best remote mIoU drop",
    "target_hit_rate": "target hit rate",
    "final_visibility": "realised visibility",
    "final_visibility_local": "realised visibility, local contrast",
}

# The fields that must NOT move between two points of one curve. Printed by
# name when they do -- "the curve mixes lr 0.01 and lr 0.2" is actionable,
# "the curve is inconsistent" is not.
AXES = ("loss_fn", "steps", "lr", "lr_schedule", "placement", "csf_param",
        "csf_enforce", "img_h", "img_w", "patch_scale", "image", "n_images",
        "sample_seed", "csf_lref")


# -----------------------------------------------------------------------------
#  reading a run
# -----------------------------------------------------------------------------
def load_run(d: Path) -> dict | None:
    """(config, rows, kind) for one run directory, or None if it is not one."""
    cfg, rows, kind = None, None, None

    cfg_p = d / "config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text())

    s, r = d / "summary.json", d / "results.json"
    if s.exists():
        blob = json.loads(s.read_text())
        cfg = cfg or blob.get("config")
        if blob.get("records"):
            rows, kind = blob["records"], "population"
        elif blob.get("per_seed"):
            rows, kind = blob["per_seed"], "seeds"
    if rows is None and r.exists():
        blob = json.loads(r.read_text())
        cfg = cfg or blob.get("config")
        rows, kind = [blob], "single"

    if not rows or not cfg:
        return None
    return {"dir": d, "cfg": cfg, "rows": rows, "kind": kind}


def stat(rows: list, key: str, seed: int = 0) -> dict | None:
    r"""
    (mean, lo, hi, n) for one metric over a run's rows.

    THE INTERVAL DEPENDS ON WHAT THE ROWS ARE, and the three cases are not
    interchangeable. A population run's rows are images -- a sample of the
    validation set -- so the bootstrap CI is the right interval and n is large
    enough to earn one. A --seeds run's rows are repeats of ONE image: they
    measure optimiser spread, not scene spread, so the band is +/- 1 sd and
    must not be called a confidence interval. A single run has no interval at
    all and is drawn without one, rather than with a zero-width one that would
    read as a precise measurement.
    """
    vals = [float(x[key]) for x in rows if x.get(key) is not None]
    if not vals:
        return None
    v = np.asarray(vals, dtype=np.float64)
    if v.size >= 8:
        d = describe(v, seed=seed)
        lo, hi = d["ci95_boot"]
        return {"mean": d["mean"], "lo": lo, "hi": hi, "n": int(v.size),
                "band": "bootstrap 95% CI"}
    if v.size > 1:
        m, sd = float(v.mean()), float(v.std(ddof=1))
        return {"mean": m, "lo": m - sd, "hi": m + sd, "n": int(v.size),
                "band": "+/- 1 sd over seeds"}
    return {"mean": float(v[0]), "lo": None, "hi": None, "n": 1,
            "band": "single run, no interval"}


def from_sweep(sweep_dir: Path, metrics: list, seed: int):
    r"""
    The tau ladder of a scripts/sweep_operating_point.py sweep, AT ITS OWN
    OPERATING POINT, plus any raw arm dropped into the same cells/ directory.

    A GLOB OVER cells/ CANNOT DO THIS, and the reason is the sweep's own
    dedup. The star's centre lies on several stages at once, so the rung at the
    incumbent tau is run ONCE and keeps whichever stage's tag reached it first
    -- usually the lr stage's. Globbing cells/*_tau* therefore silently drops
    that rung, while globbing cells/* picks up the whole lr ladder as a dozen
    extra points stacked at one tau.

    So the manifest is read instead: decisions[loss]["operating_point"] names
    the lr, run length and enforcement the sweep chose for that objective, and
    only cells matching it become curve points. That is also the only reading
    under which the curve means anything -- a tau ladder assembled from cells
    at four different learning rates is a picture of the lr grid.
    """
    man_p = sweep_dir / "sweep.json"
    if not man_p.exists():
        return [], [], [(sweep_dir, "no sweep.json -- not a sweep directory")]
    man = json.loads(man_p.read_text())
    cells = sweep_dir / "cells"
    if not cells.is_dir():
        return [], [], [(sweep_dir, "no cells/ directory")]

    dirs = sorted(d for d in cells.iterdir() if d.is_dir())
    curve, ceiling, skipped = collect(dirs, metrics, seed)

    ops = {loss: d["operating_point"]
           for loss, d in (man.get("decisions") or {}).items()
           if isinstance(d, dict) and d.get("operating_point")}
    if not ops:
        print(f"  {sweep_dir.name}: the manifest records no operating point "
              f"(the lr/steps stages did not\n    finish), so every csf cell "
              f"is taken. If the sweep ran more than its tau stage,\n    the "
              f"curve now mixes learning rates -- check the warning below.")
        return curve, ceiling, skipped

    keep = []
    for p in curve:
        ax = p["axes"]
        for loss, op in ops.items():
            if (ax.get("loss_fn") == loss
                    and ax.get("lr") is not None
                    and abs(float(ax["lr"]) - float(op["lr"])) < 1e-12
                    and int(ax.get("steps", -1)) == int(op["steps"])
                    and ax.get("csf_enforce") == op.get("enforce")):
                keep.append(p)
                break
    dropped = len(curve) - len(keep)
    op_txt = "; ".join(f"{k}: lr {v['lr']:g}, {v['steps']} steps, "
                       f"enforce {v['enforce']}" for k, v in ops.items())
    print(f"  {sweep_dir.name}: {len(keep)} cells at the operating point "
          f"({op_txt});\n    {dropped} off-operating-point cells ignored.")
    return keep, ceiling, skipped


def collect(dirs: list, metrics: list, seed: int):
    """Every run directory, sorted into curve points and ceiling points."""
    curve, ceiling, skipped = [], [], []
    for d in dirs:
        run = load_run(d)
        if run is None:
            skipped.append((d, "no results.json / summary.json"))
            continue
        cfg, rows = run["cfg"], run["rows"]
        mode = cfg.get("patch_mode")
        pt = {"dir": str(d), "kind": run["kind"], "arch": cfg.get("arch"),
              "mode": mode, "tag": cfg.get("tag", ""),
              "axes": {k: cfg.get(k) for k in AXES if k in cfg},
              "n_rows": len(rows)}
        for m in set(metrics) | {"final_visibility", "drop_remote"}:
            pt[m] = stat(rows, m, seed=seed)

        if mode in CSF_MODES:
            pt["tau"] = float(cfg.get("csf_threshold"))
            pt["enforce"] = cfg.get("csf_enforce", "nominal")
            curve.append(pt)
        elif mode in CEILING_MODES:
            ceiling.append(pt)
        else:
            skipped.append((d, f"--patch_mode {mode} is neither csf nor raw"))
    return curve, ceiling, skipped


# -----------------------------------------------------------------------------
#  where tau stops controlling anything
# -----------------------------------------------------------------------------
def knee(points: list, slack: float) -> dict:
    r"""
    The largest tau the run actually held, and the rule that decided it.

    TWO RULES, because the enforcement mode changes what "held" can mean.

    enforce=realised   the run rescales every step to put realised visibility
                       AT tau, so the test is realised/requested ~ 1. This is
                       scripts/sweep_operating_point.py:decide_tau_range()'s
                       rule, deliberately identical: two definitions of the
                       usable range is one too many.

    enforce=nominal    the bound is per-bin and realised visibility sits below
                       the request by a factor nobody chose, so a ratio test
                       would flag every rung. The operational question is the
                       same either way -- does asking for more still GET
                       more -- so the fallback is saturation: the last rung
                       whose realised visibility grew by at least `slack` over
                       the rung below it.
    """
    rows = sorted([p for p in points if p.get("final_visibility")],
                  key=lambda p: p["tau"])
    if not rows:
        return {"tau_max_tracking": None,
                "rule": "no realised visibility recorded; nothing to check"}

    modes = {p.get("enforce") for p in rows}
    ladder = [{"tau": p["tau"], "realised": p["final_visibility"]["mean"]}
              for p in rows]

    if modes == {"realised"}:
        ok = []
        for x in ladder:
            x["ratio"] = x["realised"] / max(x["tau"], 1e-12)
            x["tracking"] = abs(x["ratio"] - 1.0) <= slack
            if x["tracking"]:
                ok.append(x["tau"])
        return {"tau_max_tracking": max(ok) if ok else None, "ladder": ladder,
                "rule": f"realised visibility within {slack:.0%} of the "
                        f"request (--csf_enforce realised)"}

    held = ladder[0]["tau"]
    for prev, cur in zip(ladder, ladder[1:]):
        cur["grew"] = cur["realised"] > prev["realised"] * (1.0 + slack)
        if not cur["grew"]:
            break
        held = cur["tau"]
    modestr = "/".join(sorted(str(m) for m in modes))
    return {"tau_max_tracking": held, "ladder": ladder,
            "rule": f"last rung whose realised visibility grew by more than "
                    f"{slack:.0%} over the rung below (--csf_enforce "
                    f"{modestr})"}


def dedup_identical(points: list) -> tuple:
    r"""
    Collapse points that are the SAME CONFIGURATION at the same tau.

    The sweep dedups its overlapping stages, so this should be a no-op on a
    clean sweep -- but a --force re-run, or a sweep resumed after its manifest
    was lost, leaves two directories holding the same cell, and a curve drawn
    through both zig-zags between two samples of one condition. Identical
    configurations are one point; the run with the most rows behind it wins.

    Points that share a tau but DIFFER in some axis are left alone: those are
    two conditions, the caller warns about them by name, and silently keeping
    one would be choosing a result.
    """
    by_key: dict = {}
    for p in points:
        key = (p["tau"], json.dumps(p["axes"], sort_keys=True, default=str))
        prev = by_key.get(key)
        if prev is None or p["n_rows"] > prev["n_rows"]:
            by_key[key] = p
    kept = list(by_key.values())
    return kept, len(points) - len(kept)


def axis_drift(points: list) -> dict:
    """Fields that move between points of one curve. Empty is what you want."""
    drift = {}
    for k in AXES:
        vals = {json.dumps(p["axes"].get(k)) for p in points if k in p["axes"]}
        vals = {v for v in vals if v != "null"}
        if len(vals) > 1:
            drift[k] = sorted(json.loads(v) for v in vals)
    return drift


# -----------------------------------------------------------------------------
#  the figure
# -----------------------------------------------------------------------------
def draw(by_arch: dict, metrics: list, x_key: str, normalise: bool,
         slack: float, out: Path, title, ceiling_pick: str,
         share_y: bool) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = list(by_arch)
    nrow, ncol = len(metrics), len(archs)

    # THE LEGEND IS FIGURE-LEVEL, so its labels cannot carry per-column
    # numbers. Previously it lived inside the top-left cell, which made it one
    # architecture's legend describing three, and cost that cell the corner the
    # annotation now uses. Sample sizes move to the column titles, where they
    # belong to the column they describe; anything that happens to be uniform
    # across the whole figure is folded into the shared label instead.
    bands = {p[m]["band"] for d in by_arch.values() for p in d["curve"]
             for m in metrics if p.get(m)}
    n_ceil = {len(d["ceiling"]) for d in by_arch.values()}
    band_txt = f", {next(iter(bands))}" if len(bands) == 1 else ""
    ceil_txt = (f", strongest of {next(iter(n_ceil))}"
                if len(n_ceil) == 1 and next(iter(n_ceil)) > 1 else "")
    CSF_LABEL = f"csf (CSF-bounded residual){band_txt}"
    CEIL_LABEL = f"unconstrained (raw){ceil_txt}"
    SHADE_LABEL = "tau no longer held (range fit binds)"
    unnormalised: list = []
    # sharey="row" BY DEFAULT, because the columns are architectures and the
    # comparison the panel exists for is across them. Per-axis autoscaling
    # makes a 25-point drop and a 55-point drop draw the same curve, which is
    # the one reading of this figure that must not be possible by accident.
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             sharey=("row" if share_y else False),
                             figsize=(4.3 * ncol, 3.1 * nrow))
    notes = {}

    for j, arch in enumerate(archs):
        pts = sorted(by_arch[arch]["curve"], key=lambda p: p["tau"])
        ceil = by_arch[arch]["ceiling"]
        k = knee(pts, slack)
        notes[arch] = {"knee": k, "axis_drift": axis_drift(pts),
                       "n_curve_points": len(pts),
                       "n_ceiling_runs": len(ceil)}

        for i, m in enumerate(metrics):
            ax = axes[i][j]
            rows = [p for p in pts if p.get(m)]
            if x_key == "realised":
                rows = [p for p in rows if p.get("final_visibility")]
                xs = [p["final_visibility"]["mean"] for p in rows]
            else:
                xs = [p["tau"] for p in rows]

            # THE CEILING FIRST, because --normalise needs its denominator.
            cy = None
            cand = [c for c in ceil if c.get(m)]
            if cand:
                if ceiling_pick == "best":
                    best = max(cand, key=lambda c: c[m]["mean"])
                else:
                    best = max(cand, key=lambda c: c[m]["n"])
                cy = best[m]["mean"]
                den = cy if (normalise and abs(cy) > 1e-9) else 1.0
                ax.axhline(cy / den, color="#c44e52", lw=1.6, ls="--",
                           label=CEIL_LABEL, zorder=1)
                if best[m]["lo"] is not None:
                    ax.axhspan(best[m]["lo"] / den, best[m]["hi"] / den,
                               color="#c44e52", alpha=0.12, zorder=0)

            den = cy if (normalise and cy and abs(cy) > 1e-9) else 1.0
            if normalise and den == 1.0:
                # A column with no ceiling stays in raw units while its
                # neighbours become fractions -- and the row shares a y axis,
                # so the two are drawn on one scale. Silently unreadable.
                unnormalised.append(f"{arch}/{m}")
            if rows:
                ys = np.array([p[m]["mean"] / den for p in rows])
                lo = np.array([(p[m]["lo"] if p[m]["lo"] is not None
                                else p[m]["mean"]) / den for p in rows])
                hi = np.array([(p[m]["hi"] if p[m]["hi"] is not None
                                else p[m]["mean"]) / den for p in rows])
                ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys],
                            marker="o", ms=5, lw=1.8, capsize=3,
                            color="#4c72b0", ecolor="#4c72b0", elinewidth=1,
                            zorder=3, label=CSF_LABEL)

            # EVERYTHING ABOVE THE KNEE IS SHADED. Those runs happened and are
            # still drawn, but a flat tail inside the shading is the budget
            # saturating, not the attack.
            if x_key == "tau" and k.get("tau_max_tracking") and xs:
                tmax = k["tau_max_tracking"]
                if tmax < max(xs):
                    ax.axvspan(tmax, max(xs) * 1.15, color="#888", alpha=0.13,
                               zorder=0, label=SHADE_LABEL)

            # 1 JND IS THE ONLY LANDMARK ON EITHER AXIS. Both are visibility
            # in JND, and 1.0 is the detection threshold -- the same number
            # analysis/pick_lr.py defaults its ceiling to. Left of it the
            # perturbation is below threshold, which is the entire claim of
            # this attack family; right of it the figure is measuring how
            # strong a VISIBLE perturbation can be, which is a different
            # question. Drawn so the reader does not have to hold that
            # boundary in their head while reading a log axis.
            if xs and min(xs) < 1.0 < max(xs) * 1.15:
                ax.axvline(1.0, color="#555", lw=1.0, ls=":", zorder=1,
                           label="1 JND (detection threshold)")

            # LOG ON BOTH AXES CHOICES. Requested tau spans 0.05 to 16 and
            # realised visibility spans the same two decades, because it is
            # the same quantity measured instead of asked for. On a linear
            # realised axis every rung below 1 JND -- which is every rung that
            # is actually invisible, i.e. the entire operating range -- piles
            # up against the left spine.
            ax.set_xscale("log")
            if i == 0:
                ns = {p[mm]["n"] for p in pts for mm in metrics if p.get(mm)}
                ax.set_title(arch + (f"   (n={next(iter(ns))})"
                                     if len(ns) == 1 else ""), fontsize=11)
            if j == 0:
                ax.set_ylabel(
                    f"{METRIC_SHORT.get(m, m)}\n(fraction of own ceiling)"
                    if normalise else METRIC_LABEL.get(m, m), fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel("requested tau (JND)" if x_key == "tau"
                              else "realised visibility (JND)", fontsize=9)
            ax.grid(alpha=0.25, lw=0.6)
            # EVERY ROW KEEPS ITS TICK LABELS. sharex hides them on all but the
            # bottom row by default, which leaves the middle rows as curves
            # against an unlabelled axis -- the reader has to count gridlines
            # down to the bottom row to find out what x is. The rows are
            # different METRICS, not different slices of one plot, so each has
            # to be readable where it sits. The axis NAME stays on the bottom
            # row only: it is the same axis three times over and repeating the
            # words adds nothing the numbers do not already say.
            ax.tick_params(labelsize=8, labelbottom=True)

            # The number the figure exists to produce: how much of the
            # unconstrained attack survives at the strongest budget that was
            # actually held.
            held = [p for p in rows
                    if k.get("tau_max_tracking") is not None
                    and p["tau"] <= k["tau_max_tracking"]]
            if cy and abs(cy) > 1e-9 and held:
                top = max(held, key=lambda p: p["tau"])
                frac = 100.0 * top[m]["mean"] / cy
                notes[arch].setdefault("frac_of_ceiling", {})[m] = {
                    "tau": top["tau"], "value": top[m]["mean"],
                    "ceiling": cy, "percent": frac}
                # BOTTOM RIGHT, not top left. These curves rise from the
                # bottom left, and the ceiling line they are quoted against
                # runs across the TOP -- which is exactly where the annotation
                # used to sit, printed over the line it refers to. Under the
                # plateau on the right is the one region of an increasing
                # curve that is reliably empty.
                ax.annotate(f"{frac:.0f}% of ceiling\nat tau={top['tau']:g}",
                            xy=(0.97, 0.04), xycoords="axes fraction",
                            fontsize=8, ha="right", va="bottom", color="#333",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="none", alpha=0.75))

    if unnormalised:
        print(f"\n  WARNING: --normalise, but no unconstrained arm for "
              f"{', '.join(unnormalised)}.")
        print("    Those panels are in RAW UNITS while their neighbours are "
              "fractions, and the row")
        print("    shares one y axis. Run the raw arm, or drop --normalise.")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    # ONE LEGEND, UNDER THE WHOLE FIGURE, built from whichever artists actually
    # got drawn -- a panel with no ceiling arm contributes no ceiling entry,
    # and a figure where tau never stopped tracking gets no shading entry
    # rather than a key to a band that is not there. Placed after
    # tight_layout and below the axes; savefig's tight bbox picks it up.
    handles, labels = [], []
    for row in axes:
        for ax in row:
            for h, lb in zip(*ax.get_legend_handles_labels()):
                if lb not in labels:
                    handles.append(h)
                    labels.append(lb)
    if handles:
        fig.legend(handles, labels, fontsize=8, frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, 0.02),
                   ncol=min(len(handles), 4))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return notes


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Attack-success / visibility trade-off panel, one column "
                    "per architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("runs", nargs="*", default=[],
                   help="run directories (globs are fine). csf runs become "
                        "curve points at their --csf_threshold; raw runs "
                        "become the unconstrained ceiling.")
    p.add_argument("--sweep", action="append", default=[], metavar="DIR",
                   help="a scripts/sweep_operating_point.py directory (the "
                        "one holding sweep.json). Repeatable, once per "
                        "architecture. Its tau ladder is read AT THE "
                        "OPERATING POINT the manifest records -- do not glob "
                        "its cells/ by hand, see from_sweep().")
    p.add_argument("--out", default="figures/tradeoff_panel.png",
                   help="PNG path. A .pdf, a .csv of the plotted points and a "
                        ".json of the knees are written beside it.")
    p.add_argument("--metrics", default="any_flip_rate,drop_remote",
                   help="one row per metric. Add final_visibility to show the "
                        "budget saturating in the same figure.")
    p.add_argument("--x", default="tau", choices=["tau", "realised"],
                   dest="x_key",
                   help="tau: what was requested. realised: what was measured "
                        "(final_visibility) -- the honest axis.")
    p.add_argument("--archs", default=None,
                   help="comma-separated order/filter for the columns. "
                        "Default: every arch found, alphabetically.")
    p.add_argument("--normalise", action="store_true",
                   help="plot each curve as a fraction of its own "
                        "architecture's unconstrained ceiling")
    p.add_argument("--ceiling_pick", default="best",
                   choices=["best", "largest_n"],
                   help="which raw run is the ceiling when several exist. "
                        "'best' treats it as the upper bound it is meant to "
                        "be, and is best-of-N -- the count is printed in the "
                        "legend so it cannot be quoted without it.")
    p.add_argument("--slack", type=float, default=0.05,
                   help="fractional deviation still counted as tau being held")
    p.add_argument("--free_y", dest="share_y", action="store_false",
                   default=True,
                   help="let every panel autoscale its own y. The default "
                        "shares y across a row so the architectures are "
                        "actually comparable.")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--title", default=None)
    a = p.parse_args()

    dirs = sorted({Path(x).resolve() for x in a.runs if Path(x).is_dir()})
    sweeps = [Path(x).resolve() for x in a.sweep]
    if not dirs and not sweeps:
        print("no run directories matched, and no --sweep given",
              file=sys.stderr)
        return 2
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]

    curve, ceiling, skipped = collect(dirs, metrics, a.seed)
    for s in sweeps:
        c, ce, sk = from_sweep(s, metrics, a.seed)
        curve += c
        ceiling += ce
        skipped += sk
    for d, why in skipped:
        print(f"  [skip] {d.name}: {why}")
    if not curve:
        print("no csf runs among the directories given -- nothing to plot "
              "against tau", file=sys.stderr)
        return 2

    archs = sorted({p["arch"] for p in curve})
    if a.archs:
        want = [x.strip() for x in a.archs.split(",") if x.strip()]
        missing = [x for x in want if x not in archs]
        if missing:
            print(f"  no csf runs for: {', '.join(missing)}", file=sys.stderr)
        archs = [x for x in want if x in archs]
    by_arch = {}
    for arch in archs:
        pts, collapsed = dedup_identical(
            [p for p in curve if p["arch"] == arch])
        if collapsed:
            print(f"  {arch}: {collapsed} duplicate cell(s) collapsed "
                  f"(same tau, identical configuration)")
        by_arch[arch] = {"curve": pts,
                         "ceiling": [c for c in ceiling if c["arch"] == arch]}

    for arch, d in by_arch.items():
        taus = sorted(p["tau"] for p in d["curve"])
        print(f"\n  {arch}: {len(d['curve'])} csf points at tau "
              f"{', '.join(f'{t:g}' for t in taus)}")
        if not d["ceiling"]:
            print("    NO UNCONSTRAINED ARM. The panel will have no ceiling, "
                  "so it can show that\n    the attack is weak but not "
                  "whether the CONSTRAINT is what weakened it.\n"
                  "    Run --patch_mode raw on the same images.")
        dupes = sorted({t for t in taus if taus.count(t) > 1})
        if dupes:
            # TWO POINTS AT ONE TAU IS NOT A CURVE, it is two conditions drawn
            # as one. Almost always a glob that swept up an lr ladder or an
            # enforcement arm; the drift warning below names which field moved.
            print(f"    WARNING: {len(dupes)} tau value(s) appear more than "
                  f"once ({', '.join(f'{t:g}' for t in dupes)}). The line "
                  f"will zig-zag through them.")
        drift = axis_drift(d["curve"])
        if drift:
            print(f"    WARNING: these points differ in more than tau -- "
                  f"{', '.join(drift)}.")
            for key, vals in drift.items():
                print(f"      {key}: {vals}")
            print("    The figure attributes those differences to tau. Fix "
                  "them, or say so in the caption.")

    out = Path(a.out)
    notes = draw(by_arch, metrics, a.x_key, a.normalise, a.slack, out,
                 a.title, a.ceiling_pick, a.share_y)

    rows = []
    for arch, d in by_arch.items():
        for pt in sorted(d["curve"], key=lambda q: q["tau"]) + d["ceiling"]:
            row = {"arch": arch, "mode": pt["mode"], "tau": pt.get("tau"),
                   "kind": pt["kind"], "n": pt["n_rows"], "dir": pt["dir"],
                   **pt["axes"]}
            for m in set(metrics) | {"final_visibility"}:
                if pt.get(m):
                    row[m] = pt[m]["mean"]
                    row[f"{m}_lo"] = pt[m]["lo"]
                    row[f"{m}_hi"] = pt[m]["hi"]
            rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".json").write_text(json.dumps(
        {"metrics": metrics, "x": a.x_key, "slack": a.slack,
         "normalised": a.normalise, "ceiling_pick": a.ceiling_pick,
         "per_arch": notes}, indent=2, default=str))

    print("")
    for arch, n in notes.items():
        k = n["knee"]
        print(f"  {arch}: tau held up to {k.get('tau_max_tracking')}  "
              f"({k['rule']})")
        for m, f in (n.get("frac_of_ceiling") or {}).items():
            print(f"    {m}: {f['percent']:.0f}% of the unconstrained ceiling "
                  f"at tau={f['tau']:g} "
                  f"({f['value']:.2f} vs {f['ceiling']:.2f})")
    print(f"\n  -> {out}\n  -> {out.with_suffix('.pdf')}"
          f"\n  -> {out.with_suffix('.csv')}\n  -> {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
