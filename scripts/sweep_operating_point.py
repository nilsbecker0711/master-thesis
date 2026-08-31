#!/usr/bin/env python
r"""
The sweep that fixes the operating point of the single-image CSF attack:
RUN LENGTH, LEARNING RATE, and the TAU CURVE — in one directory, with one
collective log, one machine-readable manifest, and one CSV.

    python scripts/sweep_operating_point.py --cityscapes_root $CS \
        --arch segformer_b0 --image 420 --name b0_img420

    python scripts/sweep_operating_point.py ... --dry_run   # cost first

WHAT COMES OUT
--------------
    results/sweeps/<name>/
        sweep.log          every cell's console output, in order, plus the
                           decisions. The collective log.
        sweep.json         the manifest: the grid, every cell with its status,
                           its key metrics and its directory, and the decisions
                           with the rule that produced them. Rewritten after
                           EVERY cell, so a job killed by the scheduler still
                           leaves a readable record of what finished.
        index.csv          analysis/build_index.py over cells/ — one row per
                           cell, config columns and result columns flattened.
                           This is the file a LaTeX table is generated from.
        decisions/lr.json  analysis/pick_lr.py's own decision record.
        cells/<arch>_csf_cospgd_img<N>_<cell tag>/
                           one directory per cell: run.log, config.json,
                           results.json (or summary.json + seed*/ when
                           --seeds > 1), final.pt, final_patch.png.

WHY THIS IS STAGED AND NOT A FULL GRID
--------------------------------------
tau x lr x steps at 7 x 6 x 4 with three seeds is 504 runs, and almost all of
them are uninformative: nobody needs the learning rate at tau = 1.0 and 200
steps. The sweep is a STAR around an incumbent instead — one factor at a time,
plus a small tau x lr block whose only job is to check that the star was
allowed. If the block shows an interaction, the one-factor-at-a-time answers
are not valid and the script says so rather than reporting them anyway.

    stage steps    tau, lr at the incumbent; steps over the ladder
    stage lr       tau at the incumbent; steps at stage 1's answer
    stage tau      lr at stage 2's answer; tau over the ladder
    stage grid     3 x 3 tau x lr, the interaction check
    stage enforce  --csf_enforce nominal vs realised, at the operating point

ORDER MATTERS AND THE REASON IS NOT COSMETIC. analysis/pick_lr.py selects the
learning rate AT EQUAL PERCEPTUAL COST — among runs whose realised visibility
is at or below a ceiling — so the lr answer depends on tau, while realised
visibility at a given tau depends on lr through the clip harmonics that
fit_to_range documents. The two axes are circular under --csf_enforce nominal.
They are NOT circular under --csf_enforce realised, which holds realised
visibility at exactly tau at every step: the ceiling then cannot bind, no
learning rate can buy drop by overspending the budget, and lr becomes a
question about optimisation alone. That is why `realised` is the default here
and why the enforcement stage measures what it costs rather than assuming it.

WHAT THE TAU STAGE DOES *NOT* DO
--------------------------------
It does not pick a tau. tau is the axis the whole family is reported against,
not a hyperparameter with an optimum — csf_patch_procedure.tex, Sec. 12: "the
curve over tau, not any single operating point, is the deliverable". What the
stage reports is the USABLE RANGE: the tau above which the dynamic-range fit
binds, realised visibility stops tracking the request, and tau has ceased to
control anything. Numbers above that must not be quoted as achieved, and the
manifest marks them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.utils import tee_output

REPO = Path(__file__).resolve().parents[1]

# The incumbent learning rate is NOT a constant, because the parameter it steps
# is not the same object under the two parameterisations.
#
#   squash  the parameter is a DIRECTION. render() renormalises to exactly tau,
#           so its magnitude is meaningless and lr sets how fast that direction
#           rotates. Measured at tau 0.25, S = 128: param rms 0.9939, so an
#           Adam step of 0.2 is a 20% step. That is where 0.2 comes from, and
#           every csf number this project has quoted was measured there.
#
#   pgd     the parameter IS the residual, in pixel units, and every rfft bin is
#           clamped to its budget after each step. param rms 0.0272, so the SAME
#           0.2 is a 7.4x step: it overshoots the constraint set on every
#           iteration and is clipped straight back. The scale-equivalent of
#           squash's 0.2 is 0.2 * 0.0272/0.9939 ~= 0.0055, and 0.01 is what
#           universal_csf.sh already uses for this same parameterisation.
#
# Carrying 0.2 across the two is a 36x error and it is SILENT -- the run does
# not fail, it converges somewhere else. lr_sweep.sh spells out the same
# arithmetic and centres its grid on it for the same reason.
#
# This sets only the STARTING POINT of the star. The lr grid still spans 0.001
# to 1.0 under both, deliberately: "lr so large that every step overshoots and
# is clipped back" is bang-bang descent against the boundary, which is what
# classical PGD attacks do and they work. The scale argument says where to
# start, not where the optimum is.
INCUMBENT_LR = {"squash": 0.2, "pgd": 0.01}


# The metrics lifted out of each cell into sweep.json. A DELIBERATELY SHORT
# list: the manifest is a map of the sweep, and index.csv already carries every
# column. drop without its perceptual cost beside it is the exact comparison
# pick_lr.py refuses to make, so the two visibilities are not optional here.
CELL_KEYS = ("drop_remote", "best_drop_remote", "any_flip_rate",
             "final_visibility", "final_visibility_local",
             "final_resid_rms", "final_frac_at_bound", "final_spend_mean",
             "final_frac_at_clip", "degraded_after_peak", "wall_clock_s")


# ─────────────────────────────────────────────────────────────────────────────
#  reading a finished cell
# ─────────────────────────────────────────────────────────────────────────────
def read_cell(run_dir: Path) -> dict | None:
    r"""
    (mean, sd, n) per metric for ONE cell, from either overfit.py layout.

    --seeds 1 writes results.json directly; --seeds N writes summary.json with
    per_seed rows. Normalised here so every stage below sees one shape, for the
    same reason analysis/pick_lr.py normalises its three: a second reader is a
    second place for the selection rule to drift.
    """
    s = run_dir / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        rows = d.get("per_seed") or []
    else:
        r = run_dir / "results.json"
        if not r.exists():
            return None
        rows = [json.loads(r.read_text())]
    if not rows:
        return None

    out = {"n_seeds": len(rows), "dir": str(run_dir)}
    for k in CELL_KEYS:
        vals = [float(x[k]) for x in rows if x.get(k) is not None]
        if not vals:
            continue
        out[k] = {"mean": fmean(vals),
                  "sd": stdev(vals) if len(vals) > 1 else None,
                  "min": min(vals), "max": max(vals)}
    return out


def cell_complete(cells_dir: Path, tag: str, loss: str,
                  image: int) -> Path | None:
    """
    An existing, FINISHED directory for this cell tag, or None.

    Resume is by result rather than by directory: overfit.py's increment_path
    leaves a numbered directory behind for a run that crashed on step three,
    and treating that as done would silently put a hole in the sweep.

    THE LOSS IS PART OF THE MATCH, not decoration. overfit.py names a run
    <arch>_csf_<loss>_img<N>_<tag>, so two losses sharing a cell tag differ
    only in a field the old glob did not look at -- and it sorted REVERSE, so
    the ce cell would have been "resumed" from the cospgd directory and the
    sweep would have reported one loss twice.
    """
    for d in sorted(cells_dir.glob(f"*_{loss}_img{image}_{tag}"),
                    reverse=True):
        if (d / "summary.json").exists() or (d / "results.json").exists():
            return d
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  running a cell
# ─────────────────────────────────────────────────────────────────────────────
def run_cell(a, cells_dir: Path, tag: str, *, loss: str, tau: float,
             lr: float, steps: int, enforce: str, dry: bool) -> dict:
    """
    One overfit.py invocation, streamed into the collective log as it goes.

    The child's stdout is echoed line by line rather than captured and printed
    at the end, so a sweep that dies three hours in has a log of what the last
    cell was doing rather than a truncated buffer. The child ALSO writes its
    own run.log into its own directory; this is the pooled view, that is the
    per-cell one.
    """
    cmd = [sys.executable, str(REPO / "scripts" / "overfit.py"),
           "--arch", a.arch,
           "--cityscapes_root", a.cityscapes_root,
           "--img_h", str(a.img_h), "--img_w", str(a.img_w),
           "--patch_mode", "csf", "--from_image",
           "--loss_fn", loss,
           "--placement", a.placement,
           "--patch_scale", str(a.patch_scale),
           "--image", str(a.image),
           "--csf_threshold", f"{tau:g}",
           "--csf_param", a.csf_param,
           "--csf_enforce", enforce,
           "--lr", f"{lr:g}",
           "--lr_schedule", a.lr_schedule,
           "--steps", str(steps),
           "--seeds", str(a.seeds),
           "--seed", str(a.seed),
           "--log_every", str(a.log_every),
           "--out_root", str(cells_dir),
           "--tag", tag]
    if a.no_diagnostics:
        cmd.append("--no_diagnostics")

    rec = {"tag": tag, "loss": loss, "tau": tau, "lr": lr, "steps": steps,
           "enforce": enforce, "cmd": " ".join(cmd)}

    if dry:
        rec["status"] = "planned"
        print(f"  [plan] {loss:8s} {tag:34s} tau={tau:<6g} lr={lr:<7g} "
              f"steps={steps:<5d} enforce={enforce}")
        return rec

    done = cell_complete(cells_dir, tag, loss, a.image)
    if done is not None and not a.force:
        print(f"\n  [skip] {tag} — already complete at {done}")
        rec.update({"status": "reused"}, **(read_cell(done) or {}))
        return rec

    print(f"\n{'=' * 78}\n  CELL {loss} {tag}   tau={tau:g}  lr={lr:g}  "
          f"steps={steps}  enforce={enforce}\n{'=' * 78}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd=str(REPO))
    for line in proc.stdout:
        print(line, end="")
    code = proc.wait()
    rec["returncode"] = code
    rec["cell_wall_clock_s"] = time.time() - t0

    if code != 0:
        # NOT fatal to the sweep. One architecture that OOMs at 2000 steps
        # should not cost the other twenty cells; the manifest records the
        # failure and every decision below skips the cell rather than reading
        # a half-written result.
        rec["status"] = "failed"
        print(f"\n  CELL FAILED (exit {code}) — continuing; "
              f"decisions will skip it.")
        return rec

    got = cell_complete(cells_dir, tag, loss, a.image)
    if got is None:
        rec["status"] = "no_result"
        return rec
    rec.update({"status": "ok"}, **(read_cell(got) or {}))
    return rec


def _mean(rec: dict, key: str):
    v = rec.get(key)
    return v["mean"] if isinstance(v, dict) else None


def usable(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("status") in ("ok", "reused")
            and _mean(r, "drop_remote") is not None]


# ─────────────────────────────────────────────────────────────────────────────
#  the decisions
# ─────────────────────────────────────────────────────────────────────────────
def decide_steps(recs: list[dict], tol: float) -> dict:
    r"""
    The SHORTEST run length whose drop is within `tol` of the longest tested.

    Not argmax. More steps essentially always buy more drop, so argmax returns
    the end of the ladder every time and measures the ladder rather than the
    attack. The question a run length has to answer is "has it converged", and
    the operational form of that is "does doubling it change the answer".

    THE LADDER CANNOT BE READ OFF ONE LONG RUN, and this is the reason the
    stage costs four runs instead of one: with --lr_schedule cosine the
    scheduler is built with T_max = steps, so a 400-step run anneals to zero by
    step 400 while a 2000-step run is still near its peak lr there. Truncating
    the long run's history at step 400 answers a question nobody asked.
    """
    rows = sorted(usable(recs), key=lambda r: r["steps"])
    if not rows:
        return {"steps": None, "rule": "no usable cells"}
    longest = rows[-1]
    best = _mean(longest, "drop_remote")
    for r in rows:
        if _mean(r, "drop_remote") >= best - tol:
            return {"steps": r["steps"], "reference_steps": longest["steps"],
                    "drop_at_choice": _mean(r, "drop_remote"),
                    "drop_at_longest": best, "tolerance": tol,
                    "at_grid_edge": r["steps"] == rows[-1]["steps"],
                    "rule": f"shortest run within {tol:g} mIoU of the longest "
                            f"tested ({longest['steps']} steps)"}
    return {"steps": longest["steps"], "rule": "fallback: longest tested"}


def decide_tau_range(recs: list[dict], slack: float) -> dict:
    r"""
    The usable tau range: where the REQUEST is still met.

    Under --csf_enforce realised the run holds realised visibility at exactly
    tau — until the dynamic-range fit binds, at which point it cannot, and
    every larger tau produces the same residual as the last feasible one. The
    knee is therefore visible as realised/requested falling away from 1, and
    it is reference-dependent (it is set by the headroom of the image region
    the patch covers), which is why it is measured per sweep rather than
    quoted from Sec. 12's tau ~ 0.4.

    Returns the largest tau still tracking, and marks the rest. Nothing above
    the knee is a valid operating point and nothing above it should appear in
    a results table without that flag attached.
    """
    rows = sorted(usable(recs), key=lambda r: r["tau"])
    marked, feasible = [], []
    for r in rows:
        vis = _mean(r, "final_visibility")
        ratio = None if vis is None or not r["tau"] else vis / r["tau"]
        ok = ratio is not None and abs(ratio - 1.0) <= slack
        marked.append({"tau": r["tau"], "realised_visibility": vis,
                       "realised_over_requested": ratio,
                       "drop_remote": _mean(r, "drop_remote"),
                       "resid_rms": _mean(r, "final_resid_rms"),
                       "frac_at_clip": _mean(r, "final_frac_at_clip"),
                       "tracking": ok})
        if ok:
            feasible.append(r["tau"])
    return {"slack": slack, "ladder": marked,
            "tau_max_tracking": max(feasible) if feasible else None,
            "rule": f"largest tau whose realised visibility is within "
                    f"{slack:.0%} of the request; above it the range fit "
                    f"binds and tau controls nothing"}


def check_interaction(recs: list[dict], best_lr: float, tol: float) -> dict:
    r"""
    Was the one-factor-at-a-time star allowed?

    For each tau in the block, the lr that maximises drop. If they all agree
    with each other and with the star's answer, lr and tau are separable at
    this resolution and the staged sweep is valid. If they disagree by more
    than `tol` in drop, they are not, and the honest report is the block rather
    than the star — stated here so the sweep cannot quietly present an invalid
    answer as a clean one.
    """
    rows = usable(recs)
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau"], []).append(r)

    per_tau, spreads = {}, []
    for tau, rs in sorted(by_tau.items()):
        win = max(rs, key=lambda r: _mean(r, "drop_remote"))
        at_star = [r for r in rs if r["lr"] == best_lr]
        gap = (_mean(win, "drop_remote") - _mean(at_star[0], "drop_remote")
               if at_star else None)
        per_tau[f"{tau:g}"] = {"best_lr": win["lr"],
                               "drop_at_best_lr": _mean(win, "drop_remote"),
                               "drop_at_star_lr": (_mean(at_star[0],
                                                         "drop_remote")
                                                   if at_star else None),
                               "gap_to_star": gap}
        if gap is not None:
            spreads.append(gap)

    worst = max(spreads) if spreads else None
    sep = worst is not None and worst <= tol
    return {"per_tau": per_tau, "star_lr": best_lr, "tolerance": tol,
            "worst_gap_to_star": worst, "separable": sep,
            "verdict": ("lr and tau are separable at this resolution; the "
                        "one-factor-at-a-time answers stand"
                        if sep else
                        "INTERACTION: the best lr moves with tau by more than "
                        f"{tol:g} mIoU. Report the tau x lr block, NOT the "
                        "single lr, and say so in the text.")}


def compare_losses(per_loss: dict) -> dict:
    r"""
    The losses side by side, EACH AT ITS OWN CHOSEN LEARNING RATE.

    This is the only comparison of two objectives that means anything here.
    CosPGD is CE multiplied by a detached cosine in [0, 1]
    (patchreach/losses/adversarial.py), so its gradient is strictly smaller
    than CE's on the same pixel -- scaled by a factor that is itself a function
    of how well the prediction still agrees with the label. Run both at one
    learning rate and the stronger one is whichever had its step size closer to
    right, which is a fact about the sweep and not about the objective.

    The same failure the four-architecture comparison at a single lr had, and
    the reason lr_sweep.sh exists. Recorded with the lr each number was
    measured at, so the table can never be read without it.

    NOTE what this still does not settle: CosPGD's own claim is about
    efficiency across a dataset, and adversarial.py declares two deviations
    from the paper (the cosine is detached, and Adam replaces sign-SGD with
    epsilon-projection). This measures OUR cospgd against OUR ce on ONE image.
    """
    rows = {}
    for loss, d in per_loss.items():
        op = d.get("operating_point") or {}
        rows[loss] = {"lr": op.get("lr"), "steps": op.get("steps"),
                      "drop_remote": d.get("drop_at_operating_point"),
                      "visibility": d.get("visibility_at_operating_point"),
                      "tau_max_tracking": (d.get("tau_range") or {})
                      .get("tau_max_tracking")}
    scored = {k: v for k, v in rows.items() if v["drop_remote"] is not None}
    best = (max(scored, key=lambda k: scored[k]["drop_remote"])
            if scored else None)
    same_lr = len({v["lr"] for v in scored.values()}) == 1 if scored else False
    return {"per_loss": rows, "best": best,
            "rule": "each loss at the lr chosen for it, under the same tau, "
                    "steps and enforcement",
            "note": ("both losses chose the SAME lr, so this comparison is "
                     "also valid at matched step size"
                     if same_lr else
                     "the losses chose DIFFERENT learning rates -- quote both "
                     "lrs beside the drops, never the drops alone")}


# ─────────────────────────────────────────────────────────────────────────────
def parse_grid(s: str, cast):
    return [cast(x) for x in s.replace(",", " ").split()]


def main():
    p = argparse.ArgumentParser(
        description="Staged tau / lr / run-length sweep for the single-image "
                    "CSF patch attack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cityscapes_root", required=True)
    p.add_argument("--arch", default="segformer_b0")
    p.add_argument("--image", type=int, default=420)
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument("--losses", default="cospgd ce",
                   help="objectives to sweep. Each is an OUTER axis and gets "
                        "its own run length, learning rate and tau curve -- "
                        "see the loop in main(). One name here runs the "
                        "sweep exactly as a single-loss sweep.")
    p.add_argument("--placement", default="center")
    p.add_argument("--patch_scale", type=float, default=0.25)
    p.add_argument("--csf_param", default="pgd", choices=["pgd", "squash"])
    p.add_argument("--lr_schedule", default="cosine",
                   choices=["none", "cosine"])
    p.add_argument("--enforce", default="realised",
                   choices=["nominal", "realised"],
                   help="what tau bounds in EVERY cell except the enforce "
                        "stage. 'realised' is what decouples lr from tau -- "
                        "see the module docstring before changing it.")

    p.add_argument("--name", default=None,
                   help="sweep directory name (default: arch_imgN)")
    p.add_argument("--out_root", default="results/sweeps")
    p.add_argument("--stages", default="steps,lr,tau,grid,enforce",
                   help="comma-separated subset, in this order")

    p.add_argument("--steps_grid", default="200 400 1000 2000")
    p.add_argument("--lr_grid",
                   default="0.001 0.003 0.01 0.03 0.1 0.2 0.3 1.0",
                   help="0.2 is ON the grid because every csf number this "
                        "project has quoted was measured at it. Leaving it "
                        "off means the sweep cannot confirm or refute the "
                        "incumbent, only replace it.")
    p.add_argument("--tau_grid", default="0.05 0.1 0.25 0.4 0.5 1.0")
    p.add_argument("--grid_taus", default="0.1 0.25 0.5",
                   help="tau values for the interaction block")
    p.add_argument("--grid_lr_factors", default="0.333 1 3",
                   help="lr multipliers around the chosen lr, for the block")

    p.add_argument("--incumbent_lr", type=float, default=None,
                   help="lr the run-length stage is measured at, before an lr "
                        "has been chosen. Declared rather than hidden: the "
                        "steps answer is conditional on it, which is what the "
                        "interaction block exists to test. DEFAULT RESOLVES "
                        "FROM --csf_param, and must -- see "
                        "INCUMBENT_LR at the top of this file.")
    p.add_argument("--incumbent_tau", type=float, default=0.25,
                   help="tau the steps and lr stages are measured at")

    p.add_argument("--seeds", type=int, default=3,
                   help="repeats per cell. 1 makes every number in this sweep "
                        "a single sample; this project has measured a 9.6 "
                        "mIoU spread across four identical runs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--no_diagnostics", action="store_true", default=True,
                   help="sweep cells skip the figure suite; re-run the chosen "
                        "operating point with --diagnostics for the per-class, "
                        "flow and contestability tables")
    p.add_argument("--diagnostics", dest="no_diagnostics",
                   action="store_false")

    p.add_argument("--ceiling", type=float, default=1.0,
                   help="pick_lr.py visibility ceiling, in JND")
    p.add_argument("--steps_tol", type=float, default=1.0,
                   help="mIoU tolerance for 'converged' in the steps stage")
    p.add_argument("--tau_slack", type=float, default=0.05,
                   help="fractional deviation of realised from requested "
                        "visibility still counted as tracking")
    p.add_argument("--interaction_tol", type=float, default=2.0,
                   help="mIoU gap above which lr and tau are NOT separable")

    p.add_argument("--force", action="store_true",
                   help="re-run cells that already have results")
    p.add_argument("--dry_run", action="store_true",
                   help="print the plan and the cell count, run nothing")
    a = p.parse_args()

    # Resolved here rather than in the parser default, because it depends
    # on another argument. The value reached the log either way.
    if a.incumbent_lr is None:
        a.incumbent_lr = INCUMBENT_LR[a.csf_param]

    name = a.name or f"{a.arch}_img{a.image}"
    root = Path(a.out_root) / name
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    (root / "decisions").mkdir(exist_ok=True)

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]

    # PICK UP WHERE A KILLED JOB STOPPED. The cells were already recoverable
    # from disk -- cell_complete() finds them -- but the DECISIONS were not:
    # they live only in sweep.json, and starting from an empty manifest threw
    # away the chosen run length and learning rate of every stage that had
    # finished. On a scheduler with a four-hour walltime that is the difference
    # between resuming a sweep and restarting one.
    #
    # It also lets the two objectives run as SEPARATE JOBS into one directory:
    # `--losses cospgd` then `--losses ce` under the same --name accumulates
    # both, and the cross-loss comparison fires on the second because it reads
    # the merged decisions rather than only what this invocation measured.
    prev = {}
    sweep_json = root / "sweep.json"
    if sweep_json.exists() and not a.force:
        try:
            prev = json.loads(sweep_json.read_text())
            done = [c for c in prev.get("cells", [])
                    if c.get("status") in ("ok", "reused")]
            if done or prev.get("decisions"):
                print(f"  resuming {sweep_json}: {len(done)} finished cells, "
                      f"decisions for {sorted(prev.get('decisions', {}))}")
        except json.JSONDecodeError:
            # A manifest truncated mid-write by the kill. The cells survive on
            # disk regardless, so this costs the decisions and nothing else.
            print(f"  {sweep_json} is unreadable; starting a fresh manifest")

    man = {"name": name, "config": vars(a), "stages": stages,
           "started": prev.get("started", time.strftime("%Y-%m-%d %H:%M:%S")),
           "resumed": (time.strftime("%Y-%m-%d %H:%M:%S") if prev else None),
           "cells": [c for c in prev.get("cells", [])
                     if c.get("status") in ("ok", "reused")],
           "decisions": prev.get("decisions", {})}

    def save():
        (root / "sweep.json").write_text(json.dumps(man, indent=2))

    # THE STAGES OVERLAP, AND WITHOUT THIS THE OVERLAP IS PAID FOR. The star's
    # centre lies on every stage that passes through it -- (tau 0.25, lr*,
    # steps*) is the last rung of the steps ladder, a rung of the lr ladder, a
    # rung of the tau ladder, the centre of the interaction block AND the
    # realised arm of the enforcement pair -- and the block's factor-1 column
    # is contained in the tau ladder outright. Measured on the default grid
    # that is 14 of 58 cells, 42 of 174 runs.
    #
    # Nothing is lost by running each configuration once: the seeds are the
    # same, so a second run of an identical cell is not an independent repeat
    # of anything. --seeds is what measures the spread.
    seen: dict[tuple, dict] = {}

    # Primed from the resumed manifest so a finished cell is neither re-run nor
    # appended twice. Keyed on the CONFIGURATION, not the tag, which is what
    # makes it work across a stage boundary: the same cell reached from the tau
    # ladder and from the interaction block is one entry.
    for c in man["cells"]:
        if all(k in c for k in ("loss", "tau", "lr", "steps", "enforce")):
            seen[(c["loss"], c["tau"], c["lr"], c["steps"],
                  c["enforce"])] = c

    def cell(loss, tag, **kw):
        key = (loss, kw["tau"], kw["lr"], kw["steps"], kw["enforce"])
        if key in seen:
            prev = seen[key]
            print(f"  [dedup] {loss} {tag}"
                  f"  ==  {prev['tag']}, not re-run")
            return prev
        rec = run_cell(a, cells, tag, loss=loss, dry=a.dry_run, **kw)
        seen[key] = rec
        man["cells"].append(rec)
        save()
        return rec

    with tee_output(root / "sweep.log"):
        print(f"{'#' * 78}\n#  SWEEP {name}\n#  {root}\n"
              f"#  losses: {a.losses}   stages: {', '.join(stages)}   seeds/cell: {a.seeds}\n"
              f"#  csf_param {a.csf_param}   incumbent lr {a.incumbent_lr:g}   incumbent tau {a.incumbent_tau:g}   "
              f"enforce {a.enforce}" + chr(10) +
              f"{'#' * 78}")

        losses = [x.strip() for x in a.losses.replace(",", " ").split()
                  if x.strip()]
        # Seeded with the decisions already in the manifest, so a per-loss job
        # run separately still reaches the cross-loss comparison.
        per_loss = {k: v for k, v in man["decisions"].items()
                    if isinstance(v, dict) and "operating_point" in v}

        # THE LOSS IS AN OUTER AXIS, NOT A CELL. Each objective gets its own
        # run length, its own learning rate and its own tau curve, because the
        # gradient it produces is not on the same scale as the other's: cospgd
        # is CE multiplied by a detached cosine in [0, 1], so at an identical
        # lr the two are taking different-sized steps by construction. Sharing
        # one lr between them would measure the step size and report the
        # objective -- the same error as the four-architecture comparison at a
        # single lr that lr_sweep.sh was written to undo.
        for loss in losses:
            print(f"\n{'*' * 78}\n*  LOSS {loss}\n{'*' * 78}")
            dec = {}
            man["decisions"][loss] = dec
            steps = int(parse_grid(a.steps_grid, int)[-1])
            lr = a.incumbent_lr

            # ── stage 1: run length ─────────────────────────────────────────────
            if "steps" in stages:
                print(f"\n>> STAGE steps — tau {a.incumbent_tau:g}, "
                      f"lr {a.incumbent_lr:g} (incumbent, declared)")
                recs = [cell(loss, f"steps{s}_tau{a.incumbent_tau:g}_lr{lr:g}",
                             tau=a.incumbent_tau, lr=lr, steps=s,
                             enforce=a.enforce)
                        for s in parse_grid(a.steps_grid, int)]
                if not a.dry_run:
                    d = decide_steps(recs, a.steps_tol)
                    dec["steps"] = d
                    steps = d["steps"] or steps
                    print(f"\n  CHOSEN steps = {steps}   ({d['rule']})")
                    if d.get("at_grid_edge"):
                        print("  WARNING: that is the LONGEST run tested, so the "
                              "ladder is truncated and\n           'converged' is "
                              "unproven. Extend --steps_grid and re-run.")
                    save()

            # ── stage 2: learning rate ──────────────────────────────────────────
            if "lr" in stages:
                print(f"\n>> STAGE lr — tau {a.incumbent_tau:g}, steps {steps}, "
                      f"enforce {a.enforce}")
                recs = [cell(loss, f"lr{v:g}_tau{a.incumbent_tau:g}_steps{steps}",
                             tau=a.incumbent_tau, lr=v, steps=steps,
                             enforce=a.enforce)
                        for v in parse_grid(a.lr_grid, float)]
                if not a.dry_run:
                    # THE SELECTION RULE LIVES IN pick_lr.py, and is invoked rather
                    # than reimplemented. Two copies of "best at equal perceptual
                    # cost" is one copy too many; its --out gives the audit trail.
                    dirs = [r["dir"] for r in usable(recs) if r.get("dir")]
                    out = root / "decisions" / f"lr_{loss}.json"
                    pr = subprocess.run(
                        [sys.executable, str(REPO / "analysis" / "pick_lr.py"),
                         *dirs, "--ceiling", str(a.ceiling), "--out", str(out),
                         "--default", str(a.incumbent_lr)],
                        capture_output=True, text=True, cwd=str(REPO))
                    print(pr.stderr, end="")
                    try:
                        lr = float(pr.stdout.strip().splitlines()[-1])
                    except (ValueError, IndexError):
                        lr = a.incumbent_lr
                        print(f"  pick_lr.py returned nothing usable; "
                              f"keeping lr = {lr:g}")
                    dec["lr"] = (json.loads(out.read_text())
                                              if out.exists() else {"lr": lr})
                    print(f"\n  CHOSEN lr = {lr:g}")
                    save()

            # ── stage 3: the tau curve ──────────────────────────────────────────
            if "tau" in stages:
                print(f"\n>> STAGE tau — lr {lr:g}, steps {steps}, "
                      f"enforce {a.enforce}")
                recs = [cell(loss, f"tau{t:g}_lr{lr:g}_steps{steps}",
                             tau=t, lr=lr, steps=steps, enforce=a.enforce)
                        for t in parse_grid(a.tau_grid, float)]
                if not a.dry_run:
                    d = decide_tau_range(recs, a.tau_slack)
                    dec["tau_range"] = d
                    print(f"\n  TAU CURVE — realised/requested per rung:")
                    for row in d["ladder"]:
                        r = row["realised_over_requested"]
                        print(f"    tau {row['tau']:<6g} drop "
                              f"{row['drop_remote']:+7.2f}   realised/requested "
                              f"{'n/a' if r is None else f'{r:.3f}'}"
                              f"{'' if row['tracking'] else '   <- RANGE FIT BINDS'}")
                    print(f"  usable up to tau = {d['tau_max_tracking']}")
                    save()

            # ── stage 4: the interaction check ──────────────────────────────────
            if "grid" in stages:
                print(f"\n>> STAGE grid — tau x lr interaction, steps {steps}")
                recs = []
                for t in parse_grid(a.grid_taus, float):
                    for f in parse_grid(a.grid_lr_factors, float):
                        v = lr * f
                        recs.append(cell(loss, f"grid_tau{t:g}_lr{v:g}_steps{steps}",
                                         tau=t, lr=v, steps=steps,
                                         enforce=a.enforce))
                if not a.dry_run:
                    d = check_interaction(recs, lr, a.interaction_tol)
                    dec["interaction"] = d
                    print(f"\n  {d['verdict']}")
                    save()

            # ── stage 5: what enforcement costs ─────────────────────────────────
            if "enforce" in stages:
                print(f"\n>> STAGE enforce — nominal vs realised at the "
                      f"operating point (tau {a.incumbent_tau:g}, lr {lr:g}, "
                      f"steps {steps})")
                recs = [cell(loss, f"enforce{e}_tau{a.incumbent_tau:g}_lr{lr:g}"
                             f"_steps{steps}",
                             tau=a.incumbent_tau, lr=lr, steps=steps, enforce=e)
                        for e in ("nominal", "realised")]
                if not a.dry_run:
                    got = {r["enforce"]: r for r in usable(recs)}
                    if len(got) == 2:
                        n, r = got["nominal"], got["realised"]
                        dec["enforcement_cost"] = {
                            "drop_nominal": _mean(n, "drop_remote"),
                            "drop_realised": _mean(r, "drop_remote"),
                            "cost_mIoU": _mean(n, "drop_remote")
                                         - _mean(r, "drop_remote"),
                            "visibility_nominal": _mean(n, "final_visibility"),
                            "visibility_realised": _mean(r, "final_visibility"),
                            "rule": "the nominal drop is larger and is NOT "
                                    "quotable with a tau attached: it is measured "
                                    "at whatever visibility the clip produced.",
                        }
                        print(f"\n  enforcement costs "
                              f"{dec['enforcement_cost']['cost_mIoU']:+.2f}"
                              f" mIoU and buys a tau that is actually held")
                    save()


            dec["operating_point"] = {"tau": a.incumbent_tau, "lr": lr,
                                      "steps": steps, "enforce": a.enforce,
                                      "loss": loss}
            # The headline cell for this loss: the operating point itself,
            # taken from the memo rather than looked up by tag. Whichever stage
            # ran it first owns the directory, and after dedup that is not
            # predictably the enforcement stage.
            op = seen.get((loss, a.incumbent_tau, lr, steps, a.enforce))
            if op is not None:
                dec["drop_at_operating_point"] = _mean(op, "drop_remote")
                dec["visibility_at_operating_point"] = _mean(
                    op, "final_visibility")
                dec["operating_point_cell"] = op.get("tag")
            per_loss[loss] = dec
            save()

        if not a.dry_run and len(per_loss) > 1:
            cmp_ = compare_losses(per_loss)
            man["decisions"]["loss_comparison"] = cmp_
            print(f"\n{'*' * 78}\n*  LOSS COMPARISON — each at its own "
                  f"chosen lr\n{'*' * 78}")
            for k, v in cmp_["per_loss"].items():
                d0 = v["drop_remote"]
                print(f"    {k:10s} lr {str(v['lr']):>8s}  steps "
                      f"{str(v['steps']):>5s}  drop "
                      f"{'n/a' if d0 is None else f'{d0:+.2f}'}")
            print(f"    {cmp_['note']}")
            save()

        # ── the CSV every table is generated from ───────────────────────────
        if not a.dry_run:
            man["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save()
            idx = subprocess.run(
                [sys.executable, str(REPO / "analysis" / "build_index.py"),
                 "--runs", str(cells), "--out", str(root / "index.csv")],
                capture_output=True, text=True, cwd=str(REPO))
            print(idx.stdout or idx.stderr)

        n_ok = sum(1 for c in man["cells"] if c.get("status") in
                   ("ok", "reused"))
        print(f"\n{'#' * 78}\n#  {n_ok}/{len(man['cells'])} cells with "
              f"results\n#  {root}/sweep.json   {root}/index.csv\n{'#' * 78}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
