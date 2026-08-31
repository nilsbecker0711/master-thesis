#!/usr/bin/env python
r"""
Choose a learning rate from a set of tuning runs — unattended.

    python analysis/pick_lr.py results/tune/*csf*        # -> prints e.g. 0.2

WHY THIS IS NOT `argmax drop_remote`
------------------------------------
For a CONSTRAINED attack, the strongest run is not automatically the best one.
A larger learning rate drives the residual harder against the [0,1] boundary,
the final clamp injects broadband harmonics, and REALISED visibility rises
above the tau that was requested. Selecting on drop_remote alone therefore
rewards whichever learning rate breaks the perceptual constraint hardest, and
calls it better optimisation.

Measured on this project: at lr 0.01 realised visibility sat exactly on tau,
while at lr 0.5 the locally-calibrated visibility reached 1.03 JND — past the
detection threshold the whole family is defined against.

So the rule is:

    among runs whose realised visibility is at or below `--ceiling`,
    take the one with the best mean `--metric`.

That compares learning rates AT EQUAL PERCEPTUAL COST, which is the same
control the frequency probe applies across bands and for the same reason.

If no run qualifies, nothing is silently returned: the script says so on
stderr, falls back to the least-visible run, and exits non-zero so a caller
can react.

UNCONSTRAINED MODES (raw, lap, gan) have no visibility statistic. There the
ceiling cannot bind and the rule degenerates to plain argmax, which is correct
for them — declared here rather than left implicit.

OUTPUT CONTRACT: the chosen learning rate, and nothing else, on STDOUT, so a
shell can capture it. Everything human-readable goes to STDERR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

# Preference order: the locally-calibrated number is the honest one, but it
# only exists if the run recorded it.
VISIBILITY_KEYS = ("final_visibility_local", "final_visibility")


def _read(run_dir: Path):
    r"""
    (config, records) from any of the THREE layouts that produce tuning runs.

    This existed for overfit_population.py only, and silently skipped every
    other producer -- the "no summary.json / no records" branch fired on all of
    them. overfit.py's own docstring says "analysis/pick_lr.py globs for them",
    which was not true of either of its layouts:

      population    summary.json {config, records}          <- the original
      overfit n=1   results.json {config, seed, **record}   <- no summary at all
      overfit n>1   summary.json {n_seeds, per_seed, stats} <- no config key,
                                                                no records key

    Normalising here rather than at three call sites keeps ONE selection rule.
    The rule is the whole point of the script: picking at equal perceptual cost
    rather than argmax, and a second reader would be a second place for that to
    drift.
    """
    s = run_dir / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        if d.get("records"):                       # population
            return d.get("config", {}), d["records"]
        if d.get("per_seed"):                      # overfit, repeated
            # The seed rows carry the metrics but no config -- that lives in
            # each repeat's own results.json, so take it from the first one.
            cfg = {}
            for sub in sorted(run_dir.glob("seed*/results.json")):
                cfg = json.loads(sub.read_text()).get("config", {})
                break
            return cfg, d["per_seed"]
        return None
    r = run_dir / "results.json"                   # overfit, single seed
    if r.exists():
        d = json.loads(r.read_text())
        return d.get("config", {}), [d]
    return None


def load(run_dir: Path):
    got = _read(run_dir)
    if got is None:
        return None
    cfg, recs = got
    if not recs:
        return None
    d = {"config": cfg, "records": recs}

    vis_key = next((k for k in VISIBILITY_KEYS
                    if any(r.get(k) is not None for r in recs)), None)
    vis = (mean(r[vis_key] for r in recs if r.get(vis_key) is not None)
           if vis_key else None)

    return {"dir": run_dir, "lr": cfg.get("lr"), "tag": cfg.get("tag", ""),
            "tau": cfg.get("csf_threshold"), "mode": cfg.get("patch_mode"),
            "n": len(recs), "vis_key": vis_key, "visibility": vis}, d


def _write(out, best, rows, eligible, a, *, edge: bool, fell_back: bool):
    """
    Persist the decision. No-op without --out, so the stdout contract is
    unchanged for anything already capturing it.

    The chosen lr otherwise exists ONLY on stdout, which means the number
    behind a headline result survives as terminal scrollback and nothing else.
    The candidate table goes in too, so the choice can be audited later without
    re-running the sweep.
    """
    if out is None:
        return
    out = Path(out)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lr": best["lr"],
        "metric": a.metric,
        "ceiling": a.ceiling,
        "chosen_run": str(best["dir"]),
        "score": best.get("score"),
        "visibility": best["visibility"],
        "visibility_key": best["vis_key"],
        # True means the grid stopped at the answer, so the answer IS the grid.
        "at_grid_edge": edge,
        # True means NOTHING met the ceiling and this is the least-visible run
        # rather than the best one. A caller that ignores this is quoting a
        # fallback as if it were a choice.
        "fell_back_to_least_visible": fell_back,
        "n_eligible": len(eligible),
        "candidates": [{"lr": r["lr"], "score": r.get("score"),
                        "visibility": r["visibility"], "n": r["n"],
                        "run": r["dir"].name} for r in rows],
    }, indent=2))
    print(f"  written: {out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Pick a learning rate from tuning runs")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--metric", default="drop_remote")
    p.add_argument("--ceiling", type=float, default=1.0,
                   help="max acceptable mean realised visibility, in JND. 1.0 "
                        "is the detection threshold. Ignored for modes that "
                        "record no visibility.")
    p.add_argument("--out", type=Path, default=None,
                   help="also write the decision to this JSON file: the lr, the "
                        "ceiling it was judged against, the full candidate "
                        "table, and whether it landed at a grid edge or fell "
                        "back. Without this the answer exists only on stdout.")
    p.add_argument("--default", type=float, default=None,
                   help="printed if nothing can be read at all, so an "
                        "overnight script still has a usable value")
    a = p.parse_args()

    rows = []
    for d in a.runs:
        got = load(d)
        if got is None:
            print(f"  skip {d} (no results.json/summary.json, or no records)",
                  file=sys.stderr)
            continue
        meta, full = got
        vals = [r[a.metric] for r in full.get("records", [])
                if r.get(a.metric) is not None]
        if not vals or meta["lr"] is None:
            continue
        meta["score"] = mean(vals)
        rows.append(meta)

    if not rows:
        print(f"  no usable tuning runs found", file=sys.stderr)
        if a.default is not None:
            print(a.default)
            return 0
        return 2

    rows.sort(key=lambda r: r["lr"])
    print(f"\n  {'lr':>8s}{'mean ' + a.metric:>18s}{'visibility':>13s}"
          f"{'n':>5s}   run", file=sys.stderr)
    for r in rows:
        v = "n/a" if r["visibility"] is None else f"{r['visibility']:.3f}"
        flag = ("" if r["visibility"] is None or r["visibility"] <= a.ceiling
                else "  OVER CEILING")
        print(f"  {r['lr']:>8g}{r['score']:>18.2f}{v:>13s}{r['n']:>5d}   "
              f"{r['dir'].name}{flag}", file=sys.stderr)

    eligible = [r for r in rows
                if r["visibility"] is None or r["visibility"] <= a.ceiling]

    if not eligible:
        best_vis = min(rows, key=lambda r: r["visibility"])
        print(f"\n  NO RUN met the visibility ceiling {a.ceiling:g} JND. Every "
              f"learning rate tested broke the\n  perceptual constraint, so "
              f"'best' would mean 'broke it hardest'. Falling back to the\n"
              f"  LEAST VISIBLE run (lr={best_vis['lr']:g}, visibility="
              f"{best_vis['visibility']:.3f}) — but the right fix is a lower\n"
              f"  --csf_threshold, not a different learning rate.",
              file=sys.stderr)
        _write(a.out, best_vis, rows, [], a, edge=False, fell_back=True)
        print(best_vis["lr"])
        return 1

    best = max(eligible, key=lambda r: r["score"])
    vis = "n/a" if best["visibility"] is None else f"{best['visibility']:.3f}"
    print(f"\n  chosen: lr={best['lr']:g}  ({a.metric} {best['score']:.2f}, "
          f"visibility {vis})  from {len(eligible)}/{len(rows)} eligible",
          file=sys.stderr)

    # AT AN EDGE OF THE GRID the answer is where the grid stopped, not an
    # optimum, and the two are indistinguishable from the chosen number alone.
    # Said here because this is the last place anyone looks before quoting it.
    edge = len(rows) > 1 and best["lr"] in (rows[0]["lr"], rows[-1]["lr"])
    if edge:
        which = "LOWEST" if best["lr"] == rows[0]["lr"] else "HIGHEST"
        print(f"  WARNING: that is the {which} lr tested, so the sweep is "
              f"truncated.\n           Extend the grid past {best['lr']:g} and "
              f"re-run before quoting it as best.", file=sys.stderr)

    _write(a.out, best, rows, eligible, a, edge=edge, fell_back=False)
    print(best["lr"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
