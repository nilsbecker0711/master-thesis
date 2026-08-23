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


def load(run_dir: Path):
    s = run_dir / "summary.json"
    if not s.exists():
        return None
    d = json.loads(s.read_text())
    cfg = d.get("config", {})
    recs = d.get("records", [])
    if not recs:
        return None

    vis_key = next((k for k in VISIBILITY_KEYS
                    if any(r.get(k) is not None for r in recs)), None)
    vis = (mean(r[vis_key] for r in recs if r.get(vis_key) is not None)
           if vis_key else None)

    return {"dir": run_dir, "lr": cfg.get("lr"), "tag": cfg.get("tag", ""),
            "tau": cfg.get("csf_threshold"), "mode": cfg.get("patch_mode"),
            "n": len(recs), "vis_key": vis_key, "visibility": vis}, d


def main():
    p = argparse.ArgumentParser(description="Pick a learning rate from tuning runs")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--metric", default="drop_remote")
    p.add_argument("--ceiling", type=float, default=1.0,
                   help="max acceptable mean realised visibility, in JND. 1.0 "
                        "is the detection threshold. Ignored for modes that "
                        "record no visibility.")
    p.add_argument("--default", type=float, default=None,
                   help="printed if nothing can be read at all, so an "
                        "overnight script still has a usable value")
    a = p.parse_args()

    rows = []
    for d in a.runs:
        got = load(d)
        if got is None:
            print(f"  skip {d} (no summary.json / no records)", file=sys.stderr)
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
        print(best_vis["lr"])
        return 1

    best = max(eligible, key=lambda r: r["score"])
    vis = "n/a" if best["visibility"] is None else f"{best['visibility']:.3f}"
    print(f"\n  chosen: lr={best['lr']:g}  ({a.metric} {best['score']:.2f}, "
          f"visibility {vis})  from {len(eligible)}/{len(rows)} eligible",
          file=sys.stderr)
    print(best["lr"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
