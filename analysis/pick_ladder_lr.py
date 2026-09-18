#!/usr/bin/env python
r"""
Read an lr ladder, emit the best lr per (optimiser, pixel_param) arm as shell.

    python analysis/pick_ladder_lr.py results/optimiser --glob '*ladder*'

WHY THIS EXISTS
---------------
The optimiser ablation cannot use one lr across its four arms: under
pixel_param='sigmoid' the parameter is a LOGIT and a step of lr moves the pixel
by lr*sigma'(p) (0.25 at the mid-grey init, falling toward zero as it
saturates); under 'direct' the parameter IS the pixel and a step of lr moves it
by exactly lr. Handing both the same number compares two step SIZES and reports
the difference as a step RULE effect.

So each arm gets a ladder and is scored at its own best lr. Doing the pick by
hand puts a human decision between two halves of one job, which is what made
the first version of optimiser_ablation.sh a two-sbatch, come-back-tomorrow
procedure. This makes it one job.

IT READS THE CONFIG, NOT THE DIRECTORY NAME. The names encode the arm, but they
encode it through a tag the caller passes, so a typo in the shell loop would
silently pick the wrong run's lr and nothing downstream would notice. The
config block inside summary.json is what the run actually did.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def rows(root: Path, pattern: str):
    """Every ladder run under `root`, as (optimiser, pixel_param, lr, mean)."""
    out = []
    for s in sorted(root.glob(f"{pattern}/summary.json")):
        try:
            d = json.loads(s.read_text())
            cfg, summ = d["config"], d["summary"]
            if summ.get("n", 0) == 0:
                print(f"  [!] {s.parent.name}: no images finished, skipped",
                      file=sys.stderr)
                continue
            out.append((cfg["optimiser"], cfg["pixel_param"], float(cfg["lr"]),
                        float(summ["distribution"]["mean"]), s.parent.name))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            # A half-written summary from a job that hit its wall clock is the
            # expected failure here. Name it and carry on -- losing one rung of
            # a ladder still leaves a pick, while dying takes the whole job.
            print(f"  [!] {s.parent.name}: unreadable ({e}), skipped",
                  file=sys.stderr)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--glob", default="*ladder*")
    p.add_argument("--key", default="drop_remote",
                   help="recorded for provenance; the ladder runs already "
                        "summarise on this key")
    a = p.parse_args()

    found = rows(a.root, a.glob)
    if not found:
        sys.exit(f"no readable ladder runs under {a.root}/{a.glob}")

    best, names = {}, {"adam": "A", "sign": "C"}
    for opt, pp, lr, mean, name in found:
        k = (opt, pp)
        if k not in best or mean > best[k][1]:
            best[k] = (lr, mean, name)

    print("# lr picked per arm, by mean %s on the ladder sample" % a.key,
          file=sys.stderr)
    for (opt, pp), (lr, mean, name) in sorted(best.items()):
        rung = sorted(m for o, q, _, m, _ in found if (o, q) == (opt, pp))
        # A ladder whose best rung is its TOP rung never turned over, so the
        # arm was still lr-starved and its number is a floor, not its best.
        # That is the one way this ablation produces a false "PGD is worse",
        # so it is called out here rather than left in the logs.
        top = max(l for o, q, l, _, _ in found if (o, q) == (opt, pp))
        edge = " <-- AT LADDER EDGE, widen it" if lr == top else ""
        print(f"#   {opt:4s} {pp:7s}  lr={lr:<8g} mean={mean:+6.2f}  "
              f"(rungs {[round(x, 2) for x in rung]}){edge}", file=sys.stderr)

    # shell-sourceable, on stdout, so the caller does `eval $(... )`
    key = {("adam", "sigmoid"): "LR_A", ("adam", "direct"): "LR_B",
           ("sign", "sigmoid"): "LR_C", ("sign", "direct"): "LR_D"}
    for k, var in key.items():
        if k in best:
            print(f"{var}={best[k][0]:g}")
        else:
            print(f"# {var}: arm ({k[0]}, {k[1]}) missing from the ladder",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
