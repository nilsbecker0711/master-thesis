#!/usr/bin/env python
r"""
Walk results/runs/*/ and flatten every config + result into one CSV.

    python analysis/build_index.py            # -> results/index.csv

WHY: the previous repo nested results by axis
(arch/lr/patch_mode/loss/img/cls), which broke as soon as a new axis appeared —
and untargeted runs ended up filed under `cls-1` because the path template
always included the target class. Four new axes were added in three weeks.

A flat store plus a generated index means adding an axis is a new COLUMN, not a
new directory level. Every table and figure script then reads one dataframe:

    df = pd.read_csv("results/index.csv")
    df.query("patch_mode == 'lap' and arch == 'internimage'")

The index is free because config.json already holds the full argument namespace
— nothing is maintained separately, so it cannot drift out of sync.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Result keys that overfit.py and overfit_population.py write at TOP LEVEL
# rather than nested under "final". Without these the index silently came out
# with config columns and almost no results: FLATTEN below was written against
# train.py's nested shape, and a single-image run has none of those blocks, so
# every drop, flip rate and visibility column landed empty.
TOP_LEVEL = ("clean_all", "clean_remote", "final_all", "final_remote",
             "drop_all", "drop_remote", "best_drop_remote",
             "degraded_after_peak", "any_flip_rate", "target_hit_rate",
             # the perceptual cost the drop was bought at -- a drop column
             # without these invites exactly the comparison pick_lr refuses
             "final_visibility", "final_visibility_local",
             "final_resid_rms", "final_resid_absmax",
             # csf: spectral allocation frozen? raw/lap: sigmoid saturated?
             "final_frac_at_bound", "final_spend_mean", "final_frac_at_clip",
             "placement_dist_from_centre", "placement_on_border",
             "lr_schedule", "backbone_channels", "backbone_active_channels",
             "silhouette_frac", "wall_clock_s")

# Nested keys worth promoting to top-level columns.
FLATTEN = {
    "final": ["clean_all", "clean_rem", "adv_all", "adv_rem", "drop_all",
              "drop_remote", "any_flip_rate", "target_hit_rate"],
    "rationality": ["ASI", "AGI", "ADE", "L2_to_reference"],
}


def flatten_run(run_dir: Path) -> dict | None:
    res_p, cfg_p = run_dir / "results.json", run_dir / "config.json"
    if not res_p.exists():
        return None

    res = json.loads(res_p.read_text())
    row = {"run_id": run_dir.name, "path": str(run_dir)}
    row.update(json.loads(cfg_p.read_text()) if cfg_p.exists()
               else res.get("config", {}))

    for block, keys in FLATTEN.items():
        sub = res.get(block, {}) or {}
        for k in keys:
            if k in sub:
                row[k] = sub[k]

    for k in TOP_LEVEL:
        if k in res:
            row[k] = res[k]

    hist = res.get("history", [])
    row["n_evals"] = len(hist)
    if hist:
        # train.py counts epochs, overfit.py counts steps. Record whichever the
        # run actually has: a `last_step` well below --steps is how a truncated
        # or crashed single-image run shows up in the index instead of looking
        # like a converged one.
        row["epochs_completed"] = hist[-1].get("epoch")
        row["last_step"] = hist[-1].get("step")
    return row


def flatten_seeded(run_dir: Path) -> dict | None:
    """
    A --seeds N > 1 overfit run: summary.json holds {n_seeds, per_seed, stats}
    and the config lives in each repeat's own results.json.

    Reported as ONE row of MEANS, with n_seeds carried so a single-seed row and
    a five-seed row are never mistaken for equally solid. The sd goes in beside
    the mean for the headline metric, because this project has already measured
    a 9.6-point spread across four identical single-image runs -- a mean with no
    spread beside it overstates what one number establishes.
    """
    s_p = run_dir / "summary.json"
    if not s_p.exists():
        return None
    d = json.loads(s_p.read_text())
    # Guard on `stats`, which is what the means are read from. Guarding on
    # per_seed instead dropped the row whenever the raw rows were absent but
    # the aggregate was present -- the run vanished from the index rather than
    # appearing with the numbers it did have.
    if not d.get("stats"):
        return None
    row = {"run_id": run_dir.name, "path": str(run_dir),
           "n_seeds": d.get("n_seeds")}
    for sub in sorted(run_dir.glob("seed*/results.json")):
        row.update(json.loads(sub.read_text()).get("config", {}))
        break
    for k, st in (d.get("stats") or {}).items():
        row[k] = st.get("mean")
        if st.get("sd") is not None:
            row[f"{k}_sd"] = st["sd"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/index.csv")
    a = ap.parse_args()

    rows = [r for d in sorted(Path(a.runs).glob("*")) if d.is_dir()
            for r in [flatten_run(d) or flatten_seeded(d)] if r]
    if not rows:
        raise SystemExit(f"no completed runs under {a.runs}")

    df = pd.DataFrame(rows)
    lead = [c for c in ("run_id", "arch", "patch_mode", "csf_param", "loss_fn",
                        "lr", "csf_threshold", "img_h", "img_w",
                        "drop_remote", "best_drop_remote", "any_flip_rate",
                        "final_visibility_local", "final_frac_at_bound",
                        "target_hit_rate") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} runs -> {a.out}")
    show = [c for c in lead if c in df.columns]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
