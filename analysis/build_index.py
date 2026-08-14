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

    for k in ("best_drop_remote", "backbone_channels",
              "backbone_active_channels", "silhouette_frac", "wall_clock_s"):
        if k in res:
            row[k] = res[k]

    hist = res.get("history", [])
    row["n_evals"] = len(hist)
    if hist:
        row["epochs_completed"] = hist[-1].get("epoch")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/index.csv")
    a = ap.parse_args()

    rows = [r for d in sorted(Path(a.runs).glob("*")) if d.is_dir()
            for r in [flatten_run(d)] if r]
    if not rows:
        raise SystemExit(f"no completed runs under {a.runs}")

    df = pd.DataFrame(rows)
    lead = [c for c in ("run_id", "arch", "patch_mode", "loss_fn", "img_h",
                        "img_w", "drop_remote", "any_flip_rate",
                        "target_hit_rate") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} runs -> {a.out}")
    show = [c for c in lead if c in df.columns]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
