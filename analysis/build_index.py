#!/usr/bin/env python
"""Walk results/runs/*/ and flatten every config + result into one CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TOP_LEVEL = ("clean_all", "clean_remote", "final_all", "final_remote",
             "drop_all", "drop_remote", "best_drop_remote",
             "degraded_after_peak", "any_flip_rate", "target_hit_rate",
             "final_visibility", "final_visibility_local",
             "final_resid_rms", "final_resid_absmax",
             "final_frac_at_bound", "final_spend_mean", "final_frac_at_clip",
             "placement_dist_from_centre", "placement_on_border",
             "lr_schedule", "backbone_channels", "backbone_active_channels",
             "silhouette_frac", "wall_clock_s")

FLATTEN = {
    "final": ["clean_all", "clean_rem", "adv_all", "adv_rem", "drop_all",
              "drop_remote", "any_flip_rate", "target_hit_rate",
              "clean_acc_all", "adv_acc_all", "drop_acc_all",
              "clean_acc_rem", "adv_acc_rem", "drop_acc_remote"],
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
        row["epochs_completed"] = hist[-1].get("epoch")
        row["last_step"] = hist[-1].get("step")
    return row


def flatten_seeded(run_dir: Path) -> dict | None:
    """A --seeds N > 1 overfit run: summary.json holds {n_seeds, per_seed, stats} and the config lives in each repeat's own results.json."""
    s_p = run_dir / "summary.json"
    if not s_p.exists():
        return None
    d = json.loads(s_p.read_text())
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
