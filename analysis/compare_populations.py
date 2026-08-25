#!/usr/bin/env python
r"""
Paired comparison of population runs — the script that turns two runs into a
result.

    python analysis/compare_populations.py \
        results/population/segformer_csf_cospgd_n100_center \
        results/population/segformer_csf_cospgd_n100_gradcam

    python analysis/compare_populations.py --baseline 0 runs/*/     # fan out

WHY NOT JUST READ THE TWO MEANS
-------------------------------
Because image-to-image variance swamps every effect this project measures.
semantic.py records contestability at 4.2% on one image and 24.7% on another;
the geometric factor barely moves between the same two. An unpaired comparison
of two means buries a 2-point placement effect under a 10-point scene effect
and needs ~100 images per arm to dig it back out. The paired difference cancels
the scene, so the same images under two conditions resolve the effect directly.

That is only valid if both runs covered the SAME images. They do when launched
with the same --images/--n_images/--sample_seed; this script intersects on the
image index and says loudly what it had to drop.

WHAT TO COMPARE
---------------
One axis at a time, everything else held fixed:

  placement   --placement center      vs  gradcam
              "does putting the patch on the sensitivity hotspot help, and by
              how much?" NOTE this is also the repair to an existing confound:
              the conditional generator defaults to --gen_placement gradcam
              while baseline B (overfit.py) defaults to --placement center, so
              every generator-vs-B comparison in block F currently mixes the
              attack family with the placement policy.

  loss        --loss_fn ce            vs  cospgd
              "what does the cosine weighting actually buy?" CosPGD's own
              claim. adversarial.py already declares two deviations from the
              paper (the cosine is detached, and Adam replaces sign-SGD with
              epsilon-projection), so this measures OUR cospgd, not theirs —
              say so in the writeup.

  constraint  --patch_mode raw        vs  csf
              the cost of invisibility, as a fraction of the unconstrained
              ceiling on the same images.

Comparing two axes at once produces a number nobody can attribute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.metrics import population as pop_mod


def load(run_dir: Path):
    """(label, records, summary) from a population run directory."""
    s = run_dir / "summary.json"
    if not s.exists():
        raise SystemExit(
            f"{run_dir} has no summary.json — is it a population run "
            f"directory? (overfit_population.py writes one at the end; a run "
            f"killed at the walltime has records.jsonl but no summary yet)")
    with open(s) as f:
        d = json.load(f)
    cfg = d.get("config", {})
    label = cfg.get("tag") or run_dir.name
    return label, d.get("records", []), d, cfg


def _q_label(cfg) -> str:
    """The Tsallis arm's q, as it is actually configured."""
    if cfg.get("tsallis_schedule", "const") == "const":
        return f"{cfg.get('tsallis_q', 0.0):g}"
    return (f"{cfg.get('tsallis_q_start', -2.0):g}"
            f"->{cfg.get('tsallis_q_end', 1.0):g}")


def describe_arm(label, cfg, log=print):
    """Print the knobs that define an arm, so a mismatched pair is visible."""
    log(f"  {label:<28s} "
        f"arch={cfg.get('arch')} mode={cfg.get('patch_mode')} "
        f"loss={cfg.get('loss_fn')} placement={cfg.get('placement')}"
        + (f"(margin {cfg.get('placement_margin')})"
           if cfg.get('placement') == 'gradcam' else "")
        + (f" q={_q_label(cfg)}" if cfg.get('loss_fn') == 'tsallis' else "")
        + f" tau={cfg.get('csf_threshold')} n={len(cfg.get('images', []))} "
          f"seed={cfg.get('sample_seed')}")


def warn_if_multiple_axes_differ(cfg_a, cfg_b, la, lb, log=print):
    """
    A paired difference attributes an effect to whatever changed. If two things
    changed, it attributes it to both, and the number is uninterpretable.
    """
    # tsallis_* included: a q sweep varies ONE axis, and without these keys
    # a pair that differs in q AND in something else passes the check
    # silently — the exact failure this function exists to catch.
    axes = ["arch", "patch_mode", "loss_fn", "placement", "csf_threshold",
            "patch_scale", "patch_size", "steps", "lr", "target_class",
            "tsallis_q", "tsallis_schedule", "tsallis_q_start",
            "tsallis_q_end"]
    diff = [k for k in axes if cfg_a.get(k) != cfg_b.get(k)]
    if len(diff) > 1:
        log(f"\n  WARNING: {la} and {lb} differ on MORE THAN ONE axis: "
            f"{', '.join(diff)}.")
        log(f"           A paired difference cannot attribute the effect to "
            f"any one of them.")
    elif not diff:
        log(f"\n  WARNING: {la} and {lb} differ on no recorded axis — "
            f"comparing a run against itself?")
    if cfg_a.get("sample_seed") != cfg_b.get("sample_seed"):
        log(f"  WARNING: different --sample_seed ({cfg_a.get('sample_seed')} "
            f"vs {cfg_b.get('sample_seed')}). The pairing will fall back to "
            f"whatever images happen to overlap.")
    return diff


def main():
    p = argparse.ArgumentParser(
        description="Paired comparison of population runs")
    p.add_argument("runs", nargs="+", type=Path,
                   help="population run directories (each with summary.json)")
    p.add_argument("--baseline", type=int, default=0,
                   help="index into `runs` used as the A arm for every "
                        "comparison")
    p.add_argument("--key", default="drop_remote",
                   help="metric compared. drop_remote is the result; "
                        "any_flip_rate and best_drop_remote also work.")
    p.add_argument("--out_dir", default=None,
                   help="where figures and comparison.json go "
                        "(default: alongside the baseline run)")
    a = p.parse_args()

    arms = [load(r) for r in a.runs]
    base_label, base_records, _, base_cfg = arms[a.baseline]

    out_dir = Path(a.out_dir) if a.out_dir else (
        a.runs[a.baseline] / "comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f" ARMS   (comparing everything against [{a.baseline}] {base_label})")
    print(f"{'=' * 72}")
    for k, (label, _, _, cfg) in enumerate(arms):
        describe_arm(f"[{k}] {label}", cfg)

    out = []
    for k, (label, records, _, cfg) in enumerate(arms):
        if k == a.baseline:
            continue
        warn_if_multiple_axes_differ(base_cfg, cfg, base_label, label)
        res = pop_mod.paired_compare(base_records, records, key=a.key,
                                     label_a=base_label, label_b=label)
        if res.get("n"):
            fig = out_dir / f"paired_{base_label}_vs_{label}.png".replace(
                "/", "_")
            pop_mod.plot_paired(base_records, records, fig, key=a.key,
                                label_a=base_label, label_b=label,
                                title=f"{label} vs {base_label}")
            res["figure"] = str(fig)
        out.append(res)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump({"baseline": base_label, "key": a.key,
                   "runs": [str(r) for r in a.runs],
                   "comparisons": out}, f, indent=2)
    print(f"\n  -> {out_dir}/")


if __name__ == "__main__":
    main()
