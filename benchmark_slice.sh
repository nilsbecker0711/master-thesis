#!/bin/bash
#SBATCH -p dev_gpu_a100_il
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# ═════════════════════════════════════════════════════════════════════════════
#  ONE SLICE of the Nesti/Rossolini benchmark.  Driven by driver_benchmark.sh.
#
#      ./benchmark_slice.sh --list          the config table, machine readable
#      sbatch benchmark_slice.sh <name>     one 30-minute slice of <name>
#
#  Every invocation passes --resume and the FULL --epochs. A slice stops
#  because SLURM kills it at the walltime, never because it was told to do
#  less work: --epochs sets the cosine T_max, so a slice asked for fewer
#  would silently run a different optimisation. train.py checkpoints after
#  every epoch (param, Adam moments, scheduler position, gstep, RNG) and the
#  next slice picks up exactly where this one died. Verified bit-exact
#  against an uninterrupted run in tests/test_train_resume.py.
#
#  THE CONFIG TABLE LIVES HERE, ONCE. The driver reads it back through
#  --list rather than keeping its own copy, so an out_dir cannot drift
#  between the two — the failure mode the population driver has a comment
#  about.
#
# ─────────────────────────────────────────────────────────────────────────────
#  THE PROTOCOL.  Nesti et al., WACV 2022, arXiv:2108.06179, Sec. 4.1-4.2,
#  and its journal extension Rossolini et al., TNNLS 2024.
#
#  THEIRS, reproduced:
#    * Cityscapes at the full 1024x2048 (no downscaling, no crop).
#    * 250 images sampled from the TRAINING split for optimisation.
#    * the ENTIRE 500-image validation split for evaluation.
#    * Adam, 200 epochs.
#    * mIoU and mAcc on pixels OUTSIDE the patch area.
#    * a random-patch column, which under an outside-the-patch metric is the
#      real baseline — the clean number is measured over a different pixel
#      set and is not the comparison.
#
#  WHAT WE CAN CLAIM: "we adopt the DATA PROTOCOL of Nesti et al." — not
#  "their training procedure", which the four deviations below make false.
#
#  OURS, each to be stated in the write-up:
#
#    1. NO EOT. Their non-EOT arm places the patch at the image centre every
#       iteration, which is what this runs. Their EOT arm adds 5% Gaussian
#       noise, 80-120% random scaling and bounded random translation. None of
#       it exists in Patch.apply(), which composites ONE (top,left) with
#       F.pad and expands a single mask across the batch. For universal_csf
#       the scaling half also collides with the budget: _apply_residual
#       REFUSES to resample the residual, because resampling resamples its
#       spectrum and tau then describes a different tensor than the one
#       pasted. Decide that before writing an EOT arm.
#
#    2. ONE PATCH SIZE, FIXED AT scale 0.25 — not their three-rung ladder,
#       and square rather than their 1:2 rectangles (everything here is
#       square by construction: footprint_side() returns one int and
#       patch_budget() builds a size x size envelope).
#
#         theirs      % of frame      ours              % of frame
#         150x300         2.14        --                --
#         200x400         3.82        256x256 @ 0.25    3.125
#         300x600         8.57        --                --
#
#       0.25 is the operating point every other run in this thesis uses, and
#       because scale_ref='height' fixes coverage by aspect ratio alone, it
#       is 3.125% of the frame at 512x1024 AND at 1024x2048. So this row
#       stays comparable to our own body of work; only the absolute pixel
#       count changes with resolution. Patch size already has a systematic
#       table of its own (T19), where nu_min = m_c/S is handled properly —
#       averaging over their ladder here would duplicate it worse.
#       See the GEOMETRY block below.
#
#    3. NOT THEIR LEARNING RATE (0.5, set empirically on an unconstrained
#       pixel patch). universal_csf at lr 0.2 discards the previous residual
#       every step (mean relative change 1.89) with frac_at_bound 0.93 —
#       pinned to the boundary, phase-only; lr 0.01 gives 0.15 and 0.508.
#       Table in universal_csf.sh.
#
#    4. NOT THEIR MODELS. DDRNet/BiSeNet/ICNet are real-time, which is WHY
#       they get native resolution. docs/experiment_log.md: "Native
#       resolution OR the heavy global-attention architectures the ERF
#       hypothesis requires, not both." Broken deliberately for this row,
#       and paid for with --batch_size 1.
#
#  "200 epochs" IS NOT TRANSFERABLE WITHOUT THEIR BATCH SIZE. 200 epochs on
#  250 images is 50,000 steps at batch 1 and 12,500 at batch 4. What is
#  matched here is IMAGE PRESENTATIONS (250 x 200 = 50,000); say that rather
#  than implying step-for-step equivalence. Expect the CSF arm to plateau
#  well before 200 — universal_csf.sh records 20 epochs over 2975 images as
#  ample for a residual pinned to its envelope from step 0. Do not shorten
#  it on that basis: with cosine, --epochs is a SCHEDULE parameter, so a
#  60-epoch run anneals over 15k steps and is a different optimisation, not
#  a truncation of the 200-epoch one.
# ═════════════════════════════════════════════════════════════════════════════

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set BEFORE the config table, which interpolates it, and overridable from
# the environment so --list works on a login node with nothing loaded.
CS="${CS:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes}"
export CS

EPOCHS=200

# ARCH is a variable and lands in the out_dir, so a later b5 run does not
# collide with this one. NOTE: 'segformer' is an ALIAS for segformer_b0 --
# registry.py ends with REGISTRY["segformer"] = REGISTRY["segformer_b0"] --
# so the default here is B0, the smallest MiT. That is what makes 1024x2048
# affordable at all; it is also the weakest version of the "heavy
# global-attention transformer" the ERF hypothesis is about, so if the
# benchmark row is meant to carry that claim it wants b5 and a much longer
# queue. Override with ARCH=segformer_b5 in the environment; sbatch exports
# it to the slice.
ARCH="${ARCH:-segformer_b0}"
OUTROOT="results/runs/bench_nesti/$ARCH"

# ── GEOMETRY: FIXED AT scale 0.25, NOT their size ladder ─────────────────────
# Nesti sweep three patch sizes (2.14 / 3.82 / 8.57% of the frame). We hold
# ONE operating point instead, the 0.25 every other run in this thesis uses.
#
# WHY THAT IS THE BETTER TRADE HERE. scale_ref='height' means the footprint
# is int(H*0.25) and coverage depends only on aspect ratio, which Cityscapes
# fixes at 1:2. So 0.25 is 3.125% of the frame at 512x1024 AND at 1024x2048
# -- the benchmark patch covers exactly as much of the scene as every patch
# in the rest of the results chapter, and the ONLY thing that changes at
# native resolution is the absolute pixel count (256 vs 128 a side). That
# keeps this row comparable to our own body of work, which their ladder
# would have cost us, and patch size already has its own systematic table
# (T19) where nu_min = m_c/S is handled properly.
#
# The comparison to state: we report a fixed operating point against their
# ladder, and 3.125% sits between their 150x300 (2.14%) and 200x400 (3.82%)
# rungs -- closest to the middle one. Do not quote our number against a
# specific rung of theirs as though the sizes matched.
PSCALE=0.25
PSIZE=256              # int(1024 * 0.25); universal_csf refuses a mismatch
GEOM="--patch_size $PSIZE --patch_scale $PSCALE"

# The protocol flags every training config shares. --val_every 50 because
# each validation is 500 images x 2 forwards at native resolution; the final
# 500-image evaluation runs after the loop REGARDLESS, so the headline number
# is never stale. best.pt is then the best of ~5 validations — Nesti report
# the final patch, so quote `final`, and treat best.pt as a diagnostic.
PROTO="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048 \
       --placement center --n_train 250 --train_seed 68 \
       --val_images 500 --val_every 50 --epochs $EPOCHS \
       --optimiser adam --lr_schedule cosine --batch_size 1 \
       --num_workers 16 --loss_fn cospgd --no_diagnostics --resume"

# ── the config table ─────────────────────────────────────────────────────────
# name | out_dir | script | args
#
# Six configs: the clean reference plus five rows at the single 0.25
# operating point. The two _null rows and the _grey row are --lr 0
# --epochs 1: nothing moves, and the post-loop 500-image evaluation is the
# entire point of the run.
#
#   csf       the mode under test, tau 0.25 — the rung every other CSF number
#             in the thesis is quoted at. lr 0.01: the parameter IS the
#             residual and the constraint is a projection, so lr sets movement
#             inside a small ball. NOT the 0.2 that suits the scale-invariant
#             squash parameterisation of mode='csf'.
#   csf_null  the random-patch floor. universal_csf initialises from randn and
#             projects onto the CSF envelope, so at lr 0 the patch stays a
#             random residual at EXACTLY tau — an unoptimised perturbation of
#             equal perceptual cost. The gap to `csf` is what optimisation
#             bought.
#   raw       the unconstrained ceiling, and the arm closest to what Nesti
#             actually optimise. If this lands nowhere near their published
#             drops, the problem is plumbing, not the perceptual budget —
#             which is what makes a null CSF result interpretable.
#   raw_null  random PIXELS. A raw patch initialises at uniform 0.5, so --lr 0
#             alone would measure a GREY SQUARE. --raw_init random is the
#             random patch the literature reports.
#   raw_grey  that grey square, kept deliberately. It separates two things the
#             random row conflates: "a patch of this size at this position
#             perturbs the scene" from "arbitrary high-frequency content
#             perturbs the scene". Remote mIoU excludes the footprint, so a
#             grey square moving the number is acting on pixels it does not
#             cover.
configs () {
  local T="s${PSCALE}"        # the tag: our operating point, not their size

  # The clean dataset reference, so the table is self-contained. Sentinel is
  # its own json rather than a run directory, and the directory is
  # arch-specific so a later b5 run cannot satisfy b0's sentinel (the
  # driver's completion check is a glob inside this directory).
  # clean_baseline.py mkdirs its own parent.
  printf '%s\t%s\t%s\t%s\n' \
    "clean" "results/tables/bench_nesti/$ARCH" "scripts/clean_baseline.py" \
    "--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048 \
     --images all --out results/tables/bench_nesti/$ARCH/clean --tag bench_nesti"

  printf '%s\t%s\t%s\t%s\n' "csf" "$OUTROOT/csf" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode universal_csf --csf_threshold 0.25 --lr 0.01 --tag $T"
  printf '%s\t%s\t%s\t%s\n' "csf_null" "$OUTROOT/csf_null" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode universal_csf --csf_threshold 0.25 --lr 0 --epochs 1 --tag ${T}_null"
  printf '%s\t%s\t%s\t%s\n' "raw" "$OUTROOT/raw" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode raw --lr 0.1 --tag $T"
  printf '%s\t%s\t%s\t%s\n' "raw_null" "$OUTROOT/raw_null" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode raw --raw_init random --lr 0 --epochs 1 --tag ${T}_null"
  printf '%s\t%s\t%s\t%s\n' "raw_grey" "$OUTROOT/raw_grey" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode raw --lr 0 --epochs 1 --tag ${T}_grey"
}

# ── --list: the driver's view of the table ───────────────────────────────────
# Printed BEFORE any module load, so the driver (a 1-CPU cpu-partition job)
# can read it without a GPU or a conda environment.
if [ "${1:-}" = "--list" ]; then
    configs | cut -f1,2
    exit 0
fi

WANT="${1:-}"
if [ -z "$WANT" ]; then
    echo "usage: $0 <config-name> | --list" >&2
    exit 2
fi

echo "slice for '$WANT' on $(hostname), $(date)"
mkdir -p "$ROOT/slurm/benchmark"
exec 1> "$ROOT/slurm/benchmark/${WANT}_${SLURM_JOB_ID:-local}.out"
exec 2> "$ROOT/slurm/benchmark/${WANT}_${SLURM_JOB_ID:-local}.err"

module --ignore_cache load "cuda/11.8"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib

cd "$ROOT" || exit 1
echo "cwd $(pwd)"; python --version

row="$(configs | awk -F'\t' -v n="$WANT" '$1 == n')"
if [ -z "$row" ]; then
    echo "FATAL: no config named '$WANT'. Known:" >&2
    configs | cut -f1 >&2
    exit 2
fi
out="$(printf '%s' "$row" | cut -f2)"
script="$(printf '%s' "$row" | cut -f3)"
args="$(printf '%s' "$row" | cut -f4)"

# clean_baseline.py has no --out_dir/--resume; its own --out carries the path.
if [ "$script" = "scripts/train.py" ]; then
    args="$args --out_dir $out"
fi

echo "=== $WANT ==="
echo "out    : $out"
echo "cmd    : python $script $args"
# shellcheck disable=SC2086
python "$script" $args
rc=$?
echo "slice exited $rc at $(date)"
exit $rc
