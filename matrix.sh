#!/bin/bash
#SBATCH -p dev_gpu_a100_il     # one arm per array task; whole grid -> gpu_a100_il + 06:00:00
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:30:00            # Time limit (an arm is ~4-8 min; 30 is the dev cap)
#SBATCH --mem=400000           # Memory request (1024x2048 + slide needs room)
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)
##SBATCH --output=slurm/attack_%J_%j_%a.out
##SBATCH --error=slurm/attack_%J_%j_%a.err

# ═══════════════════════════════════════════════════════════════════════════
#  T17 — the transfer matrix, ONE INVOCATION PER ARM
# ═══════════════════════════════════════════════════════════════════════════
#
# An arm is (loss, patch mode, tau), and that is the whole interface:
#
#   --loss_fn L --csf --tau T  ->  results/overfit_464/L/<arch>/<arch>_csf_L_img464_tT/best.pt
#   --loss_fn L                ->  results/overfit_464/L/<arch>/<arch>_raw_L_img464/best.pt
#
# NOTHING ELSE IS PASSED, on purpose. The architecture list is whatever
# registry-known subdirectories exist under results/overfit_464/<loss>/, and
# the evaluation resolution / scale / inference mode are read out of the source
# runs' own recorded config. So this file cannot drift out of sync with
# overfit.sh: if the 464 block was optimised at 1024x2048 with --inference
# auto, that is what the matrix evaluates at, and a block whose runs disagree
# with each other is a hard error at startup instead of a wrong table.
#
# Each invocation writes results/matrix_464/<loss>/<arm>/matrix.json plus the
# printed tables:
#     img464 : diagonal = white box (READ from the overfit run), off = MODEL transfer
#     img42  : diagonal = IMAGE transfer,                        off = JOINT transfer
#
# set -uo pipefail, NOT -e: one missing arm must not kill the rest of the grid.

set -uo pipefail

echo "Running on $(hostname)"
echo "Date: $(date)"

module --ignore_cache load "cuda/11.8"

# initialize YOUR conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=${CS:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes}

echo "Python path:"
which python
python -c "import sys; print(sys.executable)"
echo "Python version:"
python --version

# ── the grid ────────────────────────────────────────────────────────────────
# Narrowed to the one arm that is actually on disk. Widen from the command
# line as the 464 block fills in, e.g.
#   LOSSES="ce cospgd cos_margin" TAUS="0.05 0.25 0.5" DO_RAW=1 sbatch --array=0-11 matrix.sh
# TAUS MUST BE WRITTEN AS THE DIRECTORY NAME SPELLS IT. tau is carried as a
# string, not a float, so 0.25 finds _t0.25 and 25 or 0.250 find nothing --
# which the script reports by printing what IS on disk.
: "${LOSSES:=ce}"
: "${TAUS:=0.25}"
: "${DO_RAW:=0}"
: "${TAG:=T17_transfer}"

# ONE ARM IS THE SCHEDULABLE UNIT, and the reason is that the cost of an arm
# is six checkpoint loads, not the 78 forward passes -- an arm is roughly
# 4-8 min, most of it building ViT-L and B5 off the workspace filesystem.
# The full 12-arm grid in ONE job therefore loads the same six sets of
# weights twelve times over and takes ~1-2 h, which does not fit the 30 min
# dev-partition cap and does not need to: the arms are independent.
#
#   sbatch --array=0-11 matrix.sh       each task ONE arm, ~4-8 min, parallel
#   sbatch matrix.sh                    every arm sequentially, ~1-2 h
#   LIST=1 bash matrix.sh               print the arm list and exit
#
# Under --array, SLURM_ARRAY_TASK_ID indexes ARMS below, so the --array range
# must match the grid. LIST=1 prints it rather than leaving you to multiply
# it out, and an index past the end exits non-zero instead of silently
# running nothing.
ARMS=()
for L in $LOSSES; do
  # The unconstrained reference goes FIRST. raw is the ceiling on what any
  # tau can transfer, so a raw matrix that already collapses off-diagonal
  # says the csf arms are measuring the collapse and not the constraint.
  if [ "$DO_RAW" = "1" ]; then ARMS+=("$L|"); fi
  for T in $TAUS; do ARMS+=("$L|$T"); done
done

echo "LOSSES=[$LOSSES]  TAUS=[$TAUS]  DO_RAW=$DO_RAW  TAG=$TAG"
echo "${#ARMS[@]} arm(s) -> sbatch --array=0-$(( ${#ARMS[@]} - 1 )) matrix.sh"
if [ "${LIST:-0}" = "1" ]; then
  for i in "${!ARMS[@]}"; do
    L="${ARMS[$i]%|*}"; T="${ARMS[$i]#*|}"
    echo "  [$i] $L $([ -n "$T" ] && echo "csf tau=$T" || echo raw)"
  done
  exit 0
fi

n_ok=0; n_fail=0; started=$SECONDS

run () {                       # run <loss> [--csf --tau T]
  local loss="$1"; shift
  local t0=$SECONDS
  echo ""
  echo "──── $loss $*   ($(date +%H:%M:%S))"
  if python scripts/transfer_matrix.py --cityscapes_root "$CS" \
        --loss_fn "$loss" --tag "$TAG" "$@"; then
    printf '     ok    %5ds\n' $(( SECONDS - t0 ))
    n_ok=$(( n_ok + 1 ))
  else
    printf '     FAIL  %5ds   %s %s\n' $(( SECONDS - t0 )) "$loss" "$*"
    n_fail=$(( n_fail + 1 ))
  fi
}

run_arm () {                   # run_arm <index>
  local spec="${ARMS[$1]}"
  local L="${spec%|*}" T="${spec#*|}"
  if [ -n "$T" ]; then run "$L" --csf --tau "$T"; else run "$L"; fi
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ARMS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID is past the end of a "
    echo "${#ARMS[@]}-arm grid — the --array range and LOSSES/TAUS/DO_RAW "
    echo "disagree. Run LIST=1 bash matrix.sh."
    exit 2
  fi
  echo "array task $SLURM_ARRAY_TASK_ID of ${#ARMS[@]}"
  run_arm "$SLURM_ARRAY_TASK_ID"
else
  for i in "${!ARMS[@]}"; do run_arm "$i"; done
fi

echo ""
echo "════════════════════════════════════════════════════"
printf ' %d ok, %d failed, %dh %dm wall clock\n' "$n_ok" "$n_fail" \
  $(( (SECONDS-started)/3600 )) $(( ((SECONDS-started)%3600)/60 ))
echo " -> results/matrix_464/<loss>/<arm>/matrix.json"
echo "════════════════════════════════════════════════════"

# ── single arms, for a dev-partition smoke test ─────────────────────────────
# python scripts/transfer_matrix.py --cityscapes_root $CS --loss_fn ce --csf --tau 0.5
# python scripts/transfer_matrix.py --cityscapes_root $CS --loss_fn ce
#
# Restrict the lineup, or force a clean cross-architecture comparison (and then
# do NOT quote the read white-box diagonal against it — rerun the diagonal too):
# python scripts/transfer_matrix.py --cityscapes_root $CS --loss_fn ce --csf --tau 0.5 \
#     --archs segformer_b0 deeplab_18 --inference whole
