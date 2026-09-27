#!/bin/bash
#SBATCH -p gpu_a100_il         # full grid; for ONE arm use dev_gpu_a100_il + 00:20:00
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 06:00:00            # Time limit
#SBATCH --mem=200000           # Memory request (1024x2048 + slide needs room)
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
# Override from the command line, e.g.
#   LOSSES=ce TAUS=0.5 DO_RAW=0 sbatch matrix.sh
: "${LOSSES:=ce cospgd cos_margin}"
: "${TAUS:=0.05 0.25 0.5}"
: "${DO_RAW:=1}"
: "${TAG:=T17_transfer}"

echo "LOSSES=[$LOSSES]  TAUS=[$TAUS]  DO_RAW=$DO_RAW  TAG=$TAG"

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

for L in $LOSSES; do
  # The unconstrained reference goes FIRST. raw is the ceiling on what any
  # tau can transfer, so a raw matrix that already collapses off-diagonal
  # says the csf arms are measuring the collapse and not the constraint.
  if [ "$DO_RAW" = "1" ]; then run "$L"; fi
  for T in $TAUS; do run "$L" --csf --tau "$T"; done
done

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
