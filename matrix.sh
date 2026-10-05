#!/bin/bash
#SBATCH -p dev_gpu_a100_il     # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:30:00            # Time limit
#SBATCH --mem=400000           # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)
##SBATCH --output=slurm/attack_%J_%j_%a.out
##SBATCH --error=slurm/attack_%J_%j_%a.err

# T17 transfer matrix. One invocation per arm; the arm is three arguments:
#   --loss_fn L --csf --tau T   ->  <arch>_csf_L_img464_tT/best.pt
#   --loss_fn L                 ->  <arch>_raw_L_img464/best.pt
# Archs, resolution and inference mode are discovered from the source runs.
# -> results/matrix_464/<loss>/<arm>/matrix.json

set -uo pipefail              # NOT -e: one failed arm must not kill the rest

echo "Running on $(hostname)"
echo "Date: $(date)"

module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=${CS:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes}

echo "Python path:"
which python
echo "Python version:"
python --version

# TAU IS A STRING, spelled as the run directory spells it: 0.25 -> _t0.25
: "${LOSSES:=ce}"
: "${TAUS:=0.25}"
: "${DO_RAW:=0}"
: "${TAG:=T17_transfer}"

echo "LOSSES=[$LOSSES]  TAUS=[$TAUS]  DO_RAW=$DO_RAW  TAG=$TAG"

n_ok=0; n_fail=0; started=$SECONDS

run () {                      # run <loss> [--csf --tau T]
  local loss="$1"; shift
  local t0=$SECONDS
  echo ""
  echo "---- $loss $*   ($(date +%H:%M:%S))"
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
  # raw first: it is the ceiling on what any tau can transfer
  if [ "$DO_RAW" = "1" ]; then run "$L"; fi
  for T in $TAUS; do run "$L" --csf --tau "$T"; done
done

echo ""
printf '%d ok, %d failed, %dh %dm\n' "$n_ok" "$n_fail" \
  $(( (SECONDS-started)/3600 )) $(( ((SECONDS-started)%3600)/60 ))
echo "-> results/matrix_464/<loss>/<arm>/matrix.json"
