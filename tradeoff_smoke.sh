#!/bin/bash
#SBATCH -p dev_gpu_a100_il
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH --mem=400000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --output=slurm/tradeoff/smoke_%A_%a.out
#SBATCH --error=slurm/tradeoff/smoke_%A_%a.err

# ═══════════════════════════════════════════════════════════════════════════
#  SMOKE TEST for the trade-off block. Plumbing only — no result comes out.
# ═══════════════════════════════════════════════════════════════════════════
#
#     sbatch tradeoff_smoke.sh
#
# Five minutes on the dev partition is enough to answer the questions that
# otherwise surface four hours into a real allocation: does this interpreter
# have mmcv, does each checkpoint load, does --csf_enforce realised hold at
# tau on THIS architecture, does the panel script find the runs and draw.
#
# IT WRITES TO ITS OWN TREE, results/tradeoff_smoke, and that is not tidiness.
# sweep_operating_point.py resumes BY RESULT: a 20-step cell sitting in the
# real tree under the right tag is indistinguishable from a finished one, and
# the real sweep would skip it and report 20 steps as its answer. Never point
# a smoke run at $ROOT.

module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

mkdir -p slurm/tradeoff

# The attention bracket, one array task each. registry.py marks exactly these
# three "global"; deeplab/unet/internimage are a different bracket and a
# different question.
ARCHS=(segformer_b0 segformer_b5 setr_pup)
ARCH=${ARCHS[${SLURM_ARRAY_TASK_ID:-0}]}
ROOT=results/tradeoff_smoke

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "SMOKE arch=$ARCH  ->  $ROOT"
which python
python --version

# Two rungs, 20 steps. tau 0.25 is the incumbent and tau 4 is above the knee
# on every architecture measured so far, so the pair also exercises the
# range-fit path that the panel's shading depends on.
python scripts/sweep_operating_point.py \
    --cityscapes_root "$CS" --arch "$ARCH" --image 420 \
    --img_h 512 --img_w 1024 \
    --losses cospgd --stages tau --steps_grid 20 \
    --tau_grid "0.25 4" --incumbent_lr 0.01 \
    --csf_param pgd --enforce realised --lr_schedule cosine \
    --seeds 1 --name "smoke_${ARCH}" --out_root "$ROOT" || exit 1

python scripts/overfit.py \
    --arch "$ARCH" --cityscapes_root "$CS" --img_h 512 --img_w 1024 \
    --patch_mode raw --loss_fn cospgd \
    --image 420 --steps 20 --lr 0.05 --lr_schedule cosine \
    --placement center --patch_scale 0.25 \
    --seeds 1 --no_diagnostics \
    --out_root "$ROOT/smoke_${ARCH}/cells" --tag ceiling_raw || exit 1

# Task 0 draws, once the other two have had their five minutes. If it runs
# first it simply plots one column -- which still proves the reader works.
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
  python analysis/tradeoff_panel.py \
      --sweep "$ROOT"/smoke_segformer_b0 \
      --sweep "$ROOT"/smoke_segformer_b5 \
      --sweep "$ROOT"/smoke_setr_pup \
      --out figures/tradeoff_smoke.png \
      --title "SMOKE — 20 steps, 2 rungs. Not a result."
fi

echo "Done: $(date)"
