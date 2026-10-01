#!/bin/bash
#SBATCH -p gpu_a100_il
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000
#
# The Nesti/Rossolini benchmark patch in ONE job. No driver, no slicing.
#
#     sbatch bench_single.sh
#
# WHY THIS EXISTS BESIDE benchmarkb0.sh + driverb0.sh. Those cut the run into
# 30-minute dev-partition slices because that was the only GPU on offer. On a
# 24h partition the whole mechanism is unnecessary: b0 measured 6.58 it/s at
# 1024x2048 with 3-window slide + checkpointing, so 200 epochs x 250 images
# is ~2.1 h of training plus ~25 min of validation. Copy this per arch and
# change ARCH; the heavier backbones are slower but still inside 24 h.
#
# --out_dir and --resume are set anyway. They cost nothing here and make a
# requeue free if the job is pre-empted or hits the walltime: it picks up
# from train_state.pt at the last completed epoch.
#
# DO NOT run this and driverb0.sh for the same ARCH at once -- they share
# out_dir and therefore train_state.pt.
#
# ── PROTOCOL (Nesti WACV 2022 / Rossolini TNNLS 2024) ───────────────────────
# Adopted: 1024x2048 native, 250 sampled TRAIN images, the entire 500-image
# val split, Adam, 200 epochs, mIoU+mAcc outside the patch area, plus a
# random-patch baseline row (under an outside-the-patch metric the clean
# number is a different pixel set and is not the comparison).
# Say "we adopt their DATA PROTOCOL", not "their training procedure": no EOT,
# one fixed patch size at scale 0.25 rather than their three-rung ladder and
# square rather than 1:2, not their lr 0.5, not their models.
# ─────────────────────────────────────────────────────────────────────────────

set -u
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$ROOT" || exit 1
[ -f scripts/train.py ] || { echo "FATAL: $ROOT is not the repo root" >&2; exit 1; }

echo "Running on $(hostname)   $(date)"
echo "root: $ROOT"

module --ignore_cache load "${CUDA_MODULE:-}" 2>/dev/null
source /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  && conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3 2>/dev/null
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib"
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes
# Those three fail on uc3 in every launch script here and always have; what
# puts python on PATH is sbatch --export=ALL. Prove it rather than assume it.
python -c "import sys,torch; print('python:',sys.executable); \
print('torch :',torch.__version__,'cuda=',torch.cuda.is_available()); \
sys.exit(0 if torch.cuda.is_available() else 'no CUDA device')" || exit 3

ARCH="${ARCH:-segformer_b0}"
OUT="results/runs/bench_nesti/$ARCH"

# NO --scale crop. At 1024x2048 it is a no-op (Cityscapes is natively that
# size, so RandomCrop of the same shape can only return the whole frame) --
# but it reads as augmentation that is not happening, and it WOULD bite at
# any other --img_h/--img_w.
BASE="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048 \
      --inference auto --slide_checkpoint"

# --patch_size 256, NOT 128. The pasted footprint is int(1024*0.25) = 256; a
# 128 parameter grid is silently INTERPOLATED UP to it, giving a band-limited
# patch with a quarter of the free parameters. raw mode does not refuse the
# mismatch (universal_csf does), so it is invisible unless you look.
GEOM="--patch_size 256 --patch_scale 0.25 --placement center"

# --batch_size 1: 250 steps/epoch, 50,000 optimiser steps over the run, and
# what the b0 timing above was measured at. --batch_size 4 is ~4x fewer, much
# noisier-gradient-free steps (12,600) and a different optimisation under the
# same cosine schedule -- not wrong, but not the same run. Raise it only if
# wall clock forces it, and say so in the write-up.
#
# --val_every 50, not 3. Each validation is 500 images x 2 forwards at native
# resolution with slide -- ~4 minutes. At 3 it would cost more than the
# training. The final 500-image evaluation runs after the loop REGARDLESS, so
# the headline number is never stale.
COMMON="--n_train 250 --train_seed 42 --val_images 500 --val_every 50 \
        --epochs 200 --optimiser adam --lr_schedule cosine \
        --batch_size 1 --num_workers 16 --loss_fn cospgd \
        --no_diagnostics --resume"

# ── the three rows ───────────────────────────────────────────────────────────
# Sequential in one job. The two baselines are --lr 0 --epochs 1: nothing
# moves, and the 500-image evaluation after the loop is the entire point.
# They cost ~8 min each, so run them FIRST -- if something is wrong with the
# evaluation path you find out in minutes rather than after the 2 h row.
#
# No csf flags: --csf_enforce / --csf_threshold are inert under
# --patch_mode raw and only make the command line look like it says something.

# raw_null -- random PIXELS. A raw patch initialises at uniform 0.5, so --lr 0
# alone would measure a GREY SQUARE; --raw_init random is the random patch the
# literature reports as its floor.
python scripts/train.py $BASE $GEOM $COMMON \
    --patch_mode raw --raw_init random --lr 0 --epochs 1 \
    --tag s0.25_null --out_dir "$OUT/raw_null"

# raw_grey -- the occlusion control. Separates "a patch of this size here
# perturbs the scene" from "arbitrary high-frequency content does".
python scripts/train.py $BASE $GEOM $COMMON \
    --patch_mode raw --lr 0 --epochs 1 \
    --tag s0.25_grey --out_dir "$OUT/raw_grey"

# raw -- the run. ~2.1 h training + ~25 min validation for b0.
python scripts/train.py $BASE $GEOM $COMMON \
    --patch_mode raw --raw_init random --lr 0.1 \
    --tag s0.25 --out_dir "$OUT/raw"

echo
echo "Done: $(date)"
echo "Index:  python analysis/build_index.py --runs results/runs/bench_nesti \\"
echo "            --out results/tables/bench_nesti/index.csv"
