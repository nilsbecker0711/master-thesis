#!/bin/bash
#SBATCH -p accelerated         # 3 setr cells at 1024x2048 will not fit a short queue
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 04:00:00            # 3 x 1000 steps, slide inference, diagnostics on
#SBATCH --mem=40000       # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)
##SBATCH --output=slurm/attack_%J_%j_%a.out
##SBATCH --error=slurm/attack_%J_%j_%a.err

echo "Running on $(hostname)"
echo "Date: $(date)"

module --ignore_cache load "cuda/11.8"

# initialize YOUR conda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
#export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes
export RES="--img_h 512 --img_w 1024"     # revisit after Phase 0.1
export BASE="--arch segformer --cityscapes_root $CS"
echo "Python path:"
which python
python -c "import sys; print(sys.executable)"

echo "Python version:"
python --version
#--lap_freeze_edges --lap_edge_thresh 0.15 --patch_scale 0.35
#python scripts/overfit.py --arch segformer --cityscapes_root $CS --patch_mode csf --from_image --loss_fn ipatch_cospgd --target_class 16 --image 420 --steps 1000 --lr 0.2 --csf_threshold 0.25 --csf_enforce realised --seeds 5 --log_every 50 --out_root results/overfit --tag targeted_train_n5
export PYTHONNOUSERSITE=1
#python scripts/overfit.py --arch setr_pup --cityscapes_root $CS --img_h 512 --img_w 1024 --patch_mode csf --from_image --patch_size 128 --patch_scale 0.25 --placement center --csf_threshold 0.25 --csf_enforce realised --loss_fn ce --image 42 --steps 1000 --lr 0.2 --lr_schedule cosine --seeds 1
# previous occupant of this file, kept for reference (needs export T=5):
#export A="segformer_b5"
#for L in cospgd; do
#    python scripts/overfit.py --arch $A --cityscapes_root $CS \
#        --patch_mode raw --shape alpha --reference refs/dog_cut.png --lap_freeze_edges --lap_edge_thresh 0.15  --from_image --patch_scale 0.85 \
#        --placement center --csf_threshold $T --csf_enforce realised \
#        --loss_fn $L --image 464 --steps 600 --lr 0.1 \
#        --lap_alpha 0.0001 --lap_beta 0.2 --lap_gamma 0 \
#        --lr_schedule cosine  --img_w 2048 --img_h 1024 --inference auto --slide_checkpoint\
#        --log_every 20 --out_root results/sizeLPIPS --tag raw\
#
#done

# ── WHERE THE BUDGET ACTUALLY SPENDS ITS ALLOWANCE ──────────────────────────
#
# Measured on this budget at size 256, tau 1 (patchreach/patch/csf.py:
# patch_budget, and tests/test_csf.py pins the numbers):
#
#   91.7% of the envelope's power sits above 0.5 c/px, 99.6% above 0.35, on
#   22% of the bins -- the diagonal corner of the rfft grid, i.e. a 1-2 px
#   checkerboard. The radial maximum is sqrt(2)/2 = 0.707, not 0.5.
#
# Every arch here downsamples at the stem, and a 2x2 box average passes 0.75%
# of that power (4x4: 0.13%). So the allowance is concentrated in the band the
# stem discards, which is a sufficient explanation for setr's 0.7% flip rate at
# tau 1 without any appeal to architecture.
#
# --csf_max_cpp cuts the envelope BEFORE the tau normalisation, so the live
# bins absorb what the corner gave up. Verified on cpu at size 256, tau 1:
#
#                       realised v   resid rms   in-band rms   survives 2x2
#   no cut                  0.966      0.1303        0.0140          0.75%
#   --csf_max_cpp 0.35      1.000      0.0119        0.0201         36.3%
#
# Same observer model, same nominal tau, slightly HIGHER realised visibility,
# 11x less total energy, 1.4x more energy where the probe says the network
# reads. If the cut arm does not out-flip the control, the 1/CSF shape is not
# what is limiting this attack and M5 (the lr) is the next suspect. If it does,
# M3/M6/RQ1 are one finding: the envelope is mis-specified for segmentation.
#
# THE THIRD CELL EXISTS BECAUSE THE TWO ARMS HAVE DIFFERENT PARAMETER SCALES.
# Under --csf_param pgd the parameter IS the residual, so the cut arm's rms is
# 11x smaller and lr 0.1 is ~8x its rms per step. Cell 2 holds the published
# protocol fixed; cell 3 scales the step to the parameter so a null result
# cannot be blamed on the step size. Same argument as lr_sweep.sh.

export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

# --patch_size 256 IS LOAD-BEARING: int(1024*0.25) = 256, and any other value
# resamples the residual, which resamples the spectrum the budget was built on.
run () {                      # run <tag> <max_cpp> <lr>
  echo ""; echo "---- $1  max_cpp=$2  lr=$3   ($(date +%H:%M:%S))"
  python scripts/overfit.py --arch setr_pup --cityscapes_root "$CS" \
      --img_h 1024 --img_w 2048 --inference auto --slide_checkpoint \
      --patch_mode csf --from_image --patch_size 256 --patch_scale 0.25 \
      --placement center --csf_threshold 1.0 --csf_param pgd \
      --csf_enforce realised --csf_max_cpp "$2" \
      --loss_fn cospgd --image 464 --steps 1000 --lr "$3" \
      --lr_schedule cosine --seeds 1 --log_every 20 \
      --out_root results/bandcut --tag "$1"
}

run full_lr0.1   0      0.1      # control: the published tau=1 setr protocol
run cut035_lr0.1 0.35   0.1      # same step, allowance moved into the band
run cut035_lr0.01 0.35  0.01     # step scaled to the 11x smaller parameter

# Step 0 of the cut cells should log:
#   [patch] band      : 0.0078-0.3500 c/px (12,700 live bins, high cut 0.35)
#   tau=1 requested -> 1.0000 realised   (residual rms ~0.012)
# If the band line says 0.7071 the flag did not reach the budget.
