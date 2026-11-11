#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# IAP (Dutta & Zhang, ICCV 2025) against this repository's segmentors.
# The procedure, the deviations and the cost are in the docstrings of
# patchreach/patch/iap.py and scripts/iap_attack.py.

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/iap
exec 1> "slurm/iap/${SLURM_JOB_ID}_final09_ce.out"
exec 2> "slurm/iap/${SLURM_JOB_ID}_final09_ce.err"
module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/iap
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

python --version
python -c "import torchmetrics; print('torchmetrics', torchmetrics.__version__)"

BASE="--arch segformer --cityscapes_root $CS --img_h 1024 --img_w 2048"
GEOM="--patch_size 256 --patch_scale 0.25"
POP="--images fixed10 --target_class 16 --log_every 100"

# ── the arm proper: the released setting ─────────────────────────────────────
python scripts/iap_attack.py \
  --arch segformer_b0 --cityscapes_root "$CS" \
  --img_h 1024 --img_w 2048 --inference whole \
  --num_classes 19 --seed 42 \
  --patch_mode raw --pixel_param direct \
  --patch_size 256 --patch_scale 0.25 --patch_scale_ref height --shape square \
  --iap_locate cam --cam_objective ce --cam_target pred \
  --cam_layer -1 --cam_module backbone \
  --iap_tar_loss ipatch_cospgd --iap_untar_loss ce --target_class 16 \
  --iap_w1 1.5 --iap_w2 1.0 --iap_w3 3.0 --iap_reg_reduce sum \
  --iap_lrs 0.1 --iap_steps 1000 \
  --iap_stop target --iap_stop_thresh 0.99 \
  --images 464 --classes gt \
  --log_every 10 --n_panels 3 --panel_select spread \
  --out_root results/iap --tag whole_control_moreLogs_0.99_cos
