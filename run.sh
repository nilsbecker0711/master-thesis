#!/bin/bash
#SBATCH -p gpu_a100_il   # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 10:00:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=400000       # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)
##SBATCH --output=slurm/attack_%J_%j_%a.out
##SBATCH --error=slurm/attack_%J_%j_%a.err

echo "Running on $(hostname)"
echo "Date: $(date)"
TARGET_CLASS=-1
LOSS="cospgd"
exec 1> "slurm/attack/${SLURM_JOB_ID}_cls${TARGET_CLASS}_loss_${LOSS}_512x256.out"
exec 2> "slurm/attack/${SLURM_JOB_ID}_cls${TARGET_CLASS}_loss_${LOSS}_512x256.err"
module --ignore_cache load "cuda/11.8"

# initialize YOUR conda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes
export RES="--img_h 512 --img_w 1024"     # revisit after Phase 0.1
export BASE="--arch segformer --cityscapes_root $CS"
echo "Python path:"
which python
python -c "import sys; print(sys.executable)"

echo "Python version:"
python --version
#    --shape alpha \
    #--lap_freeze_edges --lap_edge_thresh 0.15 --target_class 13 \
for LR in 0.1 0.2 0.5; do python scripts/train.py --arch segformer --cityscapes_root $CS --img_h 512 --img_w 1024 --patch_mode universal_csf --patch_size 128 --patch_scale 0.25 --loss_fn cospgd --lr_schedule cosine --lr $LR --batch_size 4 --num_workers 16 --epochs 150 --val_images 500 --csf_threshold 0.25 --no_diagnostics --panel_images "" --tag 150_lr$LR; done