#!/bin/bash
#SBATCH -p gpu_a100_short  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:30:00            # Time limit (10 minutes for debugging purposes)
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
export T=0.5
export A="segformer_b0"
python scripts/overfit.py --arch $A --cityscapes_root $CS \
    --img_h 1024 --img_w 2048 --scale crop --inference auto \
    --patch_mode csf --from_image --csf_threshold $T --csf_enforce realised \
    --patch_size 256 --patch_scale 0.25 --patch_size 256\
    --placement center --cam_objective ce --cam_target pred \
    --loss_fn ipatch_cospgd --target_class 16 \
    --image 464 --steps 300 --lr 0.1 --lr_schedule cosine \
    --log_every 20 \
    --out_root results/iap/cospgd_targeted --tag t$T