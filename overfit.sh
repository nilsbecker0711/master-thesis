#!/bin/bash
#SBATCH -p dev_gpu_a100_il    # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:10:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=40000        # Memory request (adjust as needed)
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
export PYTHONNOUSERSITE=1
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
python scripts/overfit.py --arch segformer --cityscapes_root $CS --patch_mode csf --from_image --loss_fn tsallis --tsallis_schedule linear --tsallis_q_start -2 --tsallis_q_end 1 --image 420 --steps 1000 --lr 0.2 --csf_threshold 0.25 --csf_enforce realised --seeds 5 --log_every 50 --out_root results/overfit --tag q-2to1_lr0_2_n5