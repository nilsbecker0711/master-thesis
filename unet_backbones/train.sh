#!/bin/bash
#SBATCH -p gpu_mi300 #dev_gpu_a100_il  # Use the dev_gpu_4_a100 partition with A100 GPUs dev_gpu_4 
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 5:00:00     # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=30000             # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more) activate only if computation is on GPU!
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

echo "Running on $(hostname)"
echo "Date: $(date)"

module load devel/cuda/12.8

# initialize  conda
source /pfs/work9/workspace/scratch/ma_nilbecke-master_thesis/miniforge3/etc/profile.d/conda.sh

conda activate /pfs/work9/workspace/scratch/ma_nilbecke-master_thesis/miniforge3/envs/thesis

echo "Python path:"
which python
python -c "import sys; print(sys.executable)"

echo "Python version:"
python --version


python ~/master-thesis/unet_backbones/train.py --epochs 1