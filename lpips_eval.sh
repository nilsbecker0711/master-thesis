#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

# LPIPS NATURALNESS OF SAVED PATCHES — scripts/lpips_eval.py.
#
# Runs in the lpips_eval env (envs/lpips_eval.yml), NOT thesis_backup3: no
# segmentor is loaded, the frames are rebuilt from final.pt / best.pt + the
# val image. Every run directory gets <run>/lpips/{per_image.csv,summary.json,
# panel_*.png}; --out_csv collects every row in one table.
#
# COMPUTE NODES MAY HAVE NO INTERNET. lpips fetches the torchvision AlexNet /
# VGG weights on first use; warm the cache once on a login node:
#     conda activate lpips_eval
#     python -c "import lpips; lpips.LPIPS(net='alex'); lpips.LPIPS(net='vgg')"

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/lpips
exec 1> "slurm/lpips/${SLURM_JOB_ID}.out"
exec 2> "slurm/lpips/${SLURM_JOB_ID}.err"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/lpips_eval
export PYTHONNOUSERSITE=1
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes
python --version

# Score the csf runs AND an opaque raw run with the same settings: an LPIPS
# for the csf patch only means something next to the number it is low
# relative to. Adjust the globs to the runs you want in the table.
python scripts/lpips_eval.py \
    results/overfit/*csf* results/population/*csf* \
    results/overfit/*raw* \
    --cityscapes_root "$CS" --nets alex vgg --panels 4 \
    --out_csv results/lpips/lpips_all.csv

echo "Done: $(date)"
