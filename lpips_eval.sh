#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

# FULL PERCEPTUAL EVALUATION OF SAVED PATCHES — scripts/lpips_eval.py.
# Reasoning lives in that script's docstring; this file only launches it.
#
# COMPUTE NODES MAY HAVE NO INTERNET. Every metric downloads weights on first
# use, so warm the caches once on a login node:
#     conda activate lpips_eval
#     python -c "import lpips, pyiqa; [lpips.LPIPS(net=n) for n in ('alex','vgg','squeeze')]; [pyiqa.create_metric(m) for m in ('dists','stlpips','lpips+','lpips-vgg+','ahiq','topiq_fr','pieapp','niqe','brisque','clipiqa','musiq')]"

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

# ONE invocation for every run: the metric models load once, not once per
# directory. Score the csf runs AND an opaque raw one — a perceptual number
RUNS2=(
results/overfit_464/ce/setr_pup_raw_ce_img464
results/overfit_464/ce/segformer_b0_raw_ce_img464
results/overfit_464/ce/segformer_b5_raw_ce_img464
results/overfit_464/ce/deeplab_18_raw_ce_img464
results/overfit_464/ce/deeplab_101_raw_ce_img464
results/overfit_464/ce/deeplab_50_raw_ce_img464
results/overfit_464/cospgd/setr_pup_raw_cospgd_img464
results/overfit_464/cospgd/segformer_b0_raw_cospgd_img464
results/overfit_464/cospgd/segformer_b5_raw_cospgd_img464
results/overfit_464/cospgd/deeplab_18_raw_cospgd_img464
results/overfit_464/cospgd/deeplab_101_raw_cospgd_img464
results/overfit_464/cospgd/deeplab_50_raw_cospgd_img464

)
RUNS=(/pfs/work9/workspace/scratch/ma_nilbecke-thesis/master-thesis/results/overfit_464/cospgd/segformer_b0_raw_cospgd_img464
/pfs/work9/workspace/scratch/ma_nilbecke-thesis/master-thesis/results_wrong_res/unconstrained/segformer_b5_raw_cospgd_img420)

python scripts/lpips_eval.py "${RUNS[@]}" \
    --source checkpoint --checkpoint best --cityscapes_root "$CS" \
    --metrics all --anchors --tag ckpt
    #--out_csv results/perc_eval/overfit_464_lap_ckpt.csv

echo "Done: $(date)"
