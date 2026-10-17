#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH -t 24:00:00
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
exec 1> "slurm/iap/${SLURM_JOB_ID}.out"
exec 2> "slurm/iap/${SLURM_JOB_ID}.err"
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
python scripts/iap_attack.py $BASE $GEOM $POP --tag released

# ── w3 ladder ────────────────────────────────────────────────────────────────
# The regulariser is a SUM over patch elements and the patch is 9.3x the area
# the published w3=3 was tuned at (D3), so the operating point has to be found
# rather than inherited. sum84 is the same run with w3 rescaled back.
for W3 in 0.3 1.0 3.0 9.0 30.0; do
  python scripts/iap_attack.py $BASE $GEOM $POP --iap_w3 "$W3" --tag w3sweep
done
python scripts/iap_attack.py $BASE $GEOM $POP --iap_reg_reduce sum84 --tag sum84

# ── does the regulariser do anything at all? ─────────────────────────────────
# D9: the released code builds its reference as temp = delta.clone(), which is
# differentiable, so the regulariser's two gradient paths cancel exactly and
# the term optimises nothing. This arm reproduces that. If it matches the
# w3=3 'released' run above, the regulariser is inert at this operating point
# and the invisibility comes from the 1/sens update rule alone.
python scripts/iap_attack.py $BASE $GEOM $POP --iap_reg_grad released \
    --tag reggrad_released

# ── the two localisation scores, and the centred control ─────────────────────
# D1: the paper's J/Sens and the released code's J are different procedures.
# --iap_locate center isolates what localisation contributes at all.
python scripts/iap_attack.py $BASE $GEOM $POP --iap_locate cam_over_sens --tag paperloc
python scripts/iap_attack.py $BASE $GEOM $POP --iap_locate center --tag centred

# ── the untargeted extension ─────────────────────────────────────────────────
# NOT IAP, which is targeted throughout. Here because every other arm of this
# thesis reports an untargeted number and the columns have to meet somewhere.
python scripts/iap_attack.py $BASE $GEOM $POP --iap_tar_loss none \
    --iap_untar_loss cospgd --iap_stop flip --tag untargeted

echo "Done: $(date)"
