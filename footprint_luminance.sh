#!/bin/bash
#SBATCH -p single
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --mem=16000
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

# STEP 0 for the universal CSF patch. NO MODEL, NO GPU — this is image
# statistics plus closed-form CSF evaluation, so it runs on a CPU partition and
# finishes in minutes over the full 2975-image train split.
#
# WHAT IT SETTLES: the universal patch fixes ONE reference luminance because it
# cannot look at the image it lands on. This measures what that costs, by
# reporting the amplitude budget at the 5th percentile, median and 95th
# percentile of the luminance the footprint actually encounters.
#
# READ THE sRGB COLUMN, NOT THE LINEAR ONE. The residual is optimised in sRGB
# code values, so that is the space the bound has to hold in. The linear column
# is the physical quantity and is printed for the decomposition only — quoting
# it as "the budget moves 7x" would be measuring in a space nothing in the
# attack lives in.

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/footprint
exec 1> "slurm/footprint/${SLURM_JOB_ID}.out"
exec 2> "slurm/footprint/${SLURM_JOB_ID}.err"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes
export RES="--img_h 512 --img_w 1024"

python --version

# ── the headline run ─────────────────────────────────────────────────────────
# train split, centre placement, the default 128 px footprint at 512x1024.
python scripts/measure_footprint_luminance.py \
    --cityscapes_root "$CS" $RES --split train \
    --placement center --patch_scale 0.25

# ── placement control ────────────────────────────────────────────────────────
# (0.75, 0.5) is the road surface straight ahead — near-field asphalt, and the
# darkest place the patch could plausibly sit. If the luminance spread is
# materially wider here than at centre, then the CHOICE OF PLACEMENT is doing
# more to the budget than the choice of L_ref, and that ordering belongs in the
# writeup before any conclusion about content-adaptivity.
python scripts/measure_footprint_luminance.py \
    --cityscapes_root "$CS" $RES --split train \
    --placement fixed --placement_xy 0.75 0.5 --patch_scale 0.25

# ── split control ────────────────────────────────────────────────────────────
# L_ref must be chosen on train. This run exists only to confirm val is not
# systematically brighter or darker — if it is, a budget fitted on train is
# mis-set on the split every number gets reported on, and that is a confound
# rather than a result.
python scripts/measure_footprint_luminance.py \
    --cityscapes_root "$CS" $RES --split val \
    --placement center --patch_scale 0.25

# ── model control ────────────────────────────────────────────────────────────
# sso has NO luminance parameter, so under it only the contrast denominator
# responds to L_ref. The gap between this run and the barten run isolates
# Barten's sensitivity term from the Michelson term — the decomposition the
# report prints, obtained a second, independent way.
python scripts/measure_footprint_luminance.py \
    --cityscapes_root "$CS" $RES --split train \
    --placement center --patch_scale 0.25 --csf_model sso
