#!/bin/bash
#SBATCH -p dev_gpu_a100_il
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# Frequency-resolved sensitivity probe. NO TRAINING, so this is minutes rather
# than hours — it is a forward-pass sweep, like measure_erf.py.
#
# WHAT IT SETTLES: the CSF attack family assumes the network still reads the
# frequencies the eye discards. This measures whether it does, and whether the
# readable band is a model property or moves with the scene.

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/spectral
exec 1> "slurm/spectral/${SLURM_JOB_ID}.out"
exec 2> "slurm/spectral/${SLURM_JOB_ID}.err"
module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes
export RES="--img_h 512 --img_w 1024"

python --version

# ── the tau ladder ───────────────────────────────────────────────────────────
# A SINGLE tau is not informative on its own: too low and no band moves the
# prediction, which the probe reports as INCONCLUSIVE rather than as a finding.
# The deliverable is the SMALLEST tau that produces a measurable response, and
# how the band profile changes as tau rises. 0.25 is the attack modes' default.
for TAU in 0.25 0.5 1.0 2.0; do
  echo "===================== tau = $TAU ====================="
  python scripts/measure_frequency_sensitivity.py \
      --arch segformer --cityscapes_root "$CS" $RES \
      --target "$TAU" --n_images 5 --n_probes 8 --region patch
done

# ── controls, at whichever tau turned out to be the operating point ──────────
# full   : removes the receptive-field ceiling, so a dead high band cannot be
#          blamed on reach. If patch and full disagree, the disagreement is the
#          ERF and not the spectrum.
# rms    : equal AMPLITUDE instead of equal visibility. The ratio between this
#          and the visibility-normalised run is what isolates the CSF weighting.
# fixed  : the mu = 0.5 convention the attack modes currently use, against the
#          locally-measured mean this probe defaults to.
TAU=1.0
python scripts/measure_frequency_sensitivity.py \
    --arch segformer --cityscapes_root "$CS" $RES \
    --target "$TAU" --n_images 5 --region full
python scripts/measure_frequency_sensitivity.py \
    --arch segformer --cityscapes_root "$CS" $RES \
    --normalise rms --target 0.02 --n_images 5 --region patch
python scripts/measure_frequency_sensitivity.py \
    --arch segformer --cityscapes_root "$CS" $RES \
    --target "$TAU" --n_images 5 --region patch --contrast_mean fixed

# ── the bracket comparison ───────────────────────────────────────────────────
# Label-free and dataset-independent, exactly like measure_erf.py, so this runs
# on whatever weights the registry resolves — ADE20K included.
#python scripts/measure_frequency_sensitivity.py \
#    --arch internimage --cityscapes_root "$CS" $RES \
#    --target "$TAU" --n_images 5 --region patch
