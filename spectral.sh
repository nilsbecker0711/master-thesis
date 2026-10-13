#!/bin/bash
#SBATCH -p accelerated         # 16 probe runs at 1024x2048, setr on slide
#SBATCH -n 1
#SBATCH -t 04:00:00            # forward passes only, but 7 bands x 8 probes x 5 images
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

# ── THE BAND THAT WAS MISSING ──────────────────────────────────────────
#
# DEFAULT_BANDS used to stop at 0.50 c/px. The rfft grid runs to sqrt(2)/2 =
# 0.7071 -- 0.5 is the per-AXIS Nyquist, and the radial frequency is the length
# of (fy, fx), so the corner bin has both components at 0.5 and |f| = 0.707.
# Measured on the budget at size 256, tau 1:
#
#   band          CSF     budget      share of the envelope's power
#   0.20-0.35   98.93     0.0028
#   0.35-0.50   28.31     0.0103      0.4% lies below 0.35 in total
#   0.50-0.71    7.75     0.0429      91.7% lies above 0.50
#
# So the six bands RQ1 was answered over held 0.4% of the allowance, and the
# band holding the rest was never shown to a network. That is the gap this run
# closes, and it is a forward-pass sweep -- minutes, no training.
#
# WHAT TO LOOK FOR. The top band is where the budget spends, so its flip rate
# per JND is the number that decides whether the 1/CSF shape is smart or
# wasteful. Two outcomes, both findings:
#   high  -> the corner really is the bargain the envelope bets on, and the
#            weak absolute results are about the lr (M5) or reach (M2).
#   low   -> the envelope spends 92% of its allowance where the stem cannot
#            read it, and M3/M6/RQ1 are one finding.
# A third outcome is likely and is also a result: RANGE-LIMITED. Amplitude per
# JND is 0.064 there against 0.0055 at 0.20-0.35, so driving that band to the
# target cost may not fit in [0,1] on real content. summarise() flags any band
# whose realised cost is below 0.9x target; if the top band is flagged, the
# finding is that the band carrying the budget cannot reach tau without
# clipping, which is where the realised-vs-nominal gap comes from.
#
# NATIVE RESOLUTION, unlike the 512x1024 runs above: the band edges are in
# cycles per PIXEL, so halving the resolution halves the cpd every band maps
# to and the answer would not transfer to the attack runs.

export RES_NATIVE="--img_h 1024 --img_w 2048 --patch_size 256 --patch_scale 0.25"

probe () {                    # probe <arch> <tau> [extra flags]
  local arch="$1" tau="$2"; shift 2
  echo ""; echo "===== $arch  tau=$tau  $*   ($(date +%H:%M:%S))"
  python scripts/measure_frequency_sensitivity.py \
      --arch "$arch" --cityscapes_root "$CS" $RES_NATIVE \
      --inference auto --slide_checkpoint \
      --target "$tau" --n_images 5 --n_probes 8 --region patch "$@" \
      --out_dir results/spectral7
}

# The tau ladder on the three architectures the thesis leads with. 0.25 is the
# attack default; the ladder exists because a single tau that moves no band is
# reported INCONCLUSIVE rather than as a finding -- which is what happened to
# RQ1 the first time.
for ARCH in segformer_b0 segformer_b5 setr_pup; do
  for TAU in 0.25 0.5 1.0 2.0; do probe "$ARCH" "$TAU"; done
done

# region=full removes the receptive-field ceiling, so a dead top band cannot be
# blamed on reach -- the same confound M2 found in setr's raw ceiling.
for ARCH in segformer_b0 setr_pup; do probe "$ARCH" 1.0 --region full; done

# equal AMPLITUDE instead of equal visibility. The ratio between this and the
# run above is what isolates the CSF weighting itself: at equal rms the top
# band has no allowance advantage, so any remaining difference is the network.
for ARCH in segformer_b0 setr_pup; do
  probe "$ARCH" 0.02 --normalise rms
done
