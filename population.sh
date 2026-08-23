#!/bin/bash
#SBATCH -p accelerated
#SBATCH -n 1
#SBATCH -t 11:00:00
#SBATCH --mem=100000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# ═══════════════════════════════════════════════════════════════════════════
#  Per-image patches over a POPULATION — the baseline-B experiment block.
# ═══════════════════════════════════════════════════════════════════════════
#
# One patch per image, optimised independently. This is BASELINE B with an
# interval around it instead of an anecdote. NOT a universal patch; its numbers
# do not belong beside train.py's.
#
# THE DESIGN IS A FACTORIAL ON ONE SHARED SAMPLE.
#
#   placement   center | gradcam      does sensitivity-guided localisation help?
#   loss        cospgd | ce           what does the cosine weighting buy?
#   constraint  raw    | csf          what does invisibility cost?
#
# EVERY arm uses the SAME --images/--n_images/--sample_seed. That is not
# tidiness: analysis/compare_populations.py pairs arms BY IMAGE INDEX, and the
# paired difference is the only test that can resolve a 2-point effect under a
# 10-point scene-to-scene spread. Change the seed between arms and the pairing
# silently degrades to whatever images happen to overlap.
#
# CHANGE ONE AXIS AT A TIME. compare_populations.py warns when two arms differ
# on more than one axis, because a paired difference then attributes the effect
# to both and the number is uninterpretable.
#
# RESUMABLE: resubmit the same command with --resume --out_dir <its directory>.

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/population
exec 1> "slurm/population/${SLURM_JOB_ID}.out"
exec 2> "slurm/population/${SLURM_JOB_ID}.err"
module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

python --version

# ── the shared sample, fixed once ────────────────────────────────────────────
# n=100 supports a +/-2 point claim at sigma~10. Raise to 500 (the full val
# set, which is what the field does) for the headline once the pilot has told
# you seconds/image. The run itself reports whether the n it had was enough.
export N=100
export SEED=0
export SAMPLE="--images random --n_images $N --sample_seed $SEED"
export BASE="--arch segformer --cityscapes_root $CS --img_h 512 --img_w 1024"
export RUN="--steps 300 --n_panels 3 --panel_select spread"
export ROOT=results/population

# --placement_margin: the sensitivity map's hottest ridge is the near-field
# road boundary along the BOTTOM of a dashcam frame, so the unmargined argmax
# pins the patch flush against the edge where ~half its receptive field falls
# outside the image. p = 0.25 * 512 = 128, so p/2 = 64.
export GRADCAM="--placement gradcam --placement_margin 64 --cam_target pred"

# ── 0. PILOT — time it before committing ─────────────────────────────────────
python scripts/overfit_population.py $BASE $RUN \
    --patch_mode csf --from_image --loss_fn cospgd --csf_threshold 0.25 \
    --images random --n_images 10 --sample_seed $SEED \
    --out_root $ROOT --tag pilot

# ═══════════════════════════════════════════════════════════════════════════
#  BASELINES — the reference points every other arm is quoted against
# ═══════════════════════════════════════════════════════════════════════════
#
# B0  raw + center    the historical baseline B, unchanged. The ceiling with
#                     no perceptual constraint and no localisation.
# B1  raw + gradcam   the ceiling WITH localisation. Also the repair to a live
#                     confound: the conditional generator defaults to
#                     --gen_placement gradcam while baseline B has always been
#                     --placement center, so every generator-vs-B comparison in
#                     block F currently mixes attack family with placement.
#                     B1 is the arm block F should actually be compared against.

python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --patch_mode raw --loss_fn cospgd \
    --placement center --out_root $ROOT --tag B0-raw-center

python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --patch_mode raw --loss_fn cospgd \
    $GRADCAM --out_root $ROOT --tag B1-raw-gradcam

# ═══════════════════════════════════════════════════════════════════════════
#  AXIS 1 — PLACEMENT.  What does sensitivity-guided localisation buy?
# ═══════════════════════════════════════════════════════════════════════════
#
# B0 vs B1 above already answers this for raw. Repeat under csf, because the
# answer need not be the same: a CSF residual is a bounded perturbation on the
# covered content, and gradcam pushes placement toward the near field, which is
# DARK ASPHALT. Dark content has less headroom for fit_to_range AND a lower
# local mean than the mu=0.5 that CONTRAST_SCALE assumes, so localisation and
# invisibility interact rather than compose.

python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --patch_mode csf --from_image --loss_fn cospgd --csf_threshold 0.25 \
    --placement center --out_root $ROOT --tag P-csf-center

python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --patch_mode csf --from_image --loss_fn cospgd --csf_threshold 0.25 \
    $GRADCAM --out_root $ROOT --tag P-csf-gradcam

# semantic placement is the PLAUSIBILITY arm, not a strength arm: a road-surface
# marking is a threat model someone can act on (Sato et al., USENIX Sec '21).
# Quote it as the realism cost, against B0.
python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --patch_mode raw --loss_fn cospgd \
    --placement semantic --placement_class 0 \
    --out_root $ROOT --tag P-raw-semantic-road

# ═══════════════════════════════════════════════════════════════════════════
#  AXIS 2 — LOSS.  ce vs cospgd, holding placement fixed.
# ═══════════════════════════════════════════════════════════════════════════
#
# CosPGD's own claim is that the cosine weighting beats plain CE. Test it on
# OUR implementation and say so: adversarial.py declares two deviations from
# the paper — the cosine is DETACHED (they do not detach), and Adam on the raw
# gradient replaces sign-SGD with epsilon-projection, because a patch is not an
# epsilon-ball perturbation. This measures our cospgd, not theirs.
#
# Run at BOTH placements. If the cosine's advantage depends on where the patch
# sits, that is a more interesting result than either arm alone — and untargeted
# cospgd is already known here to be an implicit class selector (~97% of remote
# flips were road->car), so the two axes have a plausible mechanism to interact.

python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --patch_mode raw --loss_fn ce \
    --placement center --out_root $ROOT --tag L-raw-ce-center

python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --patch_mode raw --loss_fn ce \
    $GRADCAM --out_root $ROOT --tag L-raw-ce-gradcam

# ═══════════════════════════════════════════════════════════════════════════
#  AXIS 3 — THE TAU LADDER.  The cost of invisibility, at the better placement.
# ═══════════════════════════════════════════════════════════════════════════
for TAU in 0.05 0.1 0.5; do
  python scripts/overfit_population.py $BASE $SAMPLE $RUN \
      --patch_mode csf --from_image --loss_fn cospgd --csf_threshold "$TAU" \
      $GRADCAM --out_root $ROOT --tag "T-csf-tau$TAU-gradcam"
done

# ═══════════════════════════════════════════════════════════════════════════
#  THE ANALYSIS — where the results actually come from
# ═══════════════════════════════════════════════════════════════════════════
#
# Each call pairs every later arm against the FIRST one (--baseline 0) and
# writes a paired figure plus comparison.json. One axis per call.

# placement, under raw and under csf
python analysis/compare_populations.py \
    $ROOT/*B0-raw-center* $ROOT/*B1-raw-gradcam*
python analysis/compare_populations.py \
    $ROOT/*P-csf-center* $ROOT/*P-csf-gradcam*

# loss, at each placement
python analysis/compare_populations.py \
    $ROOT/*L-raw-ce-center* $ROOT/*B0-raw-center*
python analysis/compare_populations.py \
    $ROOT/*L-raw-ce-gradcam* $ROOT/*B1-raw-gradcam*

# the cost of invisibility, against the unconstrained ceiling at the same
# placement, and the tau ladder against its own strongest rung
python analysis/compare_populations.py \
    $ROOT/*B1-raw-gradcam* $ROOT/*T-csf-tau0.5-gradcam* \
    $ROOT/*P-csf-gradcam* $ROOT/*T-csf-tau0.1-gradcam* \
    $ROOT/*T-csf-tau0.05-gradcam*

# the realism cost of a road-surface placement
python analysis/compare_populations.py \
    $ROOT/*B0-raw-center* $ROOT/*P-raw-semantic-road*
