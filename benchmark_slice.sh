#!/bin/bash
#SBATCH -p dev_gpu_a100_il
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# ═════════════════════════════════════════════════════════════════════════════
#  ONE SLICE of the Nesti/Rossolini benchmark.  Driven by driver_benchmark.sh.
#
#      ./benchmark_slice.sh --list          the config table, machine readable
#      sbatch benchmark_slice.sh <name>     one 30-minute slice of <name>
#
#  Every invocation passes --resume and the FULL --epochs. A slice stops
#  because SLURM kills it at the walltime, never because it was told to do
#  less work: --epochs sets the cosine T_max, so a slice asked for fewer
#  would silently run a different optimisation. train.py checkpoints after
#  every epoch (param, Adam moments, scheduler position, gstep, RNG) and the
#  next slice picks up exactly where this one died. Verified bit-exact
#  against an uninterrupted run in tests/test_train_resume.py.
#
#  THE CONFIG TABLE LIVES HERE, ONCE. The driver reads it back through
#  --list rather than keeping its own copy, so an out_dir cannot drift
#  between the two — the failure mode the population driver has a comment
#  about.
#
# ─────────────────────────────────────────────────────────────────────────────
#  THE PROTOCOL.  Nesti et al., WACV 2022, arXiv:2108.06179, Sec. 4.1-4.2,
#  and its journal extension Rossolini et al., TNNLS 2024.
#
#  THEIRS, reproduced:
#    * Cityscapes at the full 1024x2048 (no downscaling, no crop).
#    * 250 images sampled from the TRAINING split for optimisation.
#    * the ENTIRE 500-image validation split for evaluation.
#    * Adam, 200 epochs.
#    * mIoU and mAcc on pixels OUTSIDE the patch area.
#    * a random-patch column, which under an outside-the-patch metric is the
#      real baseline — the clean number is measured over a different pixel
#      set and is not the comparison.
#
#  WHAT WE CAN CLAIM: "we adopt the DATA PROTOCOL of Nesti et al." — not
#  "their training procedure", which the four deviations below make false.
#
#  OURS, each to be stated in the write-up:
#
#    1. NO EOT. Their non-EOT arm places the patch at the image centre every
#       iteration, which is what this runs. Their EOT arm adds 5% Gaussian
#       noise, 80-120% random scaling and bounded random translation. None of
#       it exists in Patch.apply(), which composites ONE (top,left) with
#       F.pad and expands a single mask across the batch. For universal_csf
#       the scaling half also collides with the budget: _apply_residual
#       REFUSES to resample the residual, because resampling resamples its
#       spectrum and tau then describes a different tensor than the one
#       pasted. Decide that before writing an EOT arm.
#
#    2. ONE PATCH SIZE, FIXED AT scale 0.25 — not their three-rung ladder,
#       and square rather than their 1:2 rectangles (everything here is
#       square by construction: footprint_side() returns one int and
#       patch_budget() builds a size x size envelope).
#
#         theirs      % of frame      ours              % of frame
#         150x300         2.14        --                --
#         200x400         3.82        256x256 @ 0.25    3.125
#         300x600         8.57        --                --
#
#       0.25 is the operating point every other run in this thesis uses, and
#       because scale_ref='height' fixes coverage by aspect ratio alone, it
#       is 3.125% of the frame at 512x1024 AND at 1024x2048. So this row
#       stays comparable to our own body of work; only the absolute pixel
#       count changes with resolution. Patch size already has a systematic
#       table of its own (T19), where nu_min = m_c/S is handled properly —
#       averaging over their ladder here would duplicate it worse.
#       See the GEOMETRY block below.
#
#    3. NOT THEIR LEARNING RATE (0.5, set empirically on an unconstrained
#       pixel patch). universal_csf at lr 0.2 discards the previous residual
#       every step (mean relative change 1.89) with frac_at_bound 0.93 —
#       pinned to the boundary, phase-only; lr 0.01 gives 0.15 and 0.508.
#       Table in universal_csf.sh.
#
#    4. NOT THEIR MODELS. DDRNet/BiSeNet/ICNet are real-time, which is WHY
#       they get native resolution. docs/experiment_log.md: "Native
#       resolution OR the heavy global-attention architectures the ERF
#       hypothesis requires, not both." Broken deliberately for this row,
#       and paid for with --batch_size 1.
#
#  "200 epochs" IS NOT TRANSFERABLE WITHOUT THEIR BATCH SIZE. 200 epochs on
#  250 images is 50,000 steps at batch 1 and 12,500 at batch 4. What is
#  matched here is IMAGE PRESENTATIONS (250 x 200 = 50,000); say that rather
#  than implying step-for-step equivalence. Expect the CSF arm to plateau
#  well before 200 — universal_csf.sh records 20 epochs over 2975 images as
#  ample for a residual pinned to its envelope from step 0. Do not shorten
#  it on that basis: with cosine, --epochs is a SCHEDULE parameter, so a
#  60-epoch run anneals over 15k steps and is a different optimisation, not
#  a truncation of the 200-epoch one.
# ═════════════════════════════════════════════════════════════════════════════

set -u

# ── WHERE THE REPO IS ────────────────────────────────────────────────────────
# NOT dirname "$BASH_SOURCE". sbatch COPIES the batch script to
# /var/spool/slurmd/job<id>/slurm_script and executes the copy, so under
# SLURM that expands to the spool directory -- which is not the repo, is not
# writable, and made `python scripts/train.py` fail with "can't open file
# /var/spool/slurmd/job.../scripts/train.py".
#
# SLURM already starts a batch job in the directory sbatch was invoked from
# and exports it as SLURM_SUBMIT_DIR, so that is the answer under a job. The
# BASH_SOURCE form is kept only as the fallback for running this script
# directly on a login node (`bash benchmark_slice.sh --list`), where it is
# correct. REPO_ROOT overrides both.
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [ ! -f "$ROOT/scripts/train.py" ]; then
    echo "FATAL: '$ROOT' is not the repo root -- no scripts/train.py there." >&2
    echo "  sbatch from the repo root, or set REPO_ROOT=/path/to/master-thesis." >&2
    exit 1
fi

# Set BEFORE the config table, which interpolates it, and overridable from
# the environment so --list works on a login node with nothing loaded.
CS="${CS:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes}"
export CS

EPOCHS=200

# ARCH is a variable and lands in the out_dir, so a later b5 run does not
# collide with this one. NOTE: 'segformer' is an ALIAS for segformer_b0 --
# registry.py ends with REGISTRY["segformer"] = REGISTRY["segformer_b0"] --
# so the default here is B0, the smallest MiT. That is what makes 1024x2048
# affordable at all; it is also the weakest version of the "heavy
# global-attention transformer" the ERF hypothesis is about, so if the
# benchmark row is meant to carry that claim it wants b5 and a much longer
# queue. Override with ARCH=segformer_b5 in the environment; sbatch exports
# it to the slice.
ARCH="${ARCH:-segformer_b0}"
OUTROOT="results/runs/bench_nesti/$ARCH"

# ── GEOMETRY: FIXED AT scale 0.25, NOT their size ladder ─────────────────────
# Nesti sweep three patch sizes (2.14 / 3.82 / 8.57% of the frame). We hold
# ONE operating point instead, the 0.25 every other run in this thesis uses.
#
# WHY THAT IS THE BETTER TRADE HERE. scale_ref='height' means the footprint
# is int(H*0.25) and coverage depends only on aspect ratio, which Cityscapes
# fixes at 1:2. So 0.25 is 3.125% of the frame at 512x1024 AND at 1024x2048
# -- the benchmark patch covers exactly as much of the scene as every patch
# in the rest of the results chapter, and the ONLY thing that changes at
# native resolution is the absolute pixel count (256 vs 128 a side). That
# keeps this row comparable to our own body of work, which their ladder
# would have cost us, and patch size already has its own systematic table
# (T19) where nu_min = m_c/S is handled properly.
#
# The comparison to state: we report a fixed operating point against their
# ladder, and 3.125% sits between their 150x300 (2.14%) and 200x400 (3.82%)
# rungs -- closest to the middle one. Do not quote our number against a
# specific rung of theirs as though the sizes matched.
PSCALE=0.25
PSIZE=256              # int(1024 * 0.25); universal_csf refuses a mismatch
GEOM="--patch_size $PSIZE --patch_scale $PSCALE"

# The protocol flags every training config shares. --val_every 50 because
# each validation is 500 images x 2 forwards at native resolution; the final
# 500-image evaluation runs after the loop REGARDLESS, so the headline number
# is never stale. best.pt is then the best of ~5 validations — Nesti report
# the final patch, so quote `final`, and treat best.pt as a diagnostic.
# ── INFERENCE MODE: slide, and it is not the repo default ────────────────────
# --inference auto honours the checkpoint's declared test_cfg: SLIDE for
# segformer and setr, whole for deeplab and unet. Every other table in this
# thesis uses whole, so this is a deliberate departure, for two reasons.
#
# REALISM. mmseg's SegFormer Cityscapes config declares mode='slide', and the
# published numbers this row is compared against were produced that way. A
# model nobody deploys under whole-image forward should not be attacked under
# whole-image forward: that measures damage to a configuration that does not
# exist, and it flatters the attack.
#
# AND THE USUAL OBJECTION DOES NOT APPLY HERE. clean_baseline.py's
# SLIDE_WARNING says slide caps a patch's reach, because each window is an
# independent forward and a patch cannot influence pixels outside the windows
# containing it -- an ARCHITECTURE-DEPENDENT ceiling that confounds any reach
# measurement. True in general, and the reason the ERF and reach tables must
# stay on whole. But compute it for THIS configuration: at 1024x2048 the
# 1024x1024 crop / 768 stride grid is three windows -- [0,1024], [768,1792],
# [1024,2048] -- and a CENTRED patch of 128 or 256 px lands in all three, so
# their union is the entire frame and the cap is exactly zero. Verify with
# patchreach.models.wrapper.slide_windows before changing arch, resolution or
# placement: setr_pup's 768 crop caps the same patch to x[512,1792] of 2048,
# and off-centre placement breaks it for segformer too.
#
# COST: three windows is ~3x the forward work per step, and
# --slide_checkpoint recomputes each window in backward rather than holding
# every window's graph, which trades a second forward for the memory. That is
# what makes batch 1 at native resolution fit. The same pair of flags the
# 1024x2048 overfit runs already use.
INFER="--inference auto --slide_checkpoint"

PROTO="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048 \
       --placement center --n_train 250 --train_seed 68 $INFER \
       --val_images 500 --val_every 50 --epochs $EPOCHS \
       --optimiser adam --lr_schedule cosine --batch_size 1 \
       --num_workers 16 --loss_fn cospgd --no_diagnostics --resume"

# ── the config table ─────────────────────────────────────────────────────────
# name | out_dir | script | args
#
# Five rows at the single 0.25 operating point. NO clean row — that number
# already exists per architecture; see the note in configs() for the two
# conditions under which the existing one is a valid reference here.
#
# The two _null rows and the _grey row are --lr 0 --epochs 1: nothing
# moves, and the post-loop 500-image evaluation is the entire point of the
# run.
#
#   csf       the mode under test, tau 0.25 — the rung every other CSF number
#             in the thesis is quoted at. lr 0.01: the parameter IS the
#             residual and the constraint is a projection, so lr sets movement
#             inside a small ball. NOT the 0.2 that suits the scale-invariant
#             squash parameterisation of mode='csf'.
#   csf_null  the random-patch floor. universal_csf initialises from randn and
#             projects onto the CSF envelope, so at lr 0 the patch stays a
#             random residual at EXACTLY tau — an unoptimised perturbation of
#             equal perceptual cost. The gap to `csf` is what optimisation
#             bought.
#   raw       the unconstrained ceiling, and the arm closest to what Nesti
#             actually optimise. If this lands nowhere near their published
#             drops, the problem is plumbing, not the perceptual budget —
#             which is what makes a null CSF result interpretable.
#   raw_null  random PIXELS. A raw patch initialises at uniform 0.5, so --lr 0
#             alone would measure a GREY SQUARE. --raw_init random is the
#             random patch the literature reports.
#   raw_grey  that grey square, kept deliberately. It separates two things the
#             random row conflates: "a patch of this size at this position
#             perturbs the scene" from "arbitrary high-frequency content
#             perturbs the scene". Remote mIoU excludes the footprint, so a
#             grey square moving the number is acting on pixels it does not
#             cover.
configs () {
  local T="s${PSCALE}"        # the tag: our operating point, not their size

  # NO `clean` ROW. The clean dataset mIoU per architecture already exists
  # (results/clean_baselines*), so re-running it here would be an hour of
  # A100 per arch for a number we have.
  #
  # BUT IT ONLY SUBSTITUTES IF IT MATCHES THE ATTACK PATH. The reference for
  # this table has to be the same measurement: --img_h 1024 --img_w 2048,
  # the full val split, and --inference auto -- i.e. SLIDE for segformer,
  # because that is what these rows are now attacked under (see the
  # INFERENCE MODE block above). A 512x1024 number is a different
  # measurement, and so is a whole-image one. clean_baseline.py's
  # SLIDE_WARNING says to compare a slide number only to the published zoo
  # value; that warning was written when every attack path here used whole,
  # and the rule it encodes is really "clean and attacked must agree on the
  # inference mode". They now agree on slide.
  # Check the recorded config before quoting one beside these rows; if it
  # does not match, put the row back:
  #
  #   printf '%s\t%s\t%s\t%s\n' \
  #     "clean" "results/tables/bench_nesti/$ARCH" "scripts/clean_baseline.py" \
  #     "--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048 \
  #      --images all --out results/tables/bench_nesti/$ARCH/clean --tag bench_nesti"
  #
  # (the driver still has the sentinel branch for it, so re-adding the row
  # is the only change needed.)

  printf '%s\t%s\t%s\t%s\n' "csf" "$OUTROOT/csf" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode universal_csf --csf_threshold 0.25 --lr 0.01 --tag $T"
  printf '%s\t%s\t%s\t%s\n' "csf_null" "$OUTROOT/csf_null" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode universal_csf --csf_threshold 0.25 --lr 0 --epochs 1 --tag ${T}_null"
  printf '%s\t%s\t%s\t%s\n' "raw" "$OUTROOT/raw" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode raw --lr 0.1 --tag $T"
  printf '%s\t%s\t%s\t%s\n' "raw_null" "$OUTROOT/raw_null" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode raw --raw_init random --lr 0 --epochs 1 --tag ${T}_null"
  printf '%s\t%s\t%s\t%s\n' "raw_grey" "$OUTROOT/raw_grey" \
    "scripts/train.py" "$PROTO $GEOM --patch_mode raw --lr 0 --epochs 1 --tag ${T}_grey"
}

# ── --list: the driver's view of the table ───────────────────────────────────
# Printed BEFORE any module load, so the driver (a 1-CPU cpu-partition job)
# can read it without a GPU or a conda environment.
if [ "${1:-}" = "--list" ]; then
    configs | cut -f1,2
    exit 0
fi

WANT="${1:-}"
if [ -z "$WANT" ]; then
    # THE CONFIG NAME IS NOT OPTIONAL, and forgetting it is the single most
    # likely way to waste a submission -- `sbatch benchmark_slice.sh` is the
    # habit every other launch script in this repo teaches, and here it is
    # wrong. Spell out the fix rather than printing a bare usage line.
    {
      echo "usage: $0 <config-name> | --list"
      echo
      echo "  A config name is REQUIRED. You most likely ran"
      echo "      sbatch $(basename "$0")"
      echo "  where the other launch scripts in this repo take no argument."
      echo "  Use one of:"
      configs | cut -f1 | sed 's/^/      sbatch '"$(basename "$0")"' /'
      echo
      echo "  Or let the driver do it:  sbatch driver_benchmark.sh"
    } >&2
    exit 2
fi

echo "slice for '$WANT' on $(hostname), $(date)"
mkdir -p "$ROOT/slurm/benchmark"
# ARCH IS IN THE LOG NAME. Config names do not vary across architectures --
# six per-arch drivers all submit a slice called 'raw' -- so without it the
# logs are distinguishable only by job id, which is exactly the wrong thing
# to have to grep for when one arch is the one failing.
exec 1> "$ROOT/slurm/benchmark/${ARCH}_${WANT}_${SLURM_JOB_ID:-local}.out"
exec 2> "$ROOT/slurm/benchmark/${ARCH}_${WANT}_${SLURM_JOB_ID:-local}.err"

# ── environment ──────────────────────────────────────────────────────────────
# WHAT ACTUALLY PUTS PYTHON ON PATH HERE, and it is not the lines below.
# sbatch defaults to --export=ALL, so a job inherits the submitting shell's
# PATH: submit from a login shell with the thesis env active and python is
# already correct before any of this runs. That is why the other launch
# scripts in this repo work DESPITE these same three lines failing on uc3 --
# there is no `cuda/11.8` module, there is no ~/miniconda3, and the conda
# launcher in the workspace still carries a HoreKa shebang
# (/hkfs/.../miniconda3/bin/python) from when the install was copied across
# clusters. All three errors are non-fatal and always have been.
#
# Inheritance is fragile, though: a slice submitted from a shell without the
# env gets the system python and dies deep inside an import, which the driver
# can only read as "no progress" before giving up two slices later. So try to
# set the environment up, then VERIFY it, and refuse the slice with a legible
# message rather than burning thirty GPU-minutes on it.
#
# All three are overridable from the environment; sbatch exports them.
# CUDA_MODULE is EMPTY by default because the correct name is cluster
# specific -- `module spider cuda` on the login node, then export it.
CUDA_MODULE="${CUDA_MODULE:-}"
CONDA_SH="${CONDA_SH:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3}"

if [ -n "$CUDA_MODULE" ]; then
    module --ignore_cache load "$CUDA_MODULE" \
        || echo "[warn] module '$CUDA_MODULE' did not load; continuing"
fi
if [ -r "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    if source "$CONDA_SH" && conda activate "$CONDA_ENV"; then
        echo "[env ] conda activate $CONDA_ENV"
    else
        echo "[warn] conda activate failed; falling back to the inherited PATH"
    fi
else
    echo "[warn] no conda.sh at $CONDA_SH; relying on the inherited PATH"
fi
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib

cd "$ROOT" || exit 1
echo "cwd    : $(pwd)"
echo "python : $(command -v python || echo NONE)"

# THE PREFLIGHT. Everything above is best-effort; this is not. A slice that
# cannot import torch, or that got a GPU it cannot see, produces nothing and
# is indistinguishable to the driver from one that was merely slow.
if ! python - <<'PY'
import sys
print(f"exe    : {sys.executable}")
print(f"version: {sys.version.split()[0]}")
import torch
print(f"torch  : {torch.__version__}  cuda_available={torch.cuda.is_available()}")
sys.path.insert(0, ".")
import patchreach  # noqa: F401
print("patchreach: ok")
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible to torch")
PY
then
    echo
    echo "FATAL: the python environment is not usable in this job." >&2
    echo "  This slice would have failed deep inside an import, or run on" >&2
    echo "  CPU for its whole walltime. Nothing was started." >&2
    echo "  Checks, on the login node:" >&2
    echo "    which python && python -c 'import torch; print(torch.__version__)'" >&2
    echo "    module spider cuda        # then: export CUDA_MODULE=<name>" >&2
    echo "    ls $CONDA_SH" >&2
    echo "    head -1 \$(command -v conda)   # a /hkfs/... shebang is stale" >&2
    echo "  Simplest fix: activate the env in your login shell before" >&2
    echo "  submitting -- sbatch --export=ALL carries it into the job." >&2
    exit 3
fi

row="$(configs | awk -F'\t' -v n="$WANT" '$1 == n')"
if [ -z "$row" ]; then
    echo "FATAL: no config named '$WANT'. Known:" >&2
    configs | cut -f1 >&2
    exit 2
fi
out="$(printf '%s' "$row" | cut -f2)"
script="$(printf '%s' "$row" | cut -f3)"
args="$(printf '%s' "$row" | cut -f4)"

# clean_baseline.py has no --out_dir/--resume; its own --out carries the path.
if [ "$script" = "scripts/train.py" ]; then
    args="$args --out_dir $out"
fi

echo "=== $WANT ==="
echo "out    : $out"
echo "cmd    : python $script $args"
# shellcheck disable=SC2086
python "$script" $args
rc=$?
echo "slice exited $rc at $(date)"
exit $rc
