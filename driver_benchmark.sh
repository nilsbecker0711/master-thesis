#!/bin/bash
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000

# Feeds short dev-GPU slices back to back until the whole benchmark is done.
#
#     sbatch driver_benchmark.sh              (from the repo root)
#     sbatch driver_benchmark.sh csf_212      one config only
#
# Each slice runs benchmark_slice.sh, which passes --resume and picks up from
# train_state.pt, so this is idempotent: relaunch the driver as often as
# needed. Same shape as the population driver, with one structural change —
# a universal patch is ONE continuous optimisation rather than N independent
# per-image attacks, so the transparency of the cut is a property of
# train.py's checkpoint (param + Adam moments + cosine position + gstep +
# RNG), not of the driver. tests/test_train_resume.py pins that: a sliced
# run reproduces an uninterrupted one bit for bit.
#
# NOT on a timer. sbatch --wait blocks until the slice TERMINATES, queue time
# included, and the next one goes in immediately after.
#
# THE CONFIG TABLE IS NOT DUPLICATED HERE. It is read back from
# `benchmark_slice.sh --list`, so an out_dir cannot drift between the driver
# and the job — the failure the population driver needs a comment to catch.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

echo "driver on $(hostname), $(date)"
mkdir -p slurm/benchmark
exec 1> "slurm/benchmark/driver_${SLURM_JOB_ID:-local}.out"
exec 2> "slurm/benchmark/driver_${SLURM_JOB_ID:-local}.err"

JOB="benchmark_slice.sh"
# Literal, so no editor can turn the field separator into spaces.
TAB=$'	'
ONLY="${1:-}"            # optional: run a single config by name

# EVERY resource goes on the command line, not left to $JOB's #SBATCH lines.
# sbatch precedence is: command line > environment > script directives, and
# this driver runs with SLURM_CPUS_PER_TASK=1 and SLURM_MEM_PER_NODE=2000 in
# its own environment. Inherited, those would silently override the slice's
# #SBATCH --cpus-per-task=16 / --mem=40000 and hand it 1 CPU and 2 GB.
SLICE="-p dev_gpu_a100_il -t 00:30:00 --gres=gpu:1 -N 1 -n 1 \
       --cpus-per-task=16 --mem=40000"

# ── preflight: fail now, loudly, rather than in 3 hours ──────────────────────
if ! command -v sbatch >/dev/null; then
    echo "FATAL: no sbatch on this node - this cluster may not allow"
    echo "       submitting from a compute node. Run the driver on the login"
    echo "       node with nohup instead."
    exit 1
fi
if [ ! -f "$JOB" ]; then
    echo "FATAL: $JOB not found. cwd is $(pwd); sbatch driver_benchmark.sh"
    echo "       from the repo root."
    exit 1
fi
if ! bash "$JOB" --list >/dev/null 2>&1; then
    echo "FATAL: '$JOB --list' failed. Run it by hand and fix it first."
    exit 1
fi

# name<TAB>out_dir, one per config, in the order they should be run.
mapfile -t ROWS < <(bash "$JOB" --list)
if [ "${#ROWS[@]}" -eq 0 ]; then
    echo "FATAL: the config table is empty."
    exit 1
fi

# PROGRESS IS COUNTED IN EPOCHS ON DISK. train.py writes one PNG per epoch
# into intermediate_patches/, so the count is readable from bash without
# torch. results.json is the AUTHORITATIVE done signal — it is written once,
# at the very end, after the final 500-image evaluation. The two can differ:
# a slice killed after the last epoch but before that evaluation leaves the
# epoch count topped out with no results.json, and the next slice finishes
# the tail. That is why the sentinel, not the count, ends a config.
epochs_done () {
    local d="$1"
    if [ -d "$d/intermediate_patches" ]; then
        find "$d/intermediate_patches" -name 'epoch*.png' | wc -l
    else
        echo 0
    fi
}
finished () {
    local name="$1" d="$2"
    if [ "$name" = "clean" ]; then
        # clean_baseline.py writes one json rather than a run directory.
        ls "$d"/clean*bench_nesti*.json >/dev/null 2>&1
    else
        [ -f "$d/results.json" ]
    fi
}

# Belt and braces on top of $SLICE: strip the driver's own SLURM_* so nothing
# at all is inherited. The subshell keeps the driver's own environment intact.
submit_slice () {
    ( for v in ${!SLURM_@}; do unset "$v"; done
      sbatch --wait $SLICE "$JOB" "$1" )
}

echo "job     : $JOB"
echo "slice   : $SLICE"
echo "configs : ${#ROWS[@]}"
[ -n "$ONLY" ] && echo "only    : $ONLY"
echo

total_slices=0
for row in "${ROWS[@]}"; do
    name="${row%%${TAB}*}"
    out="${row#*${TAB}}"
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue

    if finished "$name" "$out"; then
        echo "$(date +%F\ %H:%M)  $name  already complete, skipping"
        continue
    fi
    echo "$(date +%F\ %H:%M)  === $name -> $out ==="

    stalled=0
    prev=-1
    while :; do
        if finished "$name" "$out"; then
            echo "$(date +%F\ %H:%M)  $name  COMPLETE after $(epochs_done "$out") epochs"
            break
        fi
        n=$(epochs_done "$out")
        # Two consecutive slices with no new epoch means it is failing, not
        # slow. ONE is allowed: at 1024x2048 a slice that spends its first
        # minutes on conda, the model load and cudnn warmup can legitimately
        # finish no epoch. This also catches a config whose very first slice
        # dies at startup — nothing is ever written, so the driver gives up
        # after 3 submissions instead of looping for 72 hours.
        if [ "$n" -eq "$prev" ]; then
            stalled=$((stalled + 1))
            if [ "$stalled" -ge 2 ]; then
                echo "$(date +%F\ %H:%M)  $name  no progress in 2 slices at"
                echo "  epoch $n - giving up on this config and moving on."
                echo "  check slurm/benchmark/${name}_*.err"
                break
            fi
        else
            stalled=0
        fi
        prev=$n

        echo "$(date +%F\ %H:%M)  $name  epoch $n  submitting slice"
        submit_slice "$name"; rc=$?
        total_slices=$((total_slices + 1))
        echo "$(date +%F\ %H:%M)  $name  slice returned $rc"
        sleep 10          # never spin, in case sbatch rejects instantly
    done
done

echo
echo "driver done after $total_slices slices, $(date)"
for row in "${ROWS[@]}"; do
    name="${row%%${TAB}*}"; out="${row#*${TAB}}"
    if finished "$name" "$out"; then s="done"; else s="INCOMPLETE"; fi
    printf '  %-18s %-10s %s\n' "$name" "$s" "$out"
done
echo
echo "Index the rows:"
echo "  python analysis/build_index.py --runs results/runs/bench_nesti \\"
echo "      --out results/tables/bench_nesti/index.csv"
