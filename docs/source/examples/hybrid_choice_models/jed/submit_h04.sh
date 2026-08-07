#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 SLURM_ACCOUNT {preflight|full} [BIOGEME_REPOSITORY]" >&2
    exit 2
fi

slurm_account="$1"
profile="$2"
repository_directory="${3:-$HOME/biogeme}"
git_commit="$(git -C "$repository_directory" rev-parse HEAD)"
run_base="${BIOGEME_H04_RUN_BASE:-/scratch/$USER/biogeme-h04}"
script="$repository_directory/docs/source/examples/hybrid_choice_models/jed/h04.sbatch"

partition="${BIOGEME_SLURM_PARTITION:-standard}"
qos="${BIOGEME_SLURM_QOS:-serial}"

case "$profile" in
    preflight)
        cpus=4
        memory=28G
        wall_time=01:00:00
        draws=1000
        iterations=3
        ;;
    full)
        cpus=16
        memory=112G
        wall_time=2-00:00:00
        draws=50000
        iterations=5000
        ;;
    *)
        echo "Unknown profile '$profile'; use preflight or full." >&2
        exit 2
        ;;
esac

cpus="${BIOGEME_H04_CPUS:-$cpus}"
memory="${BIOGEME_H04_MEMORY:-$memory}"
wall_time="${BIOGEME_H04_WALL_TIME:-$wall_time}"
draws="${BIOGEME_H04_NUMBER_OF_DRAWS:-$draws}"
iterations="${BIOGEME_H04_MAX_ITERATIONS:-$iterations}"

run_root="$run_base/$git_commit/$profile"
mkdir -p "$run_root/logs"

sbatch \
    --account="$slurm_account" \
    --partition="$partition" \
    --qos="$qos" \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$wall_time" \
    --output="$run_root/logs/%x-%j.out" \
    --error="$run_root/logs/%x-%j.err" \
    --export="ALL,BIOGEME_REPOSITORY=$repository_directory,BIOGEME_H04_RUN_ROOT=$run_root,BIOGEME_H04_NUMBER_OF_DRAWS=$draws,BIOGEME_H04_MAX_ITERATIONS=$iterations" \
    "$script"

echo "Run directory: $run_root"
