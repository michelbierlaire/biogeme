#!/bin/bash
set -euo pipefail

: "${BIOGEME_REPOSITORY:?BIOGEME_REPOSITORY is required}"
: "${BIOGEME_H04_RUN_ROOT:?BIOGEME_H04_RUN_ROOT is required}"

python_executable="$BIOGEME_REPOSITORY/.venv/bin/python"
source_directory="$BIOGEME_REPOSITORY/docs/source/examples/hybrid_choice_models"
work_directory="$BIOGEME_H04_RUN_ROOT/work"
results_directory="$BIOGEME_H04_RUN_ROOT/results"
cache_directory="$BIOGEME_H04_RUN_ROOT/cache"
job_temporary_directory="${SLURM_TMPDIR:-/tmp/${USER}/biogeme-${SLURM_JOB_ID:-interactive}}"

if [[ ! -x "$python_executable" ]]; then
    echo "Missing $python_executable; run jed/setup_biogeme.sh first." >&2
    exit 1
fi
if ! git -C "$BIOGEME_REPOSITORY" diff --quiet || \
    ! git -C "$BIOGEME_REPOSITORY" diff --cached --quiet; then
    echo "The Biogeme checkout has tracked modifications; refusing an unpinned run." >&2
    exit 1
fi

mkdir -p "$work_directory" "$results_directory" "$cache_directory/jax" \
    "$job_temporary_directory/xdg" "$job_temporary_directory/matplotlib"
rsync -a \
    --exclude='saved_results*/' \
    --exclude='saved_html*/' \
    --exclude='__pycache__/' \
    --exclude='*.run' \
    "$source_directory/" "$work_directory/"
ln -sfn "$results_directory" "$work_directory/saved_results"

export JAX_COMPILATION_CACHE_DIR="$cache_directory/jax"
export XDG_CACHE_HOME="$job_temporary_directory/xdg"
export MPLCONFIGDIR="$job_temporary_directory/matplotlib"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

git -C "$BIOGEME_REPOSITORY" rev-parse HEAD > "$BIOGEME_H04_RUN_ROOT/git-commit.txt"
"$python_executable" --version
echo "Commit: $(<"$BIOGEME_H04_RUN_ROOT/git-commit.txt")"
echo "Results: $results_directory"
echo "Started: $(date --iso-8601=seconds)"

cd "$work_directory"
srun --ntasks=1 "$python_executable" -u plot_h04_mode_lv_gauss_simult.py

echo "Finished: $(date --iso-8601=seconds)"
