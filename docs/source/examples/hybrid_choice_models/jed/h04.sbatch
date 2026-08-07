#!/bin/bash -l
#SBATCH --job-name=biogeme-h04
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

: "${BIOGEME_REPOSITORY:?Set BIOGEME_REPOSITORY to the cloned Biogeme repository}"
: "${BIOGEME_H04_RUN_ROOT:?Set BIOGEME_H04_RUN_ROOT to persistent shared storage}"

exec "$BIOGEME_REPOSITORY/docs/source/examples/hybrid_choice_models/jed/run_h04.sh"
