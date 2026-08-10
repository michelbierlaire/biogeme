#!/bin/bash -l
set -euo pipefail

repository="$HOME/github/biogeme"
if [[ -n "${BIOGEME_JED_REPOSITORY:-}" ]]; then
    repository="$BIOGEME_JED_REPOSITORY"
fi

exec python "$repository/jed_runs/jed_examples.py" launch "$@"
