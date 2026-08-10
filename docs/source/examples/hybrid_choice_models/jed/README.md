# Running H04 on EPFL Jed

These files run H04 from a pinned Biogeme checkout and keep results on shared
storage. Re-submitting the same Git commit and profile uses the same YAML file:
`estimate_or_load` loads a completed result, resumes post-estimation calculations
from an incomplete result, and estimates the model when no result exists.

## 1. Connect and install

Connect through the EPFL network or VPN:

```bash
ssh GASPAR_USERNAME@jed.hpc.epfl.ch
```

Clone the development branch or exact commit that contains the Hessian changes.
The reference must have been pushed to GitHub first.

```bash
git clone https://github.com/michelbierlaire/biogeme.git "$HOME/github/biogeme"
cd "$HOME/github/biogeme"
git checkout GIT_REFERENCE
```

Install `uv` in the user account if it is not already available, then create the
locked environment:

```bash
docs/source/examples/hybrid_choice_models/jed/setup_biogeme.sh \
    "$HOME/github/biogeme" GIT_REFERENCE
```

The repository requires Python 3.12 or newer. `uv sync --frozen` uses the
committed lock file and creates `$HOME/github/biogeme/.venv`.

## 2. Submit the preflight

The supplied research job uses Jed's `standard` partition and `serial` QoS. If
your allocation requires an explicit account, add
`#SBATCH --account=YOUR_ACCOUNT` to the `.run` file:

```bash
cd /home/bierlair/github/biogeme
sbatch docs/source/examples/hybrid_choice_models/plot_h04_mode_lv_gauss_simult_preflight.run
```

The preflight uses 1,000 draws and three optimization iterations. It validates
the environment, data, optimizer, result checkpoint, and chunked Hessian path;
it is not a statistical result.

Academic users select the academic partition and QoS at submission time:

Academic users must copy the `.run` file and change both `--partition` and
`--qos` to `academic` before submitting it.

Inspect the job with `squeue --me`, its log under
`/scratch/$USER/biogeme-h04/COMMIT/preflight/logs`, and the completed accounting
record with `Sjob JOB_ID` or `sacct -j JOB_ID`.

## 3. Submit the full run

After the preflight succeeds:

```bash
cd /home/bierlair/github/biogeme
sbatch docs/source/examples/hybrid_choice_models/plot_h04_mode_lv_gauss_simult.run
```

The full profile requests 18 CPU cores, 7,000 MB per core, and 48 hours. This
provides approximately 126,000 MB for the 50,000-draw calculation. Review
elapsed time, peak memory, and CPU efficiency afterward, then revise the
corresponding `#SBATCH` directives using the measured values.

By default, persistent files live below
`/scratch/$USER/biogeme-h04/GIT_COMMIT/{preflight,full}`. Set
`BIOGEME_H04_RUN_BASE` before submission to use another shared filesystem.
Never put the result directory only in node-local temporary storage.

## Restart and result semantics

The submitted source is copied into an isolated work directory, excluding the
example's bundled `saved_results`. Therefore a first run cannot accidentally
load the documentation result. The H04 example writes to the result directory
selected by `BIOGEME_H04_RESULTS_DIRECTORY`:

- no YAML: perform optimization and write the result;
- incomplete YAML: reuse saved raw estimates and complete the missing work;
- completed YAML: load it without estimating again.

The run directory records `git-commit.txt`. Use a new commit for changed code;
do not mix results produced by different source revisions. The job refuses to
run if the checkout contains tracked, uncommitted modifications.
