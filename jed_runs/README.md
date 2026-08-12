# Running Biogeme examples on JED

The scripts in this directory discover the Biogeme example programs, generate
Slurm batch scripts, submit dependency-aware jobs with `sbatch`, and collect
diagnostics. Run them from the repository root, using the repository's
`.venv`, not the golden-reference environment under `tests/`.

For the complete release checklist, including laptop result import and the
Sphinx-gallery build, see `RELEASE_WORKFLOW.md` in the repository root.

## Prepare the checkout

On JED:

```bash
cd /home/bierlair/github/biogeme
git pull --ff-only

# Verify that the checkout and its environment are being used.
.venv/bin/python -c "import sys, biogeme; print(sys.executable); print(biogeme.__file__)"
```

If the environment has not yet been created, create/synchronise it from the
repository root with `uv sync --frozen`. Do not run `uv` from a separate
project directory; the golden-reference project is intentionally isolated
under `tests/golden_reference/`.

## Start a run

Make sure all jobs from an earlier run have finished before resetting outputs.
For a recoverable reset, first inspect and then apply:

```bash
cd /home/bierlair/github/biogeme
PY=.venv/bin/python

$PY jed_runs/jed_examples.py reset --dry-run
$PY jed_runs/jed_examples.py reset --apply
```

Start or continue the release workload. The runner manages its internal state
automatically and selects only unfinished examples:

```bash
$PY jed_runs/jed_examples.py launch --not-done --dry-run
```

When the generated scripts and resource settings are correct, submit with
`sbatch` through the runner:

```bash
$PY jed_runs/jed_examples.py launch --not-done
```

The resource profiles and explicit dependency overrides are in
`jed_runs/jed_examples.toml`. A subset can be launched with `--only`, for
example:

```bash
$PY jed_runs/jed_examples.py launch --only \
  hybrid_choice_models/plot_h04_mode_lv_gauss_simult.py \
  --dry-run
```

To launch every non-light resource profile without maintaining a manual list,
use `--slow`; declared dependencies are included automatically. This is a
diagnostic option, not part of the normal release loop:

```bash
$PY jed_runs/jed_examples.py launch --slow \
  --dry-run
$PY jed_runs/jed_examples.py launch --slow \
  --force
```

Each submitted job copies its example into a job-specific temporary work
directory (`$SLURM_TMPDIR`, or `$TMPDIR`/`/tmp`), runs there, and harvests only
that job's outputs back into `saved_results` or `saved_html`. Declared
root-level text reports (for example, `revenue_1.00.txt`) remain at the
example root. This prevents concurrent jobs from seeing or harvesting one
another's artifacts.

## Monitor and diagnose

```bash
$PY jed_runs/jed_examples.py status --verbose
squeue -u "$USER"

# The global report scans all recorded attempts automatically.
$PY jed_runs/jed_error_report.py
less .jed_runs/aggregate-error-report.md
```

Generated `.run` files, Slurm logs, completion records, and reports are kept
under `.jed_runs/`. Individual generated scripts can also be submitted with
`sbatch` when needed; their paths are printed by the launch command.

After reviewing a completed run, root-level generated artifacts can be
cleaned separately:

```bash
$PY jed_runs/jed_cleanup.py
$PY jed_runs/jed_cleanup.py --apply
```

## Commit archived results

The JED runner archives outputs in `saved_results` and `saved_html`.  After
reviewing a completed run, use the commit helper to stage and commit all such
archives below `docs/source/examples`:

```bash
cd /home/bierlair/github/biogeme
PY=.venv/bin/python

$PY jed_runs/jed_commit_results.py --dry-run
$PY jed_runs/jed_commit_results.py \
  --message "Update JED example results"
```

The helper commits only files inside `saved_results` and `saved_html`.  For a
release, NetCDF files should be force-added only when they are declared in the
manifest as required posterior-draw inputs (`b01a_logit.nc` and
`b05_normal_mixture.nc`); all other Bayesian examples use YAML summaries.  It
refuses to proceed if unrelated files are already staged, so unstage other
work before running it.  It force-adds files in those two directories because
some result formats, such as NetCDF, are ignored by general repository rules.
The Git user identity must be configured in the checkout; the dry run is safe
and does not change the index.

For a completely fresh, non-recoverable cleanup, use
`jed_runs/jed_fresh_start.py` only after all Slurm jobs have finished.
