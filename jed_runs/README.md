# Running Biogeme examples on JED

The scripts in this directory discover the Biogeme example programs, generate
Slurm batch scripts, submit dependency-aware jobs with `sbatch`, and collect
diagnostics. Run them from the repository root, using the repository's
`.venv`, not the golden-reference environment under `tests/`.

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

Generate the complete dependency graph without submitting it:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
$PY jed_runs/jed_examples.py launch --dry-run --run-id "$RUN_ID"
```

When the generated scripts and resource settings are correct, submit with
`sbatch` through the runner:

```bash
$PY jed_runs/jed_examples.py launch --run-id "$RUN_ID" --force
```

The resource profiles and explicit dependency overrides are in
`jed_runs/jed_examples.toml`. A subset can be launched with `--only`, for
example:

```bash
$PY jed_runs/jed_examples.py launch --only \
  hybrid_choice_models/plot_h04_mode_lv_gauss_simult.py \
  --dry-run --run-id "$RUN_ID"
```

Each submitted job copies its example into a job-specific temporary work
directory (`$SLURM_TMPDIR`, or `$TMPDIR`/`/tmp`), runs there, and harvests only
that job's outputs back into `saved_results` or `saved_html`. This prevents
concurrent jobs from seeing or harvesting one another's artifacts.

## Monitor and diagnose

```bash
$PY jed_runs/jed_examples.py status --verbose
squeue -u "$USER"

# After a failure, inspect the newest run or select one explicitly.
$PY jed_runs/jed_error_report.py --run-id "$RUN_ID"
less ".jed_runs/$RUN_ID/error-report.md"
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

For a completely fresh, non-recoverable cleanup, use
`jed_runs/jed_fresh_start.py` only after all Slurm jobs have finished.
