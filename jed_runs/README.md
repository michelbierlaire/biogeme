# Running Biogeme examples on JED

The scripts in this directory discover the Biogeme example programs, generate
Slurm batch scripts, submit dependency-aware jobs with `sbatch`, and collect
diagnostics. Run them from the repository root, using the repository's
`.venv`, not the golden-reference environment under `tests/`.

For the complete release checklist, including laptop result import and the
Sphinx-gallery build, see `RELEASE_WORKFLOW.md` in the repository root.

The incremental release wrappers are:

```text
release_examples.py  detect and register new plot_*.py examples
release_phase1.py    prepare, submit, monitor, and finalize the JED phase
release_phase2.py    transfer, clean, import, and build the laptop documentation
release_reset.py     remove generated state for a completely fresh attempt
```

They are dry runs unless `--apply` is supplied.  Release identifiers and
resume state are stored automatically below the ignored `.jed_runs/releases`
directory.

On the laptop, use `release_phase2.py run` for the normal Phase 2 workflow. It
owns the persistent `.release_staging` directory, the filtered resumable
transfer, the strict manifest import, and the full documentation build. Do not
create the staging directory or invoke `rsync` manually unless diagnosing an
exception; the emergency recipe is documented in `RELEASE_WORKFLOW.md`.

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

Generated JED outputs are normal during a release. The wrapper recognizes
`saved_results`, `saved_html`, generated `.run`, `slurm-*`, `*_slurm.out`,
`revenue_*.txt`, and `test~*` files below the examples, plus the narrow
root-level `biogeme-smoke-*.err`/`.out` diagnostics. Authored Python, TOML,
and documentation changes still block the command and must be committed or
stashed; there is no dirty-tree bypass in the release workflow. If a previous
run left archived results that must be kept as a historical snapshot, preserve them before
starting a fresh release by reviewing and committing the archive directories,
or stash generated files:

```bash
$PY jed_runs/jed_commit_results.py --dry-run
$PY jed_runs/jed_commit_results.py --message "Preserve JED example results"

# Alternative: preserve generated files in a recoverable Git stash.
git stash push --include-untracked -m "Biogeme generated release artifacts"
```

For a genuinely new release, inspect and apply the guarded full reset instead
of deleting files manually:

```bash
$PY jed_runs/release_reset.py --scope all
$PY jed_runs/release_reset.py --scope all --apply --confirm
```

It removes generated archives, outputs, diagnostics, caches, and runner state
only; it never removes source code or input data. Do not apply it while Slurm
jobs are running. On a laptop, `squeue` is normally unavailable; the script
warns and continues only because `--confirm` explicitly acknowledges the
reset. Before applying an `all` reset from the laptop, confirm on JED with
`squeue -u "$USER"` that no jobs are running. Preserve any historical result
snapshot first.

`jed_cleanup.py --apply` removes only root-level generated copies (including
the narrow `biogeme-smoke-*.err`/`.out` diagnostics) and keeps the two archive
directories. Do not use `jed_fresh_start.py` or
`release_reset.py` if those archived results must be retained. If the jobs
already finished, inspect them with `release_phase1.py status` rather than
calling `release_phase1.py run` again.

## Start a run

For a new release, first check the example inventory and inspect the Phase-1
plan. Generated JED outputs may remain in the checkout; authored changes must
still be committed or stashed:

```bash
$PY jed_runs/release_examples.py --strict
$PY jed_runs/release_phase1.py run
$PY jed_runs/release_phase1.py run --apply
```

The same applied command can be repeated after repairs; it submits only
unfinished jobs.  Use `release_phase1.py status` and
`release_phase1.py monitor --wait` while the jobs run.  After all jobs are
`OK`, use `release_phase1.py finalize --apply` before transferring results.

If jobs were started earlier with `jed_examples.py`, `release_phase1.py run`
adopts the recorded attempts instead of resetting them. Successful examples
are therefore preserved and only unfinished jobs can be submitted. Use
`release_reset.py --scope all` only when an entirely fresh attempt is intended.

If the dirty-tree check reports a mixture of paths, generated result files may
remain; commit or stash only the authored or unrecognized paths it lists. For
disposable generated files, use the documented dry-run cleanup before applying
it; the release commands do not bypass the clean-tree check.

If a release has already been submitted, start with the read-only status
command instead of `release_phase1.py run`:

```bash
$PY jed_runs/release_phase1.py status
```

### Check the JED results

The status command does not submit or restart anything.  It reads the runner
state and prints the status of every discovered example, together with a
summary and the next recommended action:

```bash
$PY jed_runs/release_phase1.py status
```

Use it after the jobs have finished, or periodically while they are running.
For a continuously monitored run, use:

```bash
$PY jed_runs/release_phase1.py monitor --wait --poll-seconds 60
```

When the summary contains only `OK`, finalize the JED phase before transferring
results:

```bash
$PY jed_runs/release_phase1.py finalize --apply
```

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

The JED status is not considered `OK` merely because Slurm returned
`COMPLETED`: every declared output must also be present in the shared archived
example tree.  Phase 2 transfers root-level YAML/HTML/Pareto reports as well
as files already below `saved_results`/`saved_html`; the importer then places
them in the canonical laptop directories.  If a strict import finds a missing
artifact, rerun the same phase-2 command.  Until import succeeds, it refreshes
the persistent staging directory from JED rather than reusing a stale transfer.

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

To discard all generated state and start from scratch, inspect and then apply
the scoped reset:

```bash
$PY jed_runs/release_reset.py --scope all
$PY jed_runs/release_reset.py --scope all --apply --confirm
```

The reset never removes source files, input data, or the manifest, and refuses
to clean the JED scope while Slurm jobs are running.

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
