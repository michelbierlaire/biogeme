# Release workflow for examples and documentation

This is the release checklist for the executable examples in
`docs/source/examples`. It has two phases:

1. Run and repair every example on JED until the global status is `OK`.
2. Transfer the validated result fixtures to the laptop and build the
   documentation.

The JED checkout, laptop checkout, and generated fixtures must use the same
Git revision and locked `uv` environment. The runner discovers every
`plot_*.py` recursively and stores logs in the ignored `.jed_runs`
directory. That directory is internal bookkeeping; it is not part of the
release procedure.

The recommended interface is now provided by three incremental commands:

```text
jed_runs/release_examples.py   detect and register new examples
jed_runs/release_phase1.py     run and repair the JED suite
jed_runs/release_phase2.py     transfer artifacts and build the documentation
jed_runs/release_reset.py      remove all generated state for a fresh attempt
```

Release identifiers are stored automatically below `.jed_runs/releases`; the
user normally does not need to see or provide them.  The commands are dry runs
unless `--apply` is supplied, and every command ends by printing the next
action.

## Before starting

Choose the release commit and use it on both machines:

```bash
git rev-parse HEAD
git status --short
uv sync --frozen
uv run --locked --group docs python -c \
  "import sys, biogeme; print(sys.executable); print(biogeme.__file__)"
```

Authored files in the working tree should be clean; generated JED artifacts
may be present and are handled by the release wrapper as described below. On
JED, also confirm that no old Slurm job is still running:

```bash
squeue -u "$USER"
```

Do not reset generated files while a job is running.

Generated JED outputs are expected during this workflow. The release wrappers
recognize `saved_results`, `saved_html`, generated `.run`, `slurm-*`,
`*_slurm.out`, `revenue_*.txt`, and `test~*` files below the examples, plus the
narrow `biogeme-smoke-*.err`/`.out` diagnostics, as generated artifacts.
Authored changes such as Python, TOML, or documentation source edits still
stop the command and must be committed or stashed first. There is no
dirty-tree bypass in the official workflow.

If you intentionally want to begin from a completely clean release state,
use the reset script rather than removing files manually:

```bash
uv run --locked --group docs python jed_runs/release_reset.py --scope all
uv run --locked --group docs python jed_runs/release_reset.py \
    --scope all --apply --confirm
```

The first command is a dry run. The second removes generated result archives,
temporary outputs, diagnostics, caches, and runner state, but never source
code, input data, the manifest, or Git history. Do not apply it while Slurm
jobs are running. If archived results must be retained, use
`jed_commit_results.py --dry-run` and commit or stash them before applying the
reset. On a laptop, `squeue` is normally unavailable; the script warns and
continues only because `--confirm` explicitly acknowledges the reset. Confirm
on JED (for example with `squeue -u "$USER"`) that no jobs are running before
applying an `all` reset from the laptop.

If a previous JED run has already produced archived results that you want to
keep as a historical snapshot, inspect and commit only the reviewed fixtures
before starting a fresh release:

```bash
"$PY" jed_runs/jed_commit_results.py --dry-run
"$PY" jed_runs/jed_commit_results.py \
    --message "Preserve JED example results"
```

Alternatively, stash the generated files (including untracked files) for
later recovery:

```bash
git stash push --include-untracked -m "Biogeme generated release artifacts"
```

The root-level copies (including `biogeme-smoke-*.err`/`.out`) can be removed
without touching `saved_results` or `saved_html`:

```bash
"$PY" jed_runs/jed_cleanup.py
"$PY" jed_runs/jed_cleanup.py --apply
```

Do not use `jed_fresh_start.py` or `release_reset.py` when the archived results
must be retained. If the JED jobs have already completed, use
`release_phase1.py status` and then `release_phase1.py finalize --apply`; do
not invoke `release_phase1.py run` again unless starting a new release.

## Detecting and registering new examples

Run this check after adding a `plot_*.py` file and before starting Phase 1:

```bash
cd ~/github/biogeme
PY=.venv/bin/python

"$PY" jed_runs/release_examples.py --strict
```

The first invocation establishes an ignored inventory of the existing suite.
Later invocations detect new, changed, and removed examples.  A new example
must have a documentation label and, when it produces persistent results, a
complete output contract in `jed_runs/jed_examples.toml`.  The script prints a
proposed TOML block for anything that needs review.

After reviewing the proposal, register safe entries with:

```bash
"$PY" jed_runs/release_examples.py --apply
"$PY" jed_runs/release_examples.py --strict
```

Estimator examples with unknown output names are deliberately not registered
automatically.  Add their `expected_outputs` or `expected_output_globs` by
hand, then rerun the strict check.  Phase 1 performs the same check before
submission, so a new example cannot silently be omitted from a release.

## Phase 1 — Run and repair all examples on JED

Choose the path that matches the state of the JED checkout:

- For a new release, after the example inventory check and with no authored
  working-tree changes, inspect the submission plan with `run`.
- For a release that has already been submitted, use `status`; do not use
  `run` merely to inspect its progress. `run` is the start/resume submission
  command and performs the clean-tree safety check.

For a new release, inspect the plan:

```bash
"$PY" jed_runs/release_phase1.py run
```

Then execute it:

```bash
"$PY" jed_runs/release_phase1.py run --apply
```

When no previous JED attempts exist, the first applied run performs the fresh
generated-output cleanup and submits unfinished jobs. Repeating the same
command is safe: successful jobs are not resubmitted, and only `NOT_DONE` jobs
are selected. Generated artifacts already present in the checkout are accepted
by the wrapper and are either cleaned by the fresh-start step or retained when
existing JED attempts are adopted.
Preserve any old snapshot first if it must not be replaced.

If jobs were launched earlier with the lower-level `jed_examples.py` command,
the wrapper detects their recorded attempts and adopts them automatically. It
does not reset their state or rerun successful examples. This is also safe when
the earlier jobs are still running: use `release_phase1.py status` to monitor
them, and rerun `release_phase1.py run --apply` only after repairs or when
`NOT_DONE` jobs remain. To discard all old attempts intentionally, use the
explicit `release_reset.py --scope all` procedure below first.

If the command reports a dirty tree, inspect the paths in the message. Paths
under the generated-artifact allowlist may remain. Only authored or
unrecognized paths need action: commit or stash those files, then rerun the
same command. If disposable generated files need removal, use the documented
dry-run cleanup and then apply it. If the message shows only generated files,
the wrapper will proceed and will either adopt existing attempts or perform
the fresh-start cleanup.

Monitor the release with:

```bash
"$PY" jed_runs/release_phase1.py status
"$PY" jed_runs/release_phase1.py monitor --wait --poll-seconds 60
```

`status` is read-only and may be run at any time while jobs are queued or
running. It scans the global release state across all JED runs, so you do not
need to provide a run identifier. It prints the per-job status and a summary
such as `OK`, `ERROR`, `RUNNING`, `PENDING`, or `NOT_DONE`, followed by the
next recommended action. The equivalent lower-level command, useful when
more detail is needed, is:

```bash
"$PY" jed_runs/jed_examples.py status --verbose
```

Use the status result as follows:

- `OK`: Slurm completed successfully and every declared output was produced;
- `ERROR`: inspect the diagnostic and error report, repair the example, then
  invalidate it before resubmitting;
- `RUNNING` or `PENDING`: wait and run `release_phase1.py status` again;
- `NOT_DONE` or `NOT_SCHEDULED`: run `release_phase1.py run --apply` to submit
  the remaining work.

If a job fails, inspect the diagnostics, repair the source, invalidate the
failed job and its dependents, and rerun the wrapper:

```bash
"$PY" jed_runs/jed_error_report.py
"$PY" jed_runs/jed_examples.py invalidate \
    --script hybrid_choice_models/plot_h01_mode_logit.py
"$PY" jed_runs/release_phase1.py run --apply
```

The wrapper preserves the release state and resumes at the unfinished step.
When every discovered job is `OK`, finalize Phase 1:

```bash
"$PY" jed_runs/release_phase1.py finalize
"$PY" jed_runs/release_phase1.py finalize --apply
```

Finalization removes only temporary root-level copies and preserves the
archived files in `saved_results` and `saved_html`.

The lower-level commands remain available for diagnostics and exceptional
cases:

Start from a clean generated-output tree only when using the lower-level
commands directly. The dry run is required before the destructive step:

```bash
cd ~/github/biogeme
PY=.venv/bin/python

"$PY" jed_runs/jed_fresh_start.py
"$PY" jed_runs/jed_fresh_start.py --apply
```

This removes generated result files, archived fixtures, Slurm logs, caches,
diagnostics, and previous runner state while preserving source code and input
data. If old diagnostics may be needed, use
`jed_runs/jed_examples.py reset --apply` instead; it keeps a recoverable
backup.

Submit the unfinished examples. The first call submits the complete set; the
same command is used after every repair and submits only examples still marked
`NOT_DONE`:

```bash
"$PY" jed_runs/jed_examples.py launch --not-done --dry-run
"$PY" jed_runs/jed_examples.py launch --not-done
```

Dependencies are handled by the runner, and independent jobs are submitted in
parallel.

Monitor the iteration:

```bash
"$PY" jed_runs/jed_examples.py status
"$PY" jed_runs/jed_examples.py status --verbose
"$PY" jed_runs/jed_error_report.py
less .jed_runs/aggregate-error-report.md
```

The status meanings are:

- `OK`: Python, Slurm, and every declared output contract succeeded;
- `ERROR`: the job finished but failed or missed a declared output;
- `RUNNING`: the job is active or waiting in Slurm;
- `NOT_DONE`: it has not yet completed successfully.

`OK` also means that every declared output is present in the JED example's
`saved_results`/`saved_html` archive (or at its declared root location).  A
Slurm `COMPLETED` record by itself is not sufficient.  After updating the
runner, an older attempt that was previously shown as `OK` can therefore be
reported as `NOT_DONE` if its archive is missing; `release_phase1.py run
--apply` will submit only that example.

When an example fails:

1. Read the diagnostic report.
2. Fix the source or manifest on the release branch.
3. Mark the example unfinished; its dependent examples are marked unfinished
   automatically.
4. Submit the unfinished set again.

For example, after correcting `hybrid_choice_models/plot_h01_mode_logit.py`,
run:

```bash
"$PY" jed_runs/jed_examples.py invalidate \
    --script hybrid_choice_models/plot_h01_mode_logit.py
"$PY" jed_runs/jed_examples.py launch --not-done
```

`invalidate` changes the example from `ERROR` to `NOT_DONE`, removes its
declared stale result files, and marks dependent examples `NOT_DONE` as well.
The next `launch --not-done` therefore runs only the unfinished work.

Repeat this loop until the release gate succeeds:

```bash
"$PY" jed_runs/jed_examples.py status --require-all-ok
```

Do not transfer results before this command succeeds. A fast example may be
run directly on the laptop only as an exception; after verifying its declared
outputs, record it with:

```bash
uv run --locked --group docs python jed_runs/jed_examples.py mark-ok \
    --script family/plot_fixed.py --source laptop
```

## Phase 2 — Transfer results and build the documentation

On the laptop, the incremental wrapper combines the resumable transfer, the
strict manifest-limited import, and the documentation build.  A dry run is
shown first:

```bash
cd ~/MyFiles/github/biogeme
JED_REMOTE='bierlair@jed.epfl.ch:/home/bierlair/github/biogeme/docs/source/examples'

uv run --locked --group docs python jed_runs/release_phase2.py \
    run --source "$JED_REMOTE"
uv run --locked --group docs python jed_runs/release_phase2.py \
    run --source "$JED_REMOTE" --apply
```

The transfer uses `rsync --partial`, so an interrupted copy can be resumed by
rerunning the same command.  The import is strict and is attempted only after
all declared artifacts are available.  If the documentation build fails, run
only:

```bash
uv run --locked --group docs python jed_runs/release_phase2.py build --apply
```

If strict import reports missing artifacts, do not rerun individual examples
just for that message.  The phase-2 wrapper refreshes the persistent staging
directory from JED on every retry until import succeeds.  This matters when a
JED status was checked before the first transfer, or when an example writes a
declared YAML/HTML file at its workspace root.  Rerun the same `run --apply`
command after the JED results are available; the transfer is resumable and the
successful JED jobs are not resubmitted.

The phase-2 state is retained under `.jed_runs/releases`.  The wrapper never
commits changes; after a successful build, review `git status --short` and
commit manually.

The following lower-level commands describe the individual operations and
remain useful for troubleshooting.

Once every example is `OK`, remove only temporary root-level copies on JED.
The archived files in `saved_results` and `saved_html` are preserved:

```bash
"$PY" jed_runs/jed_cleanup.py
"$PY" jed_runs/jed_cleanup.py --apply
```

On the laptop, check out the same commit and remove old generated fixtures:

```bash
cd ~/MyFiles/github/biogeme
git rev-parse HEAD
git status --short
uv sync --frozen

uv run --locked --group docs python jed_runs/jed_fresh_start.py
uv run --locked --group docs python jed_runs/jed_fresh_start.py --apply
```

Stage the JED examples in the persistent, Git-ignored `.release_staging`
directory. NetCDF files are large posterior-draw archives and are not part of
the normal gallery fixture set. The single transfer below excludes every
`.nc` except the two files needed by downstream Bayesian examples:

```bash
JED_STAGE="$PWD/.release_staging/examples"
mkdir -p "$JED_STAGE"
JED_REMOTE='bierlair@jed.epfl.ch:/home/bierlair/github/biogeme/docs/source/examples'
rsync -a --partial --progress --whole-file \
    --delete \
    -e 'ssh -o Compression=no' \
    --include='*/' \
    --include='bayesian_swissmetro/saved_results/b01a_logit.nc' \
    --include='bayesian_swissmetro/saved_results/b05_normal_mixture.nc' \
    --exclude='*.nc' \
    --include='*/saved_results/***' \
    --include='*/saved_html/***' \
    --include='*.yaml' \
    --include='*.html' \
    --include='*.pareto' \
    --include='revenue_*.txt' \
    --exclude='*' \
    "$JED_REMOTE/" "$JED_STAGE/"
```

This single `rsync` invocation transfers only archived YAML/HTML/Pareto
fixtures, declared root-level YAML/HTML/Pareto/revenue reports, and the two NetCDF files
required by the gallery. It does not copy source code, input data, logs, or any
other NetCDF file. These are the only NetCDF files required by the gallery:
`plot_b01c_logit_simul.py`
uses the first for posterior-draw simulation, and
`plot_b19_individual_level_parameters.py` uses the second for observation-level
posterior means. All other Bayesian examples use YAML summaries. Because the
files are transferred in one SSH session, the passphrase is requested only
once. For unattended transfers, load the key into the macOS keychain first:

```bash
ssh-add --apple-use-keychain ~/.ssh/id_rsa
```

The staging directory is dedicated to release artifacts; `--delete` removes
stale selected files from an earlier transfer, while `--partial` preserves an
interrupted file for the next attempt. `--whole-file` avoids a delta-checksum pass. If
the transfer is interrupted, rerun the command; `--partial` keeps the partial
files. To let rsync resume a large partial file block by block, remove
`--whole-file` on the retry. Do not add `--compress` (`-z`): NetCDF is already
compressed and SSH compression usually makes this transfer slower.

The importer is manifest-limited. It copies only outputs declared in
`jed_runs/jed_examples.toml`; source code, input data, logs, caches, and
undeclared files are ignored. First perform a strict dry run:

```bash
uv run --locked --group docs python tools/import_jed_results.py \
    --profile all --strict
```

Every declared output must be available. If anything is missing, return to
Phase 1 and repair the relevant example. When the dry run is complete, apply
the import and replace stale archived result files safely:

```bash
uv run --locked --group docs python tools/import_jed_results.py \
    --profile all \
    --strict \
    --replace-results \
    --apply
```

The stage persists across terminal sessions and can be reused; rerunning
`rsync` updates changed files and preserves partial transfers. It is ignored by
Git; verify this with `git check-ignore -v "$JED_STAGE"`. Replaced files are
backed up below `.docs_runs/imports/`, and the importer
writes checksums to its `report.json`.

Build the gallery on the laptop:

```bash
make -C docs clean
make -C docs html PROFILE=full
make -C docs check-html
```

The full HTML target executes the gallery. A failure is a release blocker.
Review `docs/warnings.log`, then inspect the Git diff:

```bash
git status --short
git diff --check
git diff --name-status -- docs/source/examples
```

Commit only the reviewed source, manifest, documentation, and result changes.
Do not stage `.jed_runs`, `.docs_runs`, `.release_staging`, Slurm output,
caches, or other transfer directories.

## Starting over completely

If a genuinely fresh release is required, inspect the complete reset plan:

```bash
uv run --locked --group docs python jed_runs/release_reset.py --scope all
```

The scopes are `jed`, `laptop`, and `all`.  The reset removes only generated
artifacts, release state, staging files, documentation build output, and
caches.  It never removes source code, input data, the manifest, or the Git
repository.  It refuses to clean the JED scope while Slurm jobs are running.

Apply the reviewed plan explicitly:

```bash
uv run --locked --group docs python jed_runs/release_reset.py \
    --scope all --apply --confirm
```

After a reset, the next release starts with `release_examples.py --apply` to
establish a new ignored example inventory.

## Adding a new example

The runner automatically discovers a new `plot_*.py`, but the example still
needs a manifest entry when it has outputs or dependencies.

1. Add `plot_<name>.py` below the appropriate
   `docs/source/examples/<family>` directory.
2. Keep it hermetic: use paths relative to the example, keep input data in the
   repository, force a headless plotting backend when necessary, and protect
   multiprocessing or PyMC execution with a main guard.
3. Add an entry to `jed_runs/jed_examples.toml` when the example needs a
   dependency, a non-default resource profile, or persistent output files:

   ```toml
   [docs.examples."family/plot_<name>.py"]
   profile = "full"
   expected_outputs = ["model.yaml", "model.html"]

   [jobs."family/plot_<name>.py"]
   profile = "standard"
   ```

   Use `expected_output_globs` for runtime-generated names. Do not include
   `saved_results/` or `saved_html/` in the declared name; those are archive
   destinations managed by the runner.
4. If it consumes another example's result, declare both the dependency and
   the required input:

   ```toml
   [jobs."family/plot_consumer.py"]
   depends_on = ["family/plot_producer.py"]
   required_inputs = ["saved_results/model.yaml"]
   ```

5. Run the release-suite detector. It will propose safe entries and flag
   output contracts that need human review:

   ```bash
   "$PY" jed_runs/release_examples.py --strict
   "$PY" jed_runs/release_examples.py --apply
   "$PY" jed_runs/release_examples.py --strict
   ```

6. Test it locally in an isolated workspace:

   ```bash
   uv run --locked --group docs python tools/docs_examples.py \
       run --script family/plot_<name>.py --keep-workspace
   ```

7. Commit the source and manifest changes, then let Phase 1 execute the new
   example on JED. The strict Phase 2 import and full gallery build are
   required before release.

## Release gate

A release is ready only when:

- `status --require-all-ok` succeeds on JED;
- the strict importer finds every declared output;
- the importer checksum report is retained;
- `make -C docs html PROFILE=full` succeeds;
- `make -C docs check-html` succeeds; and
- the Git diff contains only intentional changes.

For lower-level diagnostics, see
[`docs/source/examples/JED_RUNS.md`](docs/source/examples/JED_RUNS.md).
