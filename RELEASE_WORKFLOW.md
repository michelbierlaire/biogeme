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

## Before starting

Choose the release commit and use it on both machines:

```bash
git rev-parse HEAD
git status --short
uv sync --frozen
.venv/bin/python -c "import sys, biogeme; print(sys.executable); print(biogeme.__file__)"
```

The working tree should be clean apart from ignored runtime directories. On
JED, also confirm that no old Slurm job is still running:

```bash
squeue -u "$USER"
```

Do not reset generated files while a job is running.

## Phase 1 — Run and repair all examples on JED

Start from a clean generated-output tree. The dry run is required before the
destructive step:

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
"$PY" jed_runs/jed_examples.py mark-ok \
    --script family/plot_fixed.py --source laptop
```

## Phase 2 — Transfer results and build the documentation

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

Stage the JED examples in a fresh temporary directory. NetCDF files are
large posterior-draw archives and are not part of the normal gallery fixture
set. The single transfer below excludes every `.nc` except the two files
needed by downstream Bayesian examples:

```bash
JED_STAGE_ROOT="$(mktemp -d /tmp/biogeme-jed-results.XXXXXX)"
JED_STAGE="$JED_STAGE_ROOT/examples"
mkdir -p "$JED_STAGE"
JED_REMOTE='bierlair@jed.epfl.ch:/home/bierlair/github/biogeme/docs/source/examples'
rsync -a --partial --progress --whole-file \
    -e 'ssh -o Compression=no' \
    --include='*/' \
    --include='bayesian_swissmetro/saved_results/b01a_logit.nc' \
    --include='bayesian_swissmetro/saved_results/b05_normal_mixture.nc' \
    --exclude='*.nc' \
    --include='*/saved_results/***' \
    --include='*/saved_html/***' \
    --include='revenue_*.txt' \
    --exclude='*' \
    "$JED_REMOTE/" "$JED_STAGE/"
```

This single `rsync` invocation transfers only archived YAML/HTML/Pareto
fixtures, the declared root-level revenue reports, and the two NetCDF files
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

For a fresh staging directory, `--whole-file` avoids a delta-checksum pass. If
the transfer is interrupted, rerun the command; `--partial` keeps the partial
files. To let rsync resume a large partial file block by block, remove
`--whole-file` on the retry. Do not add `--compress` (`-z`): NetCDF is already
compressed and SSH compression usually makes this transfer slower.

The importer is manifest-limited. It copies only outputs declared in
`jed_runs/jed_examples.toml`; source code, input data, logs, caches, and
undeclared files are ignored. First perform a strict dry run:

```bash
uv run --locked --group docs python tools/import_jed_results.py \
    --source "$JED_STAGE" --profile all --strict
```

Every declared output must be available. If anything is missing, return to
Phase 1 and repair the relevant example. When the dry run is complete, apply
the import and replace stale archived result files safely:

```bash
uv run --locked --group docs python tools/import_jed_results.py \
    --source "$JED_STAGE" \
    --profile all \
    --strict \
    --replace-results \
    --apply
```

Replaced files are backed up below `.docs_runs/imports/`, and the importer
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
Do not stage `.jed_runs`, `.docs_runs`, Slurm output, caches, or temporary
transfer directories.

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

5. Test it locally in an isolated workspace:

   ```bash
   uv run --locked --group docs python tools/docs_examples.py \
       run --script family/plot_<name>.py --keep-workspace
   ```

6. Commit the source and manifest changes, then let Phase 1 execute the new
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
