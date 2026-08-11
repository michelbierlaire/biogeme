# Release preparation: examples and documentation

This runbook describes the repeatable process for preparing a Biogeme
release when the Sphinx documentation includes the executable examples
in `docs/source/examples`. It covers the two-machine workflow:

- JED runs the expensive examples in isolated Slurm jobs.
- The laptop receives the validated result fixtures and builds the
  website.

The source tree, the Python environment, and the result fixtures must
all refer to the same Git revision. Do not mix results produced from one
commit with documentation scripts from another commit.

The shared source of truth is `jed_runs/jed_examples.toml`. It describes
resource profiles, dependency edges, required input artifacts, and the
exact result files (or filename patterns) that may be imported into the
laptop checkout.

## Which document to use

Keep the two operational documents separate. This file is the end-to-end
release checklist: it explains the hand-off from JED to the laptop and the
final Sphinx build. `docs/source/examples/JED_RUNS.md` is the lower-level JED
reference: it describes runner commands, Slurm diagnostics, cleanup, and
result commits. `jed_runs/README.md` is the short entry point for the runner.

This separation avoids maintaining two copies of every command while keeping
the server reference close to the JED implementation. Update both documents
when a command or workflow changes, and link to the other document instead of
duplicating a long recipe.

## At a glance

The complete release sequence is:

1.  Select and record the release commit.
2.  Make clean checkouts on JED and on the laptop.
3.  Remove old generated results on JED.
4.  Run all examples, or only the slow examples, on JED.
5.  Monitor every Slurm job and investigate failures before importing
    anything.
6.  Remove old laptop fixtures, then import the new JED fixtures
    strictly.
7.  Run the fast hermetic check and build the full Sphinx gallery.
8.  Inspect the generated site and warnings, clean root-level generated
    files, and review the Git diff.
9.  Commit the reviewed fixtures and documentation changes on a release
    branch.

The commands below use `$HOME/github/biogeme` as the checkout on JED and
`$HOME/MyFiles/github/biogeme` as an example laptop path. Replace those
paths, the JED hostname, and the account name with the local values.

## 1. Freeze the release revision

Choose the commit that will become the release candidate. On both
machines, record the revision before running examples:

``` bash
git rev-parse HEAD
git status --short
```

The working tree should be clean apart from deliberately ignored runtime
directories such as `.venv`, `.jed_runs`, and `.docs_runs`. If local
source edits are present, stop and either commit them on the release
branch or use a clean worktree. Do not overwrite uncommitted work merely
to start a run.

On JED, update to the selected revision without merging unrelated local
work:

``` bash
cd "$HOME/github/biogeme"
git fetch origin
git switch <release-branch>
git pull --ff-only
git rev-parse HEAD
```

If the server checkout contains changes that must be discarded, preserve
them elsewhere first and use the repository's approved clean-checkout
procedure. The release run must never be based on an unknown mixture of
local changes.

Synchronize the locked environment from the repository root:

``` bash
uv sync --frozen
.venv/bin/python -c "import sys, biogeme; print(sys.executable); print(biogeme.__file__)"
```

The interpreter should be the checkout's `.venv/bin/python`. Do not use
the golden-reference environment under `tests/golden_reference` for JED
jobs. On the laptop, use the same `uv.lock` and verify the revision and
Biogeme import path in the same way.

## 2. Remove results from earlier releases

Never reset or delete files while a previous Slurm job is still running.
A running job can recreate files after the reset and can overwrite a new
run's fixtures.

Check JED first:

``` bash
cd "$HOME/github/biogeme"
squeue -u "$USER"
.venv/bin/python jed_runs/jed_examples.py status --verbose
```

There are two cleanup modes.

### Recoverable reset

Use this when the old results may be useful for diagnosis. It moves
generated outputs into an ignored timestamped backup below
`.jed_runs/resets`:

``` bash
PY=.venv/bin/python
"$PY" jed_runs/jed_examples.py reset --dry-run
"$PY" jed_runs/jed_examples.py reset --apply
```

This removes generated result files from the example directories while
keeping the backup. It is the preferred recovery option when
investigating a failed release run.

### Completely fresh reset

Use this when starting a release run from zero. It removes generated
root outputs, everything below `saved_results` and `saved_html`,
generated `.run` files, Slurm logs, caches, diagnostics, and the ignored
`.jed_runs` state. It preserves Python source, input data,
configuration, and runner code:

``` bash
PY=.venv/bin/python
"$PY" jed_runs/jed_fresh_start.py       # inspect the deletion list
"$PY" jed_runs/jed_fresh_start.py --apply
```

Run the dry run carefully. The fresh-start command deliberately has no
backup. Never substitute `git clean -fdx`: that can remove source files,
input data, virtual environments, and other recoverable state.

After either reset, verify that no old result files remain in the JED
example tree:

``` bash
"$PY" jed_runs/jed_fresh_start.py
```

The command should report no generated files (apart from empty archive
directories, which are retained so the harvester can recreate them).

## 3. Choose the JED workload

The runner discovers every `plot_*.py` recursively. It adds explicit
dependency edges before submitting jobs, so independent jobs can run in
parallel and consumers wait for their producers.

### Run everything

Use this for the normal release run. Generate the Slurm scripts first
and review them before submitting:

``` bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
"$PY" jed_runs/jed_examples.py launch \
   --dry-run --run-id "$RUN_ID"
```

Inspect the generated scripts below `.jed_runs/$RUN_ID/jobs`. Check the
requested CPU/memory/time profile, the working directory, the Python
interpreter, and any `afterok` dependency. Submit only after the dry run
is correct:

``` bash
"$PY" jed_runs/jed_examples.py launch \
   --run-id "$RUN_ID" --force
```

The `--force` is safe here because the dry run has already created the
run directory with this exact `RUN_ID`. Do not reuse an existing run ID
for a different set of jobs.

### Run only slow models

Use `--slow` when the fast, non-estimation examples will be run locally
or when only expensive estimation results need refreshing. It selects
every job whose JED resource profile is not `light` and includes its
declared dependencies:

``` bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
"$PY" jed_runs/jed_examples.py launch \
   --slow --dry-run --run-id "$RUN_ID"
"$PY" jed_runs/jed_examples.py launch \
   --slow --run-id "$RUN_ID" --force
```

The standard, Bayesian, Monte-Carlo, panel-simulation, hybrid, and
assisted specification workloads are considered slow. Resource profiles
are maintained in `jed_runs/jed_examples.toml`. If a newly added model
is computationally heavy but is classified as `light`, add a job
override such as:

``` toml
[jobs."family/plot_expensive_model.py"]
profile = "standard"
```

### Run one family or one consumer

`--only` is useful for retries and targeted checks. The runner
automatically adds all dependencies of each selected script:

``` bash
"$PY" jed_runs/jed_examples.py launch \
   --only swissmetro/plot_b05c_normal_mixture_simul.py \
   --dry-run --run-id "$RUN_ID"
```

Do not combine `--only` and `--slow`. If a consumer needs a result, list
the consumer; its producer is included automatically.

### Recover a forgotten `RUN_ID`

The launch command prints the run-state directory, for example
`.jed_runs/20260810-080103`. That final directory name is the `RUN_ID`. It is
also stored in `.jed_runs/<RUN_ID>/run.json`.

If the shell variable has been lost, omit `--run-id`: both `status` and the
error reporter select the newest run directory that contains `run.json` and
print its ID:

``` bash
"$PY" jed_runs/jed_examples.py status --verbose
"$PY" jed_runs/jed_error_report.py
```

For a script or log that needs the value as a shell variable, retrieve it
without relying on directory ordering:

``` bash
RUN_ID="$("$PY" - <<'PY'
from jed_runs.jed_examples import latest_run, load_config, state_root

run = latest_run(state_root(load_config()))
if run is None or not (run / 'run.json').is_file():
    raise SystemExit('No JED run state found')
print(run.name)
PY
)"
printf 'RUN_ID=%s\n' "$RUN_ID"
test -f ".jed_runs/$RUN_ID/run.json"
```

When several runs exist, list them and choose deliberately; “newest” means
newest by filesystem modification time and may be a dry-run or a retry:

``` bash
"$PY" - <<'PY'
from jed_runs.jed_examples import load_config, state_root

root = state_root(load_config())
for path in sorted(
    (p for p in root.iterdir() if p.is_dir() and p.name != 'resets'),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
):
    if (path / 'run.json').is_file():
        print(path.name)
PY
```

Then pass the selected ID explicitly to `status`, `jed_error_report.py`, and
any archival notes. Never use a newly created run ID to refer to a different
set of jobs.

### Test an incomplete JED run

Yes. A partial run is useful for validating a new Slurm profile, one example,
or a dependency chain before committing to an overnight release run. Give it
a distinct ID such as `smoke-<timestamp>` so it cannot be mistaken for the
release run.

Use `--only` for a small, explicit selection. Dependencies of each selected
consumer are added automatically:

``` bash
TEST_RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"
"$PY" jed_runs/jed_examples.py launch \
   --only indicators/plot_b09wtp.py \
   --dry-run --run-id "$TEST_RUN_ID"
# Inspect .jed_runs/$TEST_RUN_ID/jobs, then submit the same selection.
"$PY" jed_runs/jed_examples.py launch \
   --only indicators/plot_b09wtp.py \
   --run-id "$TEST_RUN_ID" --force
"$PY" jed_runs/jed_examples.py status --run-id "$TEST_RUN_ID" --verbose
```

Use `--slow` when the purpose is to exercise every non-`light` resource
profile but not the fast reports. This is still a substantial subset; use
`--only` when a genuinely small smoke test is wanted:

``` bash
TEST_RUN_ID="slow-smoke-$(date +%Y%m%d-%H%M%S)"
"$PY" jed_runs/jed_examples.py launch \
   --slow --dry-run --run-id "$TEST_RUN_ID"
"$PY" jed_runs/jed_examples.py launch \
   --slow --run-id "$TEST_RUN_ID" --force
```

Do not combine `--only` and `--slow`. A dry run creates a state directory but
submits no Slurm jobs; its entries are expected to remain `not scheduled`.
Jobs that were not selected are absent from that run's `run.json`, not failed.
Do not reset the tree between a producer and its dependent consumer, because
the dependency's archived result is part of the test.

A partial run is not release evidence. The full release import must use
`--profile full --strict` and must wait for every declared job and artifact.
For temporary inspection, import only the completed script's declared files
with `--script` into a disposable checkout or fixture tree:

``` bash
uv run --locked --group docs python tools/import_jed_results.py \
   --source "$JED_STAGE" --profile all \
   --script indicators/plot_b02estimation.py --strict --apply
```

If a selected job has no declared output contract, there is nothing for the
importer to copy; inspect its Slurm output instead. Keep partial fixtures out
of the release tree. After all smoke jobs finish, remove root-level generated
files with `jed_cleanup.py --apply`; remove only that test's metadata with
`rm -rf -- ".jed_runs/$TEST_RUN_ID"` (never while a job is running). Use
`jed_fresh_start.py --apply` only when the entire JED checkout is disposable.

## 4. Monitor and diagnose the server run

Status can be checked while jobs are running:

``` bash
"$PY" jed_runs/jed_examples.py status --run-id "$RUN_ID"
"$PY" jed_runs/jed_examples.py status --run-id "$RUN_ID" --verbose
squeue -u "$USER"
```

The status table uses compact keywords: `OK`, `ERROR`, `RUNNING`, `PENDING`,
and `NOT_SCHEDULED`. Errors are repeated in an `Errors requiring attention`
block with their diagnostic; use `--verbose` for additional Slurm details. A
successful job requires a zero Python exit status, a successful Slurm allocation, and
(for jobs marked `requires_artifacts`) a changed result/report artifact.
The stricter per-file and per-pattern check is performed by the local
runner and by the strict laptop importer; a Slurm `COMPLETED` state
alone is not sufficient for release acceptance.

Do not import results until every required job is
`finished without error`. For a failure, generate the diagnostic report:

``` bash
"$PY" jed_runs/jed_error_report.py --run-id "$RUN_ID"
less ".jed_runs/$RUN_ID/error-report.md"
```

The report begins with a short digest and then includes the job
metadata, Slurm accounting, generated batch script, standard output,
standard error, and completion records. Common causes include a missing
dependency artifact, a wrong output filename in the manifest, an
unavailable Python package, a memory/time limit, a multiprocessing entry
point without a main guard, or a model feature unsupported by the
selected backend.

After fixing the cause, use a new run ID for the retry. Do not reset the
whole tree if successful producer artifacts are still needed by the
retry.

## 5. Finalize the JED result tree

Once the run is complete, inspect all errors one last time:

``` bash
"$PY" jed_runs/jed_examples.py status --run-id "$RUN_ID" --verbose
"$PY" jed_runs/jed_error_report.py --run-id "$RUN_ID" --output -
```

The JED harvester copies newly created root-level YAML, NetCDF, HTML,
and Pareto files into `saved_results` or `saved_html` and intentionally
leaves the root copies in place until all jobs finish. Clean those root
copies only after the run is complete:

``` bash
"$PY" jed_runs/jed_cleanup.py
"$PY" jed_runs/jed_cleanup.py --apply
```

This cleanup preserves the archived fixtures. It does not remove the JED
run metadata, which is useful for the import audit. Keep the run ID and
the JED commit hash with the release notes.

## 6. Import the fixtures on the laptop

Use a clean laptop checkout at the same Git revision as JED. Before
importing, remove old laptop fixtures; the importer intentionally does
not delete stale files that are absent from the new manifest.

``` bash
cd "$HOME/MyFiles/github/biogeme"
git rev-parse HEAD
git status --short
uv sync --frozen
```

If the laptop checkout contains old generated results, inspect and
remove them with the same fresh-start cleaner:

``` bash
uv run --locked --group docs python jed_runs/jed_fresh_start.py
uv run --locked --group docs python jed_runs/jed_fresh_start.py --apply
```

This removes old `saved_results` and `saved_html` files but preserves
the example source and input data. If the old fixtures are tracked,
their removal will appear as intentional Git deletions; the new import
will add the current versions.

Stage the JED example tree in a temporary directory. Using a fresh
temporary directory avoids mixing an old transfer with the current run:

``` bash
JED_STAGE="$(mktemp -d /tmp/biogeme-jed-results.XXXXXX)"
rsync -a user@jed.epfl.ch:/home/bierlair/github/biogeme/docs/source/examples/ \
   "$JED_STAGE/"
```

The importer accepts either this staged `docs/source/examples` directory
or a complete mounted JED checkout. First perform the strict dry run:

``` bash
uv run --locked --group docs python tools/import_jed_results.py \
   --source "$JED_STAGE" --profile full --strict
```

The command considers only `expected_outputs` and
`expected_output_globs` in `jed_runs/jed_examples.toml`. YAML, NetCDF,
and Pareto files are copied to `saved_results`; HTML files to
`saved_html`; declared text reports remain at the example root. It never
copies arbitrary logs, source files, caches, or undeclared output files.

The strict dry run must list every declared artifact as available. If it
reports a missing file, stop and inspect the JED status/error report. Do
not use `--apply` to hide a missing result. When the dry run is
complete:

``` bash
uv run --locked --group docs python tools/import_jed_results.py \
   --source "$JED_STAGE" --profile full --strict --apply
```

The importer backs up overwritten laptop fixtures below
`.docs_runs/imports/<timestamp>/backup` and writes a checksum report to
`.docs_runs/imports/<timestamp>/report.json`. Keep that report until the
release is complete.

Four jobs are currently intentionally outside the full fixture contract:
`assisted/plot_b09post_processing.py`,
`swissmetro/plot_b01e_logit_all_algos.py`, and the Swissmetro `b21c` and
`b22c` Pareto post-processing scripts. They still must finish
successfully on JED, but their in-memory or CSV-only reports are not
imported by this command.

## 7. Run and verify Sphinx Gallery

The documentation dependencies are locked. Run the inexpensive hermetic
check first; it executes the fast profile in isolated `.docs_runs`
workspaces:

``` bash
make -C docs clean
make -C docs examples-fast
make -C docs html-fast
make -C docs check-html
```

`html-fast` uses `BIOGEME_DOCS_GALLERY_PROFILE=none` and therefore
checks the Sphinx site without executing gallery examples. This
separates a quick example smoke test from a documentation rendering
failure.

Then run the release build. `html` uses the full `plot_*.py` gallery
pattern and has `abort_on_example_error=True`; any gallery exception
makes the build fail:

``` bash
make -C docs html PROFILE=full
make -C docs check-html
```

Review `docs/warnings.log`. A non-zero build is a blocker. A zero build
with warnings still requires review, especially for missing references,
unexecuted examples, tracebacks, and warnings introduced by the release.
For an additional check, run:

``` bash
make -C docs linkcheck
make -C docs doctest
```

The full gallery may leave root-level generated files in the example
directories. Inspect and remove only those files after the build:

``` bash
uv run --locked --group docs python jed_runs/jed_cleanup.py
uv run --locked --group docs python jed_runs/jed_cleanup.py --apply
```

Do not run `jed_fresh_start.py` at this point: it would delete the
imported fixtures in `saved_results` and `saved_html`.

## 8. Review and commit

Review the result tree and ensure that only intended fixture changes are
present:

``` bash
git status --short
git diff --name-status -- docs/source/examples
git diff --stat -- docs/source/examples
git diff --check
```

The ignored `.jed_runs`, `.docs_runs`, Slurm logs, caches, and temporary
transfer directory must not be staged. If the repository distributes the
archived fixtures, the helper safely stages only `saved_results` and
`saved_html`:

``` bash
uv run --locked --group docs python jed_runs/jed_commit_results.py --dry-run
uv run --locked --group docs python jed_runs/jed_commit_results.py \
   --message "Refresh example results for <version>"
```

Alternatively, review and stage the files manually. Commit the source,
manifest, documentation, and fixture changes together only after the
full gallery has passed. Push a release branch and run the GitHub
documentation workflow before merging or tagging.

## Adding a new example

Every new gallery script should be reproducible from the checkout and
should not depend on a developer's home directory or on a previous local
run.

1.  Create `plot_<name>.py` below the appropriate
    `docs/source/examples/<family>` directory. Keep input data next to
    the example or use package data; never commit generated results as
    input data.

2.  Make the script safe for a headless process. Use the existing
    data/model conventions, avoid interactive-only assumptions, and put
    multiprocessing or PyMC execution behind
    `if __name__ == '__main__':`.

3.  Decide whether it is fast, full, or JED-only. Add a
    `[docs.examples."family/plot_<name>.py"]` entry when it needs a
    profile, dependency mode, gallery override, or output contract. The
    fast profile is opt-in; do not add a slow estimator to it.

4.  For persistent outputs, declare the names written at the workspace
    root:

    ``` toml
    [docs.examples."family/plot_<name>.py"]
    profile = "full"
    expected_outputs = ["model.yaml", "model.html"]
    ```

    Use `expected_output_globs` for runtime-generated names, for example
    `["model_*.yaml"]`. Do not put `saved_results/` or `saved_html/` in
    these names; those are archive destinations handled by the importer.
    Text reports may remain at the example root. If the example
    genuinely produces no persistent artifact, set
    `requires_artifacts = false` in the JED job configuration and
    document why.

5.  If the example consumes another example's result, add a JED
    dependency and required input:

    ``` toml
    [jobs."family/plot_consumer.py"]
    depends_on = ["family/plot_estimator.py"]
    required_inputs = ["saved_results/model.yaml"]
    ```

    The dependency is also followed by the local hermetic runner. The
    producer must finish before the consumer starts.

6.  Select an appropriate JED resource profile. Use `standard` for
    normal estimation, `bayesian` for Bayesian jobs, `montecarlo` for
    high-draw jobs, `panel_simulation` for high-memory panel
    simulations, `hybrid` for the maintained high-memory hybrid job, and
    `light` only for genuinely short non-estimation reports.

7.  Validate locally from a clean workspace:

    ``` bash
    uv run --locked --group docs python tools/docs_examples.py \
       plan --script family/plot_<name>.py
    uv run --locked --group docs python tools/docs_examples.py \
       run --script family/plot_<name>.py --keep-workspace
    uv run --locked --group docs python tools/docs_examples.py status --verbose
    ```

    Inspect the job workspace and confirm that every declared output
    exists. Add or update a test under `tests/documentation` for
    dependencies and output contracts.

8.  Validate the JED batch script without submitting it:

    ``` bash
    "$PY" jed_runs/jed_examples.py launch --only \
       family/plot_<name>.py --dry-run --run-id "$RUN_ID"
    ```

9.  Run the example on JED, import its declared fixture, and include it
    in a full gallery build before considering it release-ready.

## Definition of release-ready

The examples and documentation are ready when:

- JED and laptop use the same Git revision and locked environment.
- All selected JED jobs are finished without errors, including
  dependencies.
- The strict importer finds every declared fixture and its checksum
  report is retained.
- The fast hermetic profile passes.
- `make -C docs html PROFILE=full` and `make -C docs check-html` pass.
- Warnings and links have been reviewed.
- Root-level generated files have been cleaned without deleting saved
  fixtures.
- The Git diff contains only intentional source, documentation, and
  result changes.

For operational details and command reference, see
[`docs/source/examples/JED_RUNS.md`](docs/source/examples/JED_RUNS.md). The
design of isolated local execution is described in
[`docs/source/examples_workflow.rst`](docs/source/examples_workflow.rst).
