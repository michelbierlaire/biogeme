# Running the examples on JED

The example runner discovers every plot_*.py below this directory. It
generates one Slurm job per script and submits dependent jobs with
afterok dependencies, so independent examples run concurrently while an
example that consumes another example's results waits for its producer.

The current resource defaults are based on the existing JED files:

- standard and Bayesian/Monte-Carlo estimation: 36 CPU cores, 70 hours;
- maintained H04 hybrid estimation: 8 cores, 7 GB per core, 2 days;
- non-estimation reports: 4 cores, 4 hours.

Resource profiles and dependency exceptions are in
tools/jed_examples.toml. The generated .run files, Slurm logs, job IDs,
diagnostics, and reset backups are stored below .jed_runs/, which is ignored
by Git.

## Complete workflow

Run these commands from the repository checkout on JED:

~~~bash
cd "$HOME/github/biogeme"

# 1. Inspect what will be reset.
python tools/jed_examples.py reset --dry-run

# 2. Move old generated results aside. This preserves a recoverable backup.
python tools/jed_examples.py reset --apply

# 3. Generate all jobs and submit them. Use --dry-run first if desired.
RUN_ID="$(date +%Y%m%d-%H%M%S)"
python tools/jed_examples.py launch --dry-run --run-id "$RUN_ID"
python tools/jed_examples.py launch --run-id "$RUN_ID" --force

# 4. Inspect status at any time.
python tools/jed_examples.py status
python tools/jed_examples.py status --verbose
~~~

## Investigate failed jobs

When the status report contains `finished with errors`, run the diagnostic
reporter. It examines the newest run by default; pass `--run-id` to select a
specific run. It does not rerun an example or change its outputs. The report
is written below the ignored `.jed_runs/` directory, with a short digest first
and the runner metadata, Slurm accounting, generated batch script, standard
output, standard error, and completion records afterward.

~~~bash
cd "$HOME/github/biogeme"

python tools/jed_error_report.py --run-id "$RUN_ID"
less ".jed_runs/$RUN_ID/error-report.md"
~~~

If the launch shell no longer has `RUN_ID`, omit the option to inspect the
newest run. To send the complete Markdown report directly to the terminal:

~~~bash
python tools/jed_error_report.py --run-id "$RUN_ID" --output -
~~~

## Commit a completed result set

Commit results only after every job in the run is reported as
`finished without error`. Do not commit while a job is `running`,
`scheduled and pending`, `not scheduled`, or `finished with errors`.

Run the status command once more and keep the diagnostic output with the
run state (the `.jed_runs/` directory is ignored by Git):

~~~bash
cd "$HOME/github/biogeme"

# Use --run-id "$RUN_ID" if the launch shell still has RUN_ID set.
python tools/jed_examples.py status --verbose \
  | tee ".jed_runs/${RUN_ID:-latest}-status.txt"
~~~

The reset operation moved the previous generated files into
`.jed_runs/resets/<timestamp>/`. The files that no longer exist in the
example directories must therefore appear as deletions in the Git diff;
this is how obsolete results are removed from the repository. Never use
`git clean -fdx` here: it can delete source files, input data, and the
recoverable reset backups.

Review the complete change before staging it:

~~~bash
# Deletions are intentional only for old generated results.
git status --short -- docs/source/examples
git diff --name-status -- docs/source/examples
git diff --stat -- docs/source/examples
~~~

From a clean checkout, stage the current example tree with `-A`. The `-A`
is important because it stages both newly generated files and deletions of
old tracked results. It does not add the ignored `.jed_runs/`, Slurm logs,
or generated `.run` files.

~~~bash
git add -A -- docs/source/examples

# Inspect exactly what will be committed.
git diff --cached --name-status -- docs/source/examples
git diff --cached --stat -- docs/source/examples
git diff --cached --check
~~~

Do not use `git add -f` for an entire directory. Some result directories
(for example, Bayesian or legacy latent results) are intentionally ignored
by the repository. If the new release is meant to distribute those results,
force-add only the reviewed files, for example:

~~~bash
git add -f -- docs/source/examples/bayesian_swissmetro/saved_results/*.yaml
~~~

Otherwise leave them ignored and record that decision in the release notes.
Before committing, confirm that no source code, input data, `.jed_runs`
state, Slurm output, cache, or checkpoint file is staged.

Create a dedicated branch from JED (unless the project explicitly permits
direct pushes to the release branch), then commit and push the reviewed
result set:

~~~bash
git switch -c refresh-example-results-$(date +%Y%m%d)
git commit -m "Refresh example results from JED"
git push -u origin HEAD
~~~

Open or update the corresponding pull request and check the documentation
build before merging. The commit should contain only the latest result
files, with obsolete tracked results represented by intentional deletions;
the Slurm job history and reset backups remain on JED under `.jed_runs/`.

BIOGEME_JED_REPOSITORY and BIOGEME_JED_PYTHON can override the repository
and interpreter paths. By default the runner uses the checkout's .venv
created by uv sync --frozen, and falls back to
$HOME/venvs/biogeme/bin/python if that interpreter is unavailable.

The reset command removes generated files from the example directories by
moving them into .jed_runs/resets/<timestamp>/. It preserves Python source,
input data, configuration files, and .run files. On successful completion,
new root-level YAML/NetCDF/HTML/Pareto results are harvested into the
saved_results or saved_html directory expected by dependent examples.

The status report uses both Slurm accounting and the job completion record.
For jobs that declare an expected artifact, success requires Slurm exit code
zero and a result/report artifact that was created or modified during that
run. In-memory examples intentionally have no artifact requirement. Missing
dependencies, missing outputs, interpreter failures, Python exceptions, Slurm
failures, and missing completion markers are reported as diagnostics.
