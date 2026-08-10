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
