# Biogeme release-comparison benchmark

This benchmark estimates the three Swissmetro models from the Biogeme 3.3.4
distribution with Biogeme 3.2.14, 3.3.3, and 3.3.4. It does **not** execute the
historical examples from the older releases. The model definitions are kept
in `benchmark_support.py` and are copied from:

```text
docs/source/examples/swissmetro/plot_b05a_normal_mixture.py
docs/source/examples/swissmetro/plot_b11a_cnl.py
docs/source/examples/swissmetro/plot_b12_panel.py
```

The nine generated entry points are pure estimations. They do not call
`estimate_or_load`, do not read YAML/pickle/iteration files, and disable HTML
and result-file generation. Each writes one JSON record containing the timing,
solution, convergence information, executable path, imported module path, and
the effective estimation configuration.

## 1. Prepare the release environments on JED

Use a separate worktree and `uv` environment for each tag. The preparation is
resumable and never removes an existing target:

```bash
cd /home/bierlair/github/biogeme
PY=.venv/bin/python

$PY jed_runs/benchmarks/release_comparison/prepare_environments.py \
  --worktree-root /home/bierlair/github/biogeme-benchmark-worktrees \
  --environment-root /home/bierlair/venvs/biogeme-benchmark

$PY jed_runs/benchmarks/release_comparison/prepare_environments.py \
  --worktree-root /home/bierlair/github/biogeme-benchmark-worktrees \
  --environment-root /home/bierlair/venvs/biogeme-benchmark \
  --apply
```

The command uses Python 3.12 by default. If the 3.2.14 dependencies cannot be
installed under the common Python version, stop and resolve that explicitly;
do not silently mix Python versions in the timing comparison.

Verify every environment before submitting the benchmark:

```bash
for version in 3_2_14 3_3_3 3_3_4; do
  /home/bierlair/venvs/biogeme-benchmark/biogeme_${version}/bin/python \
    -c "import sys, importlib.metadata as m, biogeme; print(sys.executable); print(m.version('biogeme')); print(biogeme.__file__)"
done
```

The reported package versions must be exactly `3.2.14`, `3.3.3`, and `3.3.4`.

## 2. Inspect the Slurm plan

The supplied batch file uses one 8-CPU, 56 GB allocation and executes all nine
estimations sequentially. The memory request respects JED's 7 GB per-CPU
limit. It writes results outside the Git checkout by default:

```bash
cd /home/bierlair/github/biogeme
sbatch --test-only jed_runs/benchmarks/release_comparison/biogeme_release_benchmark.run
```

If the environment or output roots differ, set them explicitly:

```bash
export BIOGEME_BENCHMARK_ENV_ROOT=/home/bierlair/venvs/biogeme-benchmark
export BIOGEME_BENCHMARK_RESULTS_ROOT=/home/bierlair/biogeme-benchmark-results/manual-run
```

## 3. Submit the benchmark

```bash
sbatch jed_runs/benchmarks/release_comparison/biogeme_release_benchmark.run
```

The job creates:

```text
<results-root>/raw/<release>/<model>/repeat-01.json
<results-root>/raw/<release>/<model>/repeat-01.stdout.log
<results-root>/raw/<release>/<model>/repeat-01.stderr.log
<results-root>/report.md
<results-root>/timings.csv
```

The runner continues after a failed case so that one problematic release does
not hide the other results. The Slurm job nevertheless exits nonzero if any
case fails. Inspect the corresponding `stderr.log` and `error.json`, repair
the environment or adapter, and submit a new results root.

For a local dry run of the exact nine commands:

```bash
$PY jed_runs/benchmarks/release_comparison/run_benchmark.py \
  --env-root /home/bierlair/venvs/biogeme-benchmark \
  --output-root /tmp/biogeme-release-benchmark/raw \
  --dry-run
```

## 4. Interpret the report

The timing table reports total wall-clock time around `estimate()`. The JSON
also records Biogeme's internal optimization time and evaluation counts. The
correctness section uses Biogeme 3.3.4 as the reference and reports differences
in the final log likelihood and every estimated parameter.

The CNL model is deterministic and should agree to numerical precision. The
mixture and panel models use Monte Carlo integration. Equal seeds and draw
specifications are supplied, but different release generations may produce
different draw streams; their solution differences must therefore be judged
against simulation noise rather than bitwise equality.

The first run is intentionally one repetition per case. If the timings are
noisy, repeat the matrix without changing the environments:

```bash
$PY jed_runs/benchmarks/release_comparison/run_benchmark.py \
  --env-root /home/bierlair/venvs/biogeme-benchmark \
  --output-root /home/bierlair/biogeme-benchmark-results/repeated/raw \
  --repetitions 3

$PY jed_runs/benchmarks/release_comparison/compare_results.py \
  --results-root /home/bierlair/biogeme-benchmark-results/repeated/raw \
  --markdown /home/bierlair/biogeme-benchmark-results/repeated/report.md \
  --csv /home/bierlair/biogeme-benchmark-results/repeated/timings.csv \
  --strict
```

Do not compare runs made with different CPU allocations, thread settings,
draw counts, or parameter files. Keep the raw JSON and Slurm logs with the
report so the result remains reproducible.
