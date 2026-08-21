# Apollo benchmark

This directory contains Apollo/R versions of the two simulated Swissmetro
models used in the Biogeme release benchmark:

- `b05a_normal_mixture.R`: 2,000 inter-individual normal draws;
- `b12_panel.R`: 5,000 inter-individual normal draws and panel products.

The data filtering, variable scaling, starting values, and availability rules
are copied from the Biogeme benchmark. Apollo generates its own pseudo-random
Monte Carlo draws; matching the draw realization is deliberately not required
for this timing experiment. The b05a run requests an analytical Hessian; the
b12 run uses `hessianRoutine="none"`, matching the Biogeme benchmark policy.

## Prepare Apollo on JED

Use the R installation selected for the benchmark and install Apollo in the
user R library if it is not already available:

```bash
Rscript -e 'install.packages(c("apollo", "jsonlite"), repos="https://cloud.r-project.org")'
Rscript -e 'library(apollo); library(jsonlite); cat(as.character(packageVersion("apollo")), "\n")'
```

Record the Apollo and R versions in the job log. If the selected Apollo
release still imports the archived `RSGHB` package, install the matching
archived RSGHB binary/source as well. Prefer the current Apollo release on
JED, and do not mix R libraries between runs.

## Inspect and submit

From the repository root:

```bash
sbatch --test-only jed_runs/benchmarks/release_comparison/apollo/apollo_benchmark.run
sbatch jed_runs/benchmarks/release_comparison/apollo/apollo_benchmark.run
```

The job runs the two estimators sequentially with one computational thread
inside the same 8-CPU/56-GB JED allocation used for the Biogeme benchmark.
The extra allocated CPUs satisfy JED's memory-per-CPU limit; they are not used
by Apollo. Outputs are written outside the Git checkout by default:

```text
<results-root>/raw/b05a_normal_mixture.json
<results-root>/raw/b12_panel.json
<results-root>/logs/b05a_normal_mixture.log
<results-root>/logs/b12_panel.log
<results-root>/report.md
<results-root>/timings.csv
```

To use a different result directory or R executable:

```bash
export APOLLO_BENCHMARK_RESULTS_ROOT=/home/bierlair/apollo-benchmark-results/manual-run
export APOLLO_RSCRIPT=/path/to/Rscript
sbatch jed_runs/benchmarks/release_comparison/apollo/apollo_benchmark.run
```

The generated report compares Apollo with the Biogeme 3.3.4 timings from
`docs/biogeme-benchmark-results/66162235/timings.csv`. It does not rerun any
Biogeme estimator.
