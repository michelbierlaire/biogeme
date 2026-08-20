# Biogeme release benchmark

The models are the Biogeme 3.3.4 Swissmetro specifications. Only the compatibility adapter changes across releases.

Results directory: `/home/bierlair/biogeme-benchmark-results/66162235/raw`

## Median wall-clock estimation time (seconds)

| Model | 3.2.14 | 3.3.3 | 3.3.4 |
|---|---:|---:|---:|
| `b05a_normal_mixture` | 1705.011 | 47.970 | 38.891 |
| `b11a_cnl` | 38.588 | 155.023 | 2.637 |
| `b12_panel` | 8180.078 | 109.022 | 101.023 |

## Median warm time for one likelihood/gradient evaluation (seconds)

The first call is intentionally excluded from these measurements because it can include JAX compilation. The total wall-clock time above includes compilation.

| Model | 3.2.14 | 3.3.3 | 3.3.4 |
|---|---:|---:|---:|
| `b05a_normal_mixture` | 37.167 | 0.548 | 0.406 |
| `b11a_cnl` | 0.728 | 0.016 | 0.003 |
| `b12_panel` | 161.057 | 1.429 | 1.155 |

## Median warm time for one likelihood/gradient/Hessian evaluation (seconds)

The first call is intentionally excluded from these measurements because it can include JAX compilation. The total wall-clock time above includes compilation.

| Model | 3.2.14 | 3.3.3 | 3.3.4 |
|---|---:|---:|---:|
| `b05a_normal_mixture` | 53.748 | 3.678 | 2.758 |
| `b11a_cnl` | 0.997 | 0.124 | 0.021 |
| `b12_panel` | 293.513 | n/a | n/a |

## Correctness comparisons

The reference is Biogeme 3.3.4. Differences are reported rather than silently treated as failures. The two Monte Carlo models may differ because equivalent seeds do not guarantee identical draw streams across implementation generations.

| Model | Candidate | Δ log-likelihood | Max abs parameter difference | Max relative parameter difference |
|---|---|---:|---:|---:|
| `b05a_normal_mixture` | 3.2.14 | 1.211 | 0.007 | 0.004 |
| `b05a_normal_mixture` | 3.3.3 | 0.000 | 0.000 | 0.000 |
| `b11a_cnl` | 3.2.14 | 0.000 | 0.002 | 0.001 |
| `b11a_cnl` | 3.3.3 | 0.000 | 0.001 | 0.001 |
| `b12_panel` | 3.2.14 | 0.088 | 0.933 | 0.933 |
| `b12_panel` | 3.3.3 | 0.000 | 0.000 | 0.000 |

## Run metadata

Each JSON record contains the executable path, imported Biogeme module path, package version, seed, draw count, configuration, convergence flag, Biogeme-reported optimization diagnostics, and warm evaluation timings. Hessian timing is `n/a` when second derivatives were disabled for that model.

