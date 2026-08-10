# Monte Carlo Draw-Stability Diagnostic: b27_monte_carlo

## Execution status

**Execution status:** Completed.

All planned evaluations completed.

## Diagnostic conclusion

**Diagnostic conclusion:** Unstable.

## Recommendation

**Recommendation:** More draws recommended.

More draws are recommended because discrepancies remain above the configured tolerances at a higher draw level.

## Motivation for the recommendation

- At least one discrepancy remains above tolerance at the highest completed draw level.
- The objective discrepancy remains above tolerance at the highest completed draw level.
- The gradient discrepancy remains above tolerance at the highest completed draw level.

## Purpose and limitations

This practical diagnostic assesses sensitivity to the number and design of
Monte Carlo draws. It is not a formal integration-error confidence interval.
No re-estimation was performed. The estimated parameter vector remained fixed.

- No draw-design limitation was identified.

## Methodology

The original estimation result is the reference. For each planned draw level,
the model criterion and its gradient were evaluated at the fixed estimated
parameters using a fresh diagnostic draw design. Pseudo-random, antithetic,
and MLHS designs were regenerated independently. Native Halton designs used a
diagnostic-only randomized modulo-one shift; ordinary estimation behavior was
not changed. No Hessian was requested. The first Ctrl-C requests a graceful
stop after the active evaluation has been checkpointed; a second Ctrl-C may
terminate immediately.

## Explanation of calculated quantities

The objective is the model criterion evaluated at the fixed estimated
parameters. The gradient is the derivative of that criterion at those same
parameters. Objective discrepancies measure sensitivity of the criterion to
the draw design. Gradient discrepancies are especially important because they
indicate whether the simulated optimum may move. The infinity norm is the
largest absolute component of the gradient difference; the Euclidean norm
summarizes its overall magnitude.

## Configuration

- Original number of draws: 2000
- Draw factors: [0.5, 1.0, 2.0]
- Replications per level: 1
- Time budget: 300.0 seconds
- Maximum draws: 4000
- Runtime safety factor: 1.5
- Objective tolerance: 0.001
- Gradient tolerance: 1e-05
- Minimum conclusive level factor: 2.0

## Results

| Draws | Replication | Objective | Absolute objective difference | Relative objective difference | Gradient $L_\infty$ difference | Gradient $L_2$ difference | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1 | -5215.201649 | 0.4163 | 7.982e-05 | 0.8254 | 1.133 | 1.817 |
| 2000 | 1 | -5215.462062 | 0.6767 | 0.0001298 | 0.9981 | 1.346 | 3.305 |
| 4000 | 1 | -5215.097229 | 0.3118 | 5.98e-05 | 0.8474 | 1.024 | 6.490 |

## Interruption or time-budget details

- Completed evaluations: 3
- Uncompleted evaluations: 0
- Skipped because of the time budget: 0

## Completed and uncompleted evaluations

Completed: 3 of 3.

Planned but not started:

- None.

Skipped because of the time budget:

- None.

## Seeds and randomization identifiers

- 1000 draws, replication 1: seed 1584127666 (diagnostic-seed-1584127666)
- 2000 draws, replication 1: seed 4260784625 (diagnostic-seed-4260784625)
- 4000 draws, replication 1: seed 20440104 (diagnostic-seed-20440104)

## Suggested next action

More draws are recommended because discrepancies remain above the configured tolerances at a higher draw level.
