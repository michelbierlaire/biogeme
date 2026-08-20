"""Benchmark fast and safe likelihood expressions in fresh processes.

Examples::

    uv run python tests/jax/benchmark_fast_stable_expressions.py nested fast gradient
    uv run python tests/jax/benchmark_fast_stable_expressions.py sparse-cnl safe hessian
"""

from __future__ import annotations

import argparse
import json
import statistics
from time import perf_counter

import numpy as np
import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.expressions.log_cross_nested import LogCrossNested
from biogeme.expressions.log_nested import LogNested
from biogeme.expressions.sparse_log_cross_nested import SparseLogCrossNested
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.nests import (
    NestsForCrossNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    OneNestForNestedLogit,
)
from biogeme.profiling.timing import timed_call
from biogeme.second_derivatives import SecondDerivativesMode


def build_model(
    family: str,
    implementation: str,
    *,
    observations: int,
    alternatives: int,
    nests_count: int,
    explicit_mu: bool,
) -> tuple[ModelElements, dict[str, float]]:
    generator = np.random.default_rng(59421)
    dataframe = pd.DataFrame(
        {
            f'x{alternative}': generator.normal(size=observations)
            for alternative in range(alternatives)
        }
    )
    dataframe['choice'] = np.arange(observations) % alternatives + 1
    database = Database('fast_stable_benchmark', dataframe)
    beta = Beta('beta', -0.7, None, None, 0)
    nest_mu = Beta('nest_mu', 1.4, 1.0, 4.0, 0)
    global_mu = Beta('global_mu', 1.1, 0.5, 3.0, 0)
    utilities = {
        alternative + 1: beta * Variable(f'x{alternative}')
        for alternative in range(alternatives)
    }

    if family == 'nested':
        allocations: list[list[int]] = [list() for _ in range(nests_count)]
        for alternative in range(alternatives):
            allocations[alternative % nests_count].append(alternative + 1)
        nests = NestsForNestedLogit(
            choice_set=list(utilities),
            tuple_of_nests=tuple(
                OneNestForNestedLogit(nest_mu, allocation, f'nest_{index}')
                for index, allocation in enumerate(allocations)
            ),
        )
        expression_class = {
            'fast': LogNested,
            'safe': LogNested,
        }[implementation]
    else:
        allocations_cnl: list[dict[int, float]] = [
            dict() for _ in range(nests_count)
        ]
        if family == 'dense-cnl':
            for nest in range(nests_count):
                allocations_cnl[nest] = {
                    alternative + 1: 1.0 / nests_count
                    for alternative in range(alternatives)
                }
        else:
            for alternative in range(alternatives):
                first = alternative % nests_count
                second = (alternative + 3) % nests_count
                allocations_cnl[first][alternative + 1] = 0.4
                allocations_cnl[second][alternative + 1] = 0.6
        nests = NestsForCrossNestedLogit(
            choice_set=list(utilities),
            tuple_of_nests=tuple(
                OneNestForCrossNestedLogit(
                    nest_mu, allocation, f'nest_{index}'
                )
                for index, allocation in enumerate(allocations_cnl)
            ),
        )
        if family == 'dense-cnl':
            expression_class = {
                'fast': LogCrossNested,
                'safe': LogCrossNested,
            }[implementation]
        else:
            expression_class = {
                'fast': SparseLogCrossNested,
                'safe': SparseLogCrossNested,
            }[implementation]

    expression = expression_class(
        util=utilities,
        av=None,
        nests=nests,
        choice=Variable('choice'),
        mu=global_mu if explicit_mu else None,
    )
    return (
        ModelElements.from_expression_and_weight(
            log_like=expression,
            weight=None,
            adapter=RegularAdapter(database=database),
            use_jit=True,
        ),
        {
            'beta': -0.7,
            'nest_mu': 1.4,
            **({'global_mu': 1.1} if explicit_mu else {}),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'family', choices=('nested', 'dense-cnl', 'sparse-cnl')
    )
    parser.add_argument(
        'implementation', choices=('fast', 'safe')
    )
    parser.add_argument(
        'mode', choices=('value', 'gradient', 'hessian', 'bhhh')
    )
    parser.add_argument('--observations', type=int, default=5_000)
    parser.add_argument('--alternatives', type=int, default=30)
    parser.add_argument('--nests', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--global-mu', action='store_true')
    arguments = parser.parse_args()

    elements, betas = build_model(
        arguments.family,
        arguments.implementation,
        observations=arguments.observations,
        alternatives=arguments.alternatives,
        nests_count=arguments.nests,
        explicit_mu=arguments.global_mu,
    )
    build_start = perf_counter()
    evaluator = CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=arguments.implementation == 'safe',
    )
    evaluator_build = perf_counter() - build_start
    requested = {
        'value': (False, False, False),
        'gradient': (True, False, False),
        'hessian': (True, True, False),
        'bhhh': (True, False, True),
    }[arguments.mode]

    def evaluate():
        return evaluator.evaluate(betas, *requested)

    result, first_call = timed_call(evaluate)
    warm_times = [
        timed_call(evaluate)[1] for _ in range(arguments.repeats)
    ]
    print(
        json.dumps(
            {
                'family': arguments.family,
                'implementation': arguments.implementation,
                'mode': arguments.mode,
                'observations': arguments.observations,
                'alternatives': arguments.alternatives,
                'nests': arguments.nests,
                'global_mu': arguments.global_mu,
                'evaluator_build_seconds': evaluator_build,
                'first_call_seconds': first_call,
                'warm_median_seconds': statistics.median(warm_times),
                'warm_mean_seconds': statistics.mean(warm_times),
                'warm_min_seconds': min(warm_times),
                'warm_max_seconds': max(warm_times),
                'finite': bool(
                    np.isfinite(result.function)
                    and (
                        result.gradient is None
                        or np.isfinite(result.gradient).all()
                    )
                    and (
                        result.hessian is None
                        or np.isfinite(result.hessian).all()
                    )
                    and (
                        result.bhhh is None
                        or np.isfinite(result.bhhh).all()
                    )
                ),
                'repeats': arguments.repeats,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
