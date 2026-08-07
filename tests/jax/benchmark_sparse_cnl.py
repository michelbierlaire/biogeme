"""Benchmark dense and experimental sparse CNL expressions in fresh processes.

Examples::

    uv run python tests/jax/benchmark_sparse_cnl.py dense sparse gradient
    uv run python tests/jax/benchmark_sparse_cnl.py sparse sparse gradient
    uv run python tests/jax/benchmark_sparse_cnl.py dense dense hessian
    uv run python tests/jax/benchmark_sparse_cnl.py sparse dense hessian
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
from biogeme.expressions.sparse_log_cross_nested import SparseLogCrossNested
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.profiling.timing import timed_call
from biogeme.second_derivatives import SecondDerivativesMode


def build_model(
    implementation: str,
    membership_pattern: str,
    *,
    observations: int,
    alternatives: int,
    nests_count: int,
) -> tuple[ModelElements, dict[str, float], int]:
    generator = np.random.default_rng(73491)
    dataframe = pd.DataFrame(
        {
            f'x{alternative}': generator.normal(size=observations)
            for alternative in range(alternatives)
        }
    )
    dataframe['choice'] = np.arange(observations) % alternatives + 1
    database = Database('synthetic_sparse_cnl', dataframe)

    beta = Beta('beta', -0.7, None, None, 0)
    mu = Beta('mu', 1.4, 1.0, 4.0, 0)
    utilities = {
        alternative + 1: beta * Variable(f'x{alternative}')
        for alternative in range(alternatives)
    }

    allocations: list[dict[int, float]] = [dict() for _ in range(nests_count)]
    if membership_pattern == 'sparse':
        for alternative in range(alternatives):
            first_nest = alternative % nests_count
            second_nest = (alternative + 3) % nests_count
            allocations[first_nest][alternative + 1] = 0.4
            allocations[second_nest][alternative + 1] = 0.6
        active_memberships = 2 * alternatives
    else:
        for nest in range(nests_count):
            allocations[nest] = {
                alternative + 1: 1.0 / nests_count
                for alternative in range(alternatives)
            }
        active_memberships = nests_count * alternatives

    nests = NestsForCrossNestedLogit(
        choice_set=list(utilities),
        tuple_of_nests=tuple(
            OneNestForCrossNestedLogit(
                nest_param=mu,
                dict_of_alpha=allocation,
                name=f'nest_{nest}',
            )
            for nest, allocation in enumerate(allocations)
        ),
    )
    expression_class = {
        'dense': LogCrossNested,
        'sparse': SparseLogCrossNested,
        'log-domain': LogCrossNested,
    }[implementation]
    expression = expression_class(
        util=utilities,
        av=None,
        nests=nests,
        choice=Variable('choice'),
    )
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=True,
    )
    return elements, {'beta': -0.7, 'mu': 1.4}, active_memberships


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'implementation', choices=('dense', 'sparse', 'log-domain')
    )
    parser.add_argument('membership_pattern', choices=('sparse', 'dense'))
    parser.add_argument('mode', choices=('value', 'gradient', 'hessian', 'bhhh'))
    parser.add_argument('--observations', type=int, default=5000)
    parser.add_argument('--alternatives', type=int, default=30)
    parser.add_argument('--nests', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=50)
    args = parser.parse_args()

    elements, betas, active_memberships = build_model(
        args.implementation,
        args.membership_pattern,
        observations=args.observations,
        alternatives=args.alternatives,
        nests_count=args.nests,
    )
    build_start = perf_counter()
    evaluator = CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=args.implementation == 'log-domain',
    )
    evaluator_build = perf_counter() - build_start
    requested = {
        'value': (False, False, False),
        'gradient': (True, False, False),
        'hessian': (True, True, False),
        'bhhh': (True, False, True),
    }[args.mode]

    def evaluate():
        return evaluator.evaluate(betas, *requested)

    _, first_call = timed_call(evaluate)
    warm_times = [timed_call(evaluate)[1] for _ in range(args.repeats)]
    print(
        json.dumps(
            {
                'implementation': args.implementation,
                'membership_pattern': args.membership_pattern,
                'mode': args.mode,
                'observations': args.observations,
                'alternatives': args.alternatives,
                'nests': args.nests,
                'active_memberships': active_memberships,
                'dense_memberships': args.alternatives * args.nests,
                'evaluator_build_seconds': evaluator_build,
                'first_call_seconds': first_call,
                'warm_median_seconds': statistics.median(warm_times),
                'warm_mean_seconds': statistics.mean(warm_times),
                'warm_min_seconds': min(warm_times),
                'warm_max_seconds': max(warm_times),
                'repeats': args.repeats,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
