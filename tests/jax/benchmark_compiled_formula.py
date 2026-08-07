"""Benchmark the production JAX evaluator on Swissmetro CNL.

Run each mode in a fresh process so compilation and warm execution remain
separate, for example::

    uv run python tests/jax/benchmark_compiled_formula.py gradient
    uv run python tests/jax/benchmark_compiled_formula.py hessian
"""

from __future__ import annotations

import argparse
import json
import statistics
from time import perf_counter

from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    read_data,
)
from biogeme.expressions import Beta
from biogeme.expressions.log_cross_nested import LogCrossNested
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.likelihood.model_estimation import model_estimation
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.optimization import bio_bfgs
from biogeme.profiling.timing import timed_call
from biogeme.second_derivatives import SecondDerivativesMode


def build_model_elements(
    cnl_implementation: str,
) -> tuple[ModelElements, dict[str, float]]:
    database = read_data()
    asc_car = Beta('asc_car', 0, None, None, 0)
    asc_train = Beta('asc_train', 0, None, None, 0)
    asc_sm = Beta('asc_sm', 0, None, None, 1)
    beta_time = Beta('beta_time', -1, None, None, 0)
    beta_cost = Beta('beta_cost', -1, None, None, 0)
    mu_existing = Beta('mu_existing', 1.2, 1, 5, 0)
    mu_public = Beta('mu_public', 1.3, 1, 5, 0)
    alpha_existing = Beta('alpha_existing', 0.5, 0.01, 0.99, 0)

    utilities = {
        1: asc_train
        + beta_time * TRAIN_TT_SCALED
        + beta_cost * TRAIN_COST_SCALED,
        2: asc_sm + beta_time * SM_TT_SCALED + beta_cost * SM_COST_SCALED,
        3: asc_car + beta_time * CAR_TT_SCALED + beta_cost * CAR_CO_SCALED,
    }
    availability = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}
    nests = NestsForCrossNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=mu_existing,
                dict_of_alpha={1: alpha_existing, 2: 0.0, 3: 1.0},
                name='existing',
            ),
            OneNestForCrossNestedLogit(
                nest_param=mu_public,
                dict_of_alpha={1: 1 - alpha_existing, 2: 1.0, 3: 0.0},
                name='public',
            ),
        ),
    )
    expression_class = {
        'dense': LogCrossNested,
        'log-domain': LogCrossNested,
    }[cnl_implementation]
    expression = expression_class(
        util=utilities,
        av=availability,
        nests=nests,
        choice=CHOICE,
    )
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=True,
    )
    betas = {
        'asc_car': 0.2,
        'asc_train': -0.1,
        'beta_time': -1.1,
        'beta_cost': -0.8,
        'mu_existing': 1.2,
        'mu_public': 1.3,
        'alpha_existing': 0.45,
    }
    return elements, betas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', choices=('value', 'gradient', 'hessian', 'bhhh', 'estimation')
    )
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument(
        '--cnl-implementation',
        choices=('dense', 'log-domain'),
        default='dense',
    )
    args = parser.parse_args()

    elements, betas = build_model_elements(args.cnl_implementation)
    build_start = perf_counter()
    evaluator = CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=args.cnl_implementation == 'log-domain',
    )
    build_time = perf_counter() - build_start

    if args.mode == 'estimation':
        estimation_start = perf_counter()
        estimation = model_estimation(
            the_algorithm=bio_bfgs,
            function_evaluator=evaluator,
            parameters={'maxiter': 200, 'gtol': 1.0e-8},
            some_starting_values=betas,
            save_iterations_filename=None,
        )
        estimation_time = perf_counter() - estimation_start
        print(
            json.dumps(
                {
                    'mode': args.mode,
                    'cnl_implementation': args.cnl_implementation,
                    'observations': elements.database.num_rows(),
                    'build_seconds': build_time,
                    'estimation_seconds': estimation_time,
                    'solution': estimation.solution.tolist(),
                    'optimization_messages': {
                        key: str(value)
                        for key, value in estimation.optimization_messages.items()
                    },
                    'convergence': estimation.convergence,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

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
    result = {
        'mode': args.mode,
        'cnl_implementation': args.cnl_implementation,
        'observations': elements.database.num_rows(),
        'build_seconds': build_time,
        'first_call_seconds': first_call,
        'warm_median_seconds': statistics.median(warm_times),
        'warm_mean_seconds': statistics.mean(warm_times),
        'warm_min_seconds': min(warm_times),
        'warm_max_seconds': max(warm_times),
        'repeats': args.repeats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
