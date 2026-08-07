"""Benchmark full and chunked exact Hessians on the Gaussian hybrid model.

The script intentionally lives outside the example directory so diagnostic
Biogeme configuration files are created in a temporary directory. Examples::

    uv run python tests/jax/benchmark_chunked_hessian.py full
    uv run python tests/jax/benchmark_chunked_hessian.py chunked --block-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from biogeme.biogeme import BIOGEME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIRECTORY = (
    PROJECT_ROOT / 'docs/source/examples/hybrid_choice_models'
)
EXAMPLE = EXAMPLE_DIRECTORY / 'plot_h04_mode_lv_gauss_simult.py'


def build_expression():
    source = EXAMPLE.read_text()
    construction = source.split('# %%\n# Estimate the model with Biogeme.')[0]
    namespace = {'__name__': 'chunked_hessian_benchmark'}
    sys.path.insert(0, str(EXAMPLE_DIRECTORY))
    exec(compile(construction, str(EXAMPLE), 'exec'), namespace)
    return namespace['database'], namespace['log_likelihood']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=('full', 'chunked'))
    parser.add_argument('--block-size', type=int, default=4)
    parser.add_argument('--observation-batch-size', type=int)
    parser.add_argument('--rows', type=int, default=50)
    parser.add_argument('--draws', type=int, default=1_000)
    args = parser.parse_args()

    original_directory = Path.cwd()
    os.chdir(EXAMPLE_DIRECTORY)
    try:
        database, log_likelihood = build_expression()
    finally:
        os.chdir(original_directory)

    with tempfile.TemporaryDirectory(prefix='biogeme-hessian-') as temporary:
        os.chdir(temporary)
        database._df = database.dataframe.iloc[: args.rows].copy()
        biogeme = BIOGEME(
            database,
            log_likelihood,
            number_of_draws=args.draws,
            calculating_second_derivatives='analytical',
            use_jit=True,
            generate_yaml=False,
            generate_html=False,
        )
        evaluator = biogeme.function_evaluator
        parameters = biogeme.expressions_registry.free_betas_init_values

        start = perf_counter()
        if args.method == 'full':
            result = evaluator.evaluate(
                parameters, gradient=True, hessian=True, bhhh=False
            )
        else:
            result = evaluator.evaluate_chunked_hessian(
                parameters,
                block_size=args.block_size,
                observation_batch_size=args.observation_batch_size,
            )
        elapsed = perf_counter() - start

    print(
        json.dumps(
            {
                'method': args.method,
                'block_size': args.block_size if args.method == 'chunked' else None,
                'observation_batch_size': (
                    args.observation_batch_size
                    if args.method == 'chunked'
                    else None
                ),
                'rows': args.rows,
                'draws': args.draws,
                'parameters': len(parameters),
                'seconds': elapsed,
                'hessian_shape': list(result.hessian.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
