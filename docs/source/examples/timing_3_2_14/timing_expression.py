"""

Timing of any expression
========================

Michel Bierlaire
Sun Jul 27 2025, 16:55:31
"""

from biogeme.database import Database

from biogeme.expressions import Expression
from biogeme.tools.time import Timing


def timing_expression(
    the_expression: Expression,
    the_database: Database,
    number_of_draws=0,
    warm_up_runs=10,
    num_runs=10,
) -> dict[str, float]:
    print(f'Timing expression calculated {num_runs} times')
    the_timing = Timing(warm_up_runs=warm_up_runs, num_runs=num_runs)

    free_beta_values = the_expression.get_beta_values()

    results = {}

    # Timing when only the expression is needed
    average_time_function_only = the_timing.time_function(
        the_expression.get_value_and_derivatives,
        kwargs={
            'betas': free_beta_values.values(),
            'database': the_database,
            'aggregation': True,
            'prepare_ids': True,
            'gradient': False,
            'hessian': False,
            'bhhh': False,
            'number_of_draws': number_of_draws,
        },
    )

    results['Function only'] = average_time_function_only

    # Timing when the log likelihood and the gradient
    average_time_function_gradient = the_timing.time_function(
        the_expression.get_value_and_derivatives,
        kwargs={
            'betas': free_beta_values.values(),
            'database': the_database,
            'aggregation': True,
            'prepare_ids': True,
            'gradient': True,
            'hessian': False,
            'bhhh': False,
            'number_of_draws': number_of_draws,
        },
    )
    results['Function and gradient'] = average_time_function_gradient

    # Timing when the log likelihood, the gradient and hessian are needed
    average_time_function_gradient_hessian = the_timing.time_function(
        the_expression.get_value_and_derivatives,
        kwargs={
            'betas': free_beta_values.values(),
            'database': the_database,
            'aggregation': True,
            'prepare_ids': True,
            'gradient': True,
            'hessian': True,
            'bhhh': False,
            'number_of_draws': number_of_draws,
        },
    )
    results['Function, gradient and hessian'] = average_time_function_gradient_hessian

    return results
