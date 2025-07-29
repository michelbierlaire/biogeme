"""

Timing of any expression
========================

Michel Bierlaire
Sun Jul 27 2025, 16:55:31
"""

from biogeme.calculator import CompiledFormulaEvaluator
from biogeme.database import Database

from biogeme.expressions import Expression, collect_init_values
from biogeme.model_elements import ModelElements
from biogeme.second_derivatives import SecondDerivativesMode
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

    free_beta_values = collect_init_values(the_expression)
    model_elements_jit = ModelElements.from_expression_and_weight(
        log_like=the_expression,
        weight=None,
        database=the_database,
        number_of_draws=number_of_draws,
        use_jit=True,
    )
    function_jit = CompiledFormulaEvaluator(
        model_elements=model_elements_jit,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    model_elements_no_jit = ModelElements.from_expression_and_weight(
        log_like=the_expression,
        weight=None,
        database=the_database,
        number_of_draws=number_of_draws,
        use_jit=False,
    )
    function_no_jit = CompiledFormulaEvaluator(
        model_elements=model_elements_no_jit,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )

    results = {}

    # Timing when only the expression is needed
    average_time_function_only = the_timing.time_function(
        function_jit.evaluate,
        kwargs={
            'the_betas': free_beta_values,
            'gradient': False,
            'hessian': False,
            'bhhh': False,
        },
    )

    results['Function only (with JIT)'] = average_time_function_only

    average_time_function_only_no_jit = the_timing.time_function(
        function_no_jit.evaluate,
        kwargs={
            'the_betas': free_beta_values,
            'gradient': False,
            'hessian': False,
            'bhhh': False,
        },
    )

    results['Function only (without JIT)'] = average_time_function_only_no_jit

    # Timing when the log likelihood and the gradient
    average_time_function_gradient = the_timing.time_function(
        function_jit.evaluate,
        kwargs={
            'the_betas': free_beta_values,
            'gradient': True,
            'hessian': False,
            'bhhh': False,
        },
    )
    results['Function and gradient (with JIT)'] = average_time_function_gradient

    # Timing when the log likelihood and the gradient
    average_time_function_gradient_no_jit = the_timing.time_function(
        function_no_jit.evaluate,
        kwargs={
            'the_betas': free_beta_values,
            'gradient': True,
            'hessian': False,
            'bhhh': False,
        },
    )
    results['Function and gradient (without JIT)'] = (
        average_time_function_gradient_no_jit
    )

    # Timing when the log likelihood, the gradient and hessian are needed
    average_time_function_gradient_hessian = the_timing.time_function(
        function_jit.evaluate,
        kwargs={
            'the_betas': free_beta_values,
            'gradient': True,
            'hessian': True,
            'bhhh': False,
        },
    )
    results['Function, gradient and hessian (with JIT)'] = (
        average_time_function_gradient_hessian
    )

    average_time_function_gradient_hessian_no_jit = the_timing.time_function(
        function_no_jit.evaluate,
        kwargs={
            'the_betas': free_beta_values,
            'gradient': True,
            'hessian': True,
            'bhhh': False,
        },
    )
    results['Function, gradient and hessian (without JIT)'] = (
        average_time_function_gradient_hessian_no_jit
    )

    return results
