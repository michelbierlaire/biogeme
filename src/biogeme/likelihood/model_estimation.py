"""
Run a model estimation using a specified optimization algorithm.

This module defines the interface for executing a Biogeme-compatible
optimization routine and collecting its results.

Michel Bierlaire
Sun Mar 30 16:07:55 2025
"""

import logging
from datetime import datetime
from typing import Any, NamedTuple

import numpy as np
from biogeme_optimization.function import FunctionToMinimize

from biogeme.default_parameters import ParameterValue
from biogeme.jax_calculator import (
    CallableExpression,
    CompiledFormulaEvaluator,
    function_from_compiled_formula,
)
from biogeme.likelihood.negative_likelihood import NegativeLikelihood
from biogeme.optimization import OptimizationAlgorithm

logger = logging.getLogger(__name__)


class AlgorithmResults(NamedTuple):
    """
    Container for the results returned by an optimization algorithm.

    :param solution: Optimal values of the parameters as a NumPy array.
    :param optimization_messages: Dictionary with diagnostic messages from the optimizer.
    :param convergence: Boolean indicating whether the optimization terminated successfully.
    """

    solution: np.ndarray
    optimization_messages: dict[str, Any]
    convergence: bool


def optimization(
    the_algorithm: OptimizationAlgorithm,
    the_function: FunctionToMinimize,
    starting_values: np.ndarray,
    bounds: list[tuple[float, float]],
    variable_names: list[str],
    parameters: dict[str, Any],
) -> AlgorithmResults:
    """
    Run an optimization algorithm to estimate model parameters.

    :param the_algorithm: Optimization algorithm conforming to the Biogeme interface.
    :param the_function: Function to minimize, providing function value and derivatives.
    :param starting_values: Initial guess for the optimization variables.
    :param bounds: List of (lower, upper) bounds for each parameter.
    :param variable_names: Names of the variables (used for reporting or algorithm diagnostics).
    :param parameters: Dictionary of additional parameters passed to the optimization algorithm.

    :return: A tuple with:
        - x_star: the optimal solution (array of estimated parameters),
        - optimization_messages: a dictionary with diagnostic messages and timing,
        - convergence: a boolean indicating whether the optimization converged successfully.
    """
    the_function.set_variables(starting_values)
    start_time = datetime.now()
    output = the_algorithm(
        fct=the_function,
        init_betas=starting_values,
        bounds=bounds,
        variable_names=variable_names,
        parameters=parameters,
    )
    x_star, optimization_messages, convergence = output
    optimization_messages["Optimization time"] = datetime.now() - start_time
    return AlgorithmResults(
        solution=x_star,
        optimization_messages=optimization_messages,
        convergence=convergence,
    )


def model_estimation(
    the_algorithm: OptimizationAlgorithm,
    function_evaluator: CompiledFormulaEvaluator,
    parameters: dict[str, ParameterValue],
    some_starting_values: dict[str, float],
    save_iterations_filename: str | None,
) -> AlgorithmResults:
    """
    Estimate a model using the specified optimization algorithm and modeling elements.

    This function prepares the model log-likelihood function and its derivatives based
    on the provided modeling elements and starting values. It constructs the objective
    function and delegates the actual optimization to the `optimization` routine.

    :param the_algorithm: The optimization algorithm to use.
    :param function_evaluator: Object with the compiled information to evaluate the function.
    :param parameters: Dictionary of configuration parameters for the estimation.
    :param some_starting_values: Initial values for a subset or all of the model's free parameters.
    :param save_iterations_filename: If not None, the name of the file where to save the best iterations.

    :return: A tuple containing:
        - the optimal parameter values as a NumPy array,
        - a dictionary of optimization diagnostic messages,
        - a boolean indicating whether the optimization converged successfully.
    """
    starting_values = function_evaluator.model_elements.expressions_registry.complete_dict_of_free_beta_values(
        the_betas=some_starting_values
    )

    the_function: CallableExpression = function_from_compiled_formula(
        the_compiled_function=function_evaluator,
        the_betas=starting_values,
    )

    the_function_to_minimize = NegativeLikelihood(
        dimension=function_evaluator.model_elements.expressions_registry.number_of_free_betas,
        loglikelihood=the_function,
        parameters=parameters,
    )

    the_function_to_minimize.set_variables(np.array(list(starting_values.values())))
    if save_iterations_filename is not None:
        the_function_to_minimize.save_iterations(
            filename_for_best_iteration=save_iterations_filename,
            free_betas_names=function_evaluator.model_elements.expressions_registry.free_betas_names,
        )

    max_number_parameters_to_report = parameters.get('max_number_parameters_to_report')
    variable_names = (
        function_evaluator.model_elements.expressions_registry.free_betas_names
        if max_number_parameters_to_report is None
        else (
            function_evaluator.model_elements.expressions_registry.free_betas_names
            if function_evaluator.model_elements.expressions_registry.number_of_free_betas
            <= max_number_parameters_to_report
            else None
        )
    )

    return optimization(
        the_algorithm=the_algorithm,
        the_function=the_function_to_minimize,
        starting_values=function_evaluator.model_elements.expressions_registry.get_betas_array(
            starting_values
        ),
        bounds=function_evaluator.model_elements.expressions_registry.bounds,
        variable_names=variable_names,
        parameters=parameters,
    )
