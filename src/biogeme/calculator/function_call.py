"""Module to create callable functions from Biogeme expressions.

This utility allows converting a symbolic Biogeme expression into a callable
function that can be evaluated with a given parameter vector.

Michel Bierlaire
Fri Mar 28 19:03:28 2025
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from biogeme.function_output import FunctionOutput, NamedFunctionOutput
from .single_formula import CompiledFormulaEvaluator
from ..constants import LOG_LIKE
from ..database import Database
from ..expressions import Expression
from ..model_elements import ModelElements
from ..second_derivatives import SecondDerivativesMode


class CallableExpression(Protocol):
    def __call__(
        self,
        x: np.ndarray,
        gradient: bool,
        hessian: bool,
        bhhh: bool,
    ) -> FunctionOutput: ...


class NamedCallableExpression(Protocol):
    def __call__(
        self,
        x: np.ndarray,
        gradient: bool,
        hessian: bool,
        bhhh: bool,
    ) -> NamedFunctionOutput: ...


def function_from_compiled_formula(
    the_compiled_function: CompiledFormulaEvaluator,
    the_betas: dict[str, float],
    named_output: bool = False,
) -> CallableExpression | NamedCallableExpression:
    """Create a callable function from a symbolic Biogeme expression.

    :param the_compiled_function: Compiled function evaluator.
    :param the_betas: Dictionary mapping parameter names to initial values.
    :param named_output: if True, the entries of the derivatives are associated with parameter names

    :return: A callable that takes a NumPy array of parameter values
             and returns a FunctionOutput.
    """

    if named_output:

        def the_named_function(
            x: np.ndarray,
            gradient: bool,
            hessian: bool,
            bhhh: bool,
        ) -> NamedFunctionOutput:
            """Evaluate the Biogeme expression with updated parameter values.

            :param x: A NumPy array of new parameter values.
            :param gradient: If True, compute the gradient of the function.
            :param hessian: If True, compute the Hessian of the function.
            :param bhhh: If True, compute the BHHH matrix (outer product of gradients).
            :return: The evaluated FunctionOutput object.
            """
            the_betas.update(dict(zip(the_betas.keys(), x)))
            result: FunctionOutput = the_compiled_function.evaluate(
                the_betas=the_betas, gradient=gradient, hessian=hessian, bhhh=bhhh
            )
            named_result = the_compiled_function.model_elements.generate_named_output(
                function_output=result
            )
            return named_result

        return the_named_function

    def the_function(
        x: np.ndarray,
        gradient: bool,
        hessian: bool,
        bhhh: bool,
    ) -> FunctionOutput:
        """Evaluate the Biogeme expression with updated parameter values.

        :param x: A NumPy array of new parameter values.
        :param gradient: If True, compute the gradient of the function.
        :param hessian: If True, compute the Hessian of the function.
        :param bhhh: If True, compute the BHHH matrix (outer product of gradients).
        :return: The evaluated FunctionOutput object.
        """
        the_betas.update(dict(zip(the_betas.keys(), x)))
        return the_compiled_function.evaluate(
            the_betas=the_betas, gradient=gradient, hessian=hessian, bhhh=bhhh
        )

    return the_function


def function_from_expression(
    expression: Expression,
    database: Database,
    numerically_safe: bool,
    use_jit: bool,
    the_betas: dict[str, float],
    number_of_draws: int | None = None,
    named_output: bool = False,
) -> CallableExpression | NamedCallableExpression:

    model_elements = ModelElements(
        expressions={LOG_LIKE: expression},
        database=database,
        number_of_draws=number_of_draws,
        use_jit=use_jit,
    )
    compiled_formula = CompiledFormulaEvaluator(
        model_elements=model_elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=numerically_safe,
    )
    the_function = function_from_compiled_formula(
        the_compiled_function=compiled_formula,
        the_betas=the_betas,
        named_output=named_output,
    )
    return the_function
