"""Arithmetic expressions accepted by Biogeme: log

Michel Bierlaire
10.04.2025 11:07
"""

from __future__ import annotations

import logging

import jax
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from jax import numpy as jnp

from biogeme.floating_point import EPSILON
from .base_expressions import ExpressionOrNumeric
from .bayesian import PymcModelBuilderType
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class log(UnaryOperator):
    """
    logarithm expression
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def deep_flat_copy(self) -> log:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_child = self.child.deep_flat_copy()
        return type(self)(child=copy_child)

    def __str__(self) -> str:
        return f'log({self.child})'

    def __repr__(self) -> str:
        return f'log({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.log(self.child.get_value())

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each
            expression
        :return: the function takes two parameters: the parameters, and one row
            of the database.
        """
        child_jax = self.child.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        if numerically_safe:

            def the_jax_function(
                parameters: jnp.ndarray,
                one_row: jnp.ndarray,
                the_draws: jnp.ndarray,
                the_random_variables: jnp.ndarray,
            ) -> jnp.ndarray:
                child_value = child_jax(
                    parameters, one_row, the_draws, the_random_variables
                )
                epsilon = EPSILON
                slope = 1.0 / epsilon
                intercept = jnp.log(epsilon) - slope * epsilon
                approx_log = slope * child_value + intercept

                return jax.lax.cond(
                    child_value < epsilon,
                    lambda _: approx_log,
                    lambda _: jnp.log(child_value),
                    operand=None,
                )
                # result = jnp.log(
                #    (child_value + jnp.sqrt(child_value**2 + EPSILON**2)) / 2
                # )
                # return result

            return the_jax_function

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            child_value = child_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return jnp.log(child_value)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        child_pymc = self.child.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            child_value = child_pymc(dataframe=dataframe)
            return pt.log(child_value)

        return builder
