"""Arithmetic expressions accepted by Biogeme: normal cdf

:author: Michel Bierlaire
:date: Sat Sep  9 15:51:53 2023
"""

from __future__ import annotations

import logging
import math

import jax.numpy as jnp
import pandas as pd
import pytensor.tensor as pt
from biogeme.expressions.bayesian import PymcModelBuilderType
from jax.scipy.stats import norm

from .base_expressions import ExpressionOrNumeric
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class NormalCdf(UnaryOperator):
    """
    Cumulative Distribution Function of a normal random variable
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def deep_flat_copy(self) -> NormalCdf:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_child = self.child.deep_flat_copy()
        return type(self)(child=copy_child)

    def __str__(self) -> str:
        return f'NormalCdf({self.child})'

    def __repr__(self) -> str:
        return f'NormalCdf({repr(self.child)})'

    def recursive_construct_jax_function(
        self, numerically_safe: book
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded
            by each expression
        :return: the function takes two parameters: the parameters,
            and one row of the database.
        """

        child_jax = self.child.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            child_value = child_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            # result = 0.5 * (1.0 + jax.lax.erf(child_value / jnp.sqrt(2.0)))
            result = norm.cdf(child_value)
            # result = jnp.clip(result, a_min=jnp.finfo(float).eps)
            return result

        return the_jax_function

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        child_value = self.child.get_value()
        return 0.5 * (1.0 + math.erf(child_value / math.sqrt(2.0)))

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        PyMC builder for NormalCdf:
        - evaluate the child expression
        - apply Φ(x) = 0.5 * (1 + erf(x / √2)) elementwise
        """
        child_builder = self.child.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            x = child_builder(dataframe=dataframe)
            # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
            return 0.5 * (1.0 + pt.erf(x / pt.sqrt(pt.as_tensor_variable(2.0))))

        return builder
