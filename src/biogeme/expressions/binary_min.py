"""Arithmetic expressions accepted by Biogeme: binary min

Michel Bierlaire
Sat Jun 14 2025, 10:10:57
"""

from __future__ import annotations

import logging

import pandas as pd
from jax import numpy as jnp
from pytensor.tensor import TensorVariable, minimum

from .base_expressions import ExpressionOrNumeric
from .bayesian import PymcModelBuilderType
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class BinaryMin(BinaryOperator):
    """
    Minimum of two expressions
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> BinaryMin:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(left=left_copy, right=right_copy)

    def __str__(self) -> str:
        return f'BinaryMin({self.left}, {self.right})'

    def __repr__(self) -> str:
        return f'BinaryMin({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() <= self.right.get_value():
            return self.left.get_value()

        return self.right.get_value()

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return jnp.minimum(left_value, right_value)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return minimum(left_value, right_value)

        return builder
