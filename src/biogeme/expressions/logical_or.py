"""Arithmetic expressions accepted by Biogeme: logical or

Michel Bierlaire
Sat Jun 14 2025, 10:26:25
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import pandas as pd
from biogeme.expressions import PymcModelBuilderType
from biogeme.floating_point import JAX_FLOAT
from pytensor.tensor import TensorVariable, neq, switch

from .base_expressions import ExpressionOrNumeric
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class Or(BinaryOperator):
    """
    Logical or
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> Or:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_left = self.left.deep_flat_copy()
        copy_right = self.right.deep_flat_copy()
        return type(self)(left=copy_left, right=copy_right)

    def __str__(self) -> str:
        return f'({self.left} or {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} or {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() != 0.0:
            return 1.0
        if self.right.get_value() != 0.0:
            return 1.0
        return 0.0

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
        ) -> float:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            condition = (left_value != 0.0) | (right_value != 0.0)
            return condition.astype(JAX_FLOAT)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMC.
        Implements logical OR using numeric convention:
        - 0   → False
        - ≠0  → True
        Returns 0.0 if both sides are zero, else 1.0.
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)

            # Convert to boolean using nonzero test
            left_bool = neq(left_value, 0.0)
            right_bool = neq(right_value, 0.0)

            # Logical and, then convert back to float (0.0 or 1.0)
            return switch(left_bool | right_bool, 1.0, 0.0)

        return builder
