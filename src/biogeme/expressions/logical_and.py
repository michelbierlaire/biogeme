"""Arithmetic expressions accepted by Biogeme: logical and

Michel Bierlaire
Sat Jun 14 2025, 10:14:27
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from biogeme.floating_point import JAX_FLOAT

from .base_expressions import ExpressionOrNumeric
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class And(BinaryOperator):
    """
    Logical and
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> And:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_left = self.left.deep_flat_copy()
        copy_right = self.right.deep_flat_copy()
        return type(self)(left=copy_left, right=copy_right)

    def __str__(self) -> str:
        return f'({self.left} and {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} and {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() == 0.0:
            return 0.0
        if self.right.get_value() == 0.0:
            return 0.0
        return 1.0

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression.
        :return: the function takes three parameters: the parameters, one row of the database, and the draws.
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

            def if_true(_):
                right_value = right_jax(
                    parameters, one_row, the_draws, the_random_variables
                )
                return jnp.where(right_value != 0.0, 1.0, 0.0)

            def if_false(_):
                return jnp.array(0.0, dtype=JAX_FLOAT)

            return jax.lax.cond(left_value != 0.0, if_true, if_false, operand=None)

        return the_jax_function
