"""Arithmetic expressions accepted by Biogeme: log

Michel Bierlaire
10.04.2025 11:07
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from biogeme.floating_point import JAX_FLOAT
from .base_expressions import ExpressionOrNumeric
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
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        child_jax = self.child.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            child_value = child_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            is_invalid = child_value <= 0
            return jnp.where(is_invalid, jnp.nan, jnp.log(child_value))

        return the_jax_function


class logzero(UnaryOperator):
    """
    logarithm expression. Returns zero if the argument is zero.
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def __str__(self) -> str:
        return f'logzero({self.child})'

    def __repr__(self) -> str:
        return f'logzero({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        v = self.child.get_value()
        return 0 if v == 0 else np.log(v)

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        child_jax = self.child.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            child_value = child_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            epsilon = jnp.finfo(JAX_FLOAT).eps
            is_zero = jnp.abs(child_value) <= epsilon
            safe_child_value = jnp.where(is_zero, 1.0, child_value)
            return jnp.where(is_zero, 0.0, jnp.log(safe_child_value))

        return the_jax_function
