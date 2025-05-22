"""Arithmetic expressions accepted by Biogeme:exp

Michel Bierlaire
10.04.2025 11:48
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from biogeme.floating_point import MAX_EXP_ARG, MIN_EXP_ARG
from . import (
    ExpressionOrNumeric,
)
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class exp(UnaryOperator):
    """
    exponential expression
    """

    def __init__(self, child: ExpressionOrNumeric) -> None:
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def __str__(self) -> str:
        return f'exp({self.child})'

    def __repr__(self) -> str:
        return f'exp({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.exp(self.child.get_value())

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
            safe_value = jnp.clip(child_value, min=MIN_EXP_ARG, max=MAX_EXP_ARG)
            return jnp.exp(safe_value)

        return the_jax_function
