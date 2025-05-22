"""Arithmetic expressions accepted by Biogeme: Monte-Carlo integration

Michel Bierlaire
Thu Apr 24 2025, 18:47:01
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
from jax import vmap

from .base_expressions import ExpressionOrNumeric
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class MonteCarlo(UnaryOperator):
    """
    Monte Carlo integration
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def __str__(self) -> str:
        return f'MonteCarlo({self.child})'

    def __repr__(self) -> str:
        return f'MonteCarlo({repr(self.child)})'

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        child_jax = self.child.recursive_construct_jax_function()
        vectorized_function = vmap(
            lambda parameters, row, draws, random_variables: child_jax(
                parameters, row, draws, random_variables
            ),
            in_axes=(None, None, 0, None),
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            value_per_draw = vectorized_function(
                parameters, one_row, the_draws, the_random_variables
            )
            mean = jnp.mean(value_per_draw, axis=-1)
            return mean

        return the_jax_function
