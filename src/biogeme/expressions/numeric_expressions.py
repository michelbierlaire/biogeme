"""Arithmetic expressions accepted by Biogeme: numeric expressions

Michel Bierlaire
Tue Mar 25 18:41:06 2025
"""

from __future__ import annotations
import logging

import jax.numpy as jnp

from .base_expressions import Expression
from .jax_utils import JaxFunctionType
from ..floating_point import JAX_FLOAT

logger = logging.getLogger(__name__)


class Numeric(Expression):
    """
    Numerical expression for a simple number
    """

    def __init__(self, value: float | int | bool):
        """Constructor

        :param value: numerical value
        :type value: float
        """
        super().__init__()
        self.value = float(value)  #: numeric value

    def __str__(self) -> str:
        return '`' + str(self.value) + '`'

    def __repr__(self):
        return f'<Numeric value={self.value}>'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.value

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            return jnp.array(self.value, dtype=JAX_FLOAT)

        return the_jax_function
