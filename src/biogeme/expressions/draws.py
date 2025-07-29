"""Arithmetic expressions accepted by Biogeme: draws

Michel Bierlaire
Fri Jun 27 2025, 14:41:17
"""

from __future__ import annotations

import logging

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from .elementary_expressions import Elementary
from .elementary_types import TypeOfElementaryExpression
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class Draws(Elementary):
    """
    Draws for Monte-Carlo integration
    """

    expression_type = TypeOfElementaryExpression.DRAWS

    def __init__(self, name: str, draw_type: str):
        """Constructor

        :param name: name of the random variable with a series of draws.
        :type name: string
        :param draw_type: type of draws.
        :type draw_type: string
        """
        super().__init__(name)
        self.draw_type = draw_type
        self._is_complex = True

    def deep_flat_copy(self) -> Draws:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        return type(self)(name=self.name, draw_type=self.draw_type)

    def __str__(self) -> str:
        return f'Draws("{self.name}", "{self.draw_type}")'

    @property
    def safe_draw_id(self) -> int:
        """Check the presence of the draw ID before its usage"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for draw {self.name}")
        return self.specific_id

    def recursive_construct_jax_function(
        self, numerically_safe: bool
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
            return jnp.take(the_draws, self.safe_draw_id, axis=-1)

        return the_jax_function
