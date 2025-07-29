"""Arithmetic expressions accepted by Biogeme: random variable for numerical integration

Michel Bierlaire
Fri Jun 27 2025, 14:50:37
"""

from __future__ import annotations

import logging

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from .elementary_expressions import Elementary
from .elementary_types import TypeOfElementaryExpression
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class RandomVariable(Elementary):
    """
    Random variable for numerical integration
    """

    expression_type = TypeOfElementaryExpression.RANDOM_VARIABLE

    def __init__(self, name: str):
        """Constructor

        :param name: name of the random variable involved in the integration.
        :type name: string.
        """
        super().__init__(name)
        # Index of the random variable

    def deep_flat_copy(self) -> RandomVariable:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        return type(self)(name=self.name)

    @property
    def safe_rv_id(self) -> int:
        """Check the presence of the ID before using it"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for random variable {self.name}")
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
        ) -> jnp.array:
            return jnp.take(the_random_variables, self.safe_rv_id, axis=-1)

        return the_jax_function
