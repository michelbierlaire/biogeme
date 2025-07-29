"""Arithmetic expressions accepted by Biogeme: variables

Michel Bierlaire
Fri Jun 27 2025, 14:43:42
"""

from __future__ import annotations

import logging

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from .elementary_expressions import Elementary
from .elementary_types import TypeOfElementaryExpression
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class Variable(Elementary):
    """Explanatory variable

    This represents the explanatory variables of the choice
    model. Typically, they come from the data set.
    """

    expression_type = TypeOfElementaryExpression.VARIABLE

    def __init__(self, name: str):
        """Constructor

        :param name: name of the variable.
        :type name: string
        """
        super().__init__(name)

    def deep_flat_copy(self) -> Variable:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        return type(self)(name=self.name)

    @property
    def safe_variable_id(self) -> int:
        """Check the presence of the ID before using it"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for variable {self.name}")
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
            return jnp.take(one_row, self.safe_variable_id, axis=-1)
            # return one_row[self.variableId]

        return the_jax_function
