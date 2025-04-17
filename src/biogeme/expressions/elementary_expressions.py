"""Arithmetic expressions accepted by Biogeme: elementary expressions

Michel Bierlaire
Tue Mar 25 17:34:47 2025
"""

from __future__ import annotations

import logging

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from .base_expressions import Expression
from .elementary_types import TypeOfElementaryExpression
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class Elementary(Expression):
    """Elementary expression.

    It is typically defined by a name appearing in an expression. It
    can be a variable (from the database), or a parameter (fixed or to
    be estimated using maximum likelihood), a random variable for
    numerical integration, or Monte-Carlo integration.

    """

    expression_type = None

    def __init__(self, name: str):
        """Constructor

        :param name: name of the elementary expression.
        :type name: string

        """
        super().__init__()
        self.name = name  #: name of the elementary expression

        # self.elementary_index = None
        """The index should be unique for all elementary expressions
        appearing in a given set of formulas.
        """
        self.specific_id: int | None = None  # Index of the element in its own array.

    def __str__(self) -> str:
        """string method

        :return: name of the expression
        :rtype: str
        """
        return f"{self.name}"

    def __repr__(self):
        return f'<{self.get_class_name()} name={self.name}>'

    def get_elementary_expression(self, name: str) -> Expression | None:
        """

        :return: an elementary expression from its name if it appears in the
            expression. None otherwise.
        :rtype: biogeme.Expression
        """
        if self.name == name:
            return self

        return None

    def rename_elementary(
        self, old_name: str, new_name: str, elementary_type: TypeOfElementaryExpression
    ) -> int:
        """Rename an elementary expression
        :return: number of modifications actually performed
        """
        if self.expression_type == elementary_type and self.name == old_name:
            self.name = new_name
            return 1
        return 0

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str, Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        """
        if self.expression_type == the_type:
            return {self.name: self}
        return {}

    def set_specific_id(self, name, specific_id, the_type: TypeOfElementaryExpression):
        """The elementary IDs identify the position of each element in the corresponding datab"""
        if the_type == self.expression_type and name == self.name:
            self.specific_id = specific_id


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

    def __str__(self) -> str:
        return f'Draws("{self.name}", "{self.draw_type}")'

    @property
    def safe_draw_id(self) -> int:
        """Check the presence of the draw ID before its usage"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for draw {self.name}")
        return self.specific_id

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str, Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """

        if the_type == TypeOfElementaryExpression.DRAWS:
            # Until version 3.2.13, this function returned the following:
            # return {self.name: self.drawType}
            return {self.name: self}
        return {}

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
            return jnp.take(the_draws, self.safe_draw_id, axis=-1)

        return the_jax_function


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

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str:Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """
        if the_type == TypeOfElementaryExpression.VARIABLE:
            return {self.name: self}
        return {}

    @property
    def safe_variable_id(self) -> int:
        """Check the presence of the ID before using it"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for variable {self.name}")
        return self.specific_id

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
        ) -> jnp.array:
            return jnp.take(one_row, self.safe_variable_id, axis=-1)
            # return one_row[self.variableId]

        return the_jax_function


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

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str:Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression
        """
        if the_type == TypeOfElementaryExpression.RANDOM_VARIABLE:
            return {self.name: self}
        return {}

    @property
    def safe_rv_id(self) -> int:
        """Check the presence of the ID before using it"""
        if self.specific_id is None:
            raise BiogemeError(f"No id defined for random variable {self.name}")
        return self.specific_id

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
        ) -> jnp.array:
            return jnp.take(the_random_variables, self.safe_rv_id, axis=-1)

        return the_jax_function


def get_free_beta_values(the_expression: Expression) -> dict[str, float]:
    free_beta_expressions: dict[str:Elementary] = (
        the_expression.dict_of_elementary_expression(
            the_type=TypeOfElementaryExpression.FREE_BETA
        )
    )
    return {
        the_beta.name: the_beta.get_value()
        for the_beta in free_beta_expressions.values()
    }
