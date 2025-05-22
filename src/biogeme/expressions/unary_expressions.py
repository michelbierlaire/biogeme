"""Arithmetic expressions accepted by Biogeme: unary operators

:author: Michel Bierlaire
:date: Sat Sep  9 15:51:53 2023
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from .base_expressions import Expression, ExpressionOrNumeric
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class UnaryOperator(Expression):
    """
    Base class for arithmetic expressions that are unary operators.

    Such an expression is the result of the modification of another
    expressions, typically changing its sign.
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        super().__init__()
        self.child = validate_and_convert(child)
        self.children.append(self.child)


class UnaryMinus(UnaryOperator):
    """
    Unary minus expression

    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def __str__(self) -> str:
        return f'(-{self.child})'

    def __repr__(self) -> str:
        return f'UnaryMinus({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return -self.child.get_value()

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
            return -child_value

        return the_jax_function


class NormalCdf(UnaryOperator):
    """
    Cumulative Distribution Function of a normal random variable
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def __str__(self) -> str:
        return f'bioNormalCdf({self.child})'

    def __repr__(self) -> str:
        return f'bioNormalCdf({repr(self.child)})'

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
            return 0.5 * (1.0 + jax.lax.erf(child_value / jnp.sqrt(2.0)))

        return the_jax_function


class sin(UnaryOperator):
    """
    sine expression
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def __str__(self) -> str:
        return f'sin({self.child})'

    def __repr__(self) -> str:
        return f'sin({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.sin(self.child.get_value())

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
            return jnp.sin(child_value)

        return the_jax_function


class cos(UnaryOperator):
    """
    cosine expression
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def __str__(self) -> str:
        return f'cos({self.child})'

    def __repr__(self) -> str:
        return f'cos({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return np.cos(self.child.get_value())

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
            return jnp.cos(child_value)

        return the_jax_function
