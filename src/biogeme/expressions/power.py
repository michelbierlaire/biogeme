"""Arithmetic expressions accepted by Biogeme: power

Michel Bierlaire
10.04.2025 15:56
"""

from __future__ import annotations

import logging

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from biogeme.floating_point import LOG_CLIP_MIN
from .base_expressions import ExpressionOrNumeric
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class Power(BinaryOperator):
    """
    Power expression
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def __str__(self) -> str:
        return f'({self.left} ** {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} ** {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.get_value() ** self.right.get_value()

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )

            base = left_jax(parameters, one_row, the_draws, the_random_variables)
            exponent = right_jax(parameters, one_row, the_draws, the_random_variables)
            safe_base = jnp.clip(base, min=LOG_CLIP_MIN)
            return jnp.exp(exponent * jnp.log(safe_base))

        return the_jax_function


class PowerConstant(UnaryOperator):
    """
    Raise the argument to a constant power.
    """

    def __init__(self, child: ExpressionOrNumeric, exponent: float):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)
        self.exponent: float = exponent
        self.integer_exponent: int | None = (
            int(exponent) if exponent.is_integer() else None
        )

    def __str__(self) -> str:
        return f'{self.child}**{self.exponent}'

    def __repr__(self) -> str:
        return f'PowerConstant({repr(self.child)}, {self.exponent})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        v = self.child.get_value()
        if v == 0:
            return 0.0
        if v > 0:
            return v**self.exponent
        if self.integer_exponent is not None:
            return v**self.integer_exponent
        if v < 0:
            error_msg = f'Cannot calculate {v}**{self.exponent}'
            raise BiogemeError(error_msg)

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
            return jnp.where(
                child_value == 0, 0.0, jnp.power(child_value, self.exponent)
            )

        return the_jax_function
