"""Arithmetic expressions accepted by Biogeme: power

Michel Bierlaire
10.04.2025 15:56
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from biogeme.exceptions import BiogemeError
from biogeme.floating_point import JAX_FLOAT
from .base_expressions import ExpressionOrNumeric
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType
from .numeric_expressions import Numeric
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
        self.simplified = None
        if isinstance(left, Numeric):
            if left.value == 0:
                self.simplified = Numeric(0)
            elif left.value == 1:
                self.simplified = Numeric(1)

    def __str__(self) -> str:
        return f'({self.left} ** {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} ** {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.simplified is not None:
            return self.simplified.get_value()
        return self.left.get_value() ** self.right.get_value()

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.simplified is not None:
            return self.simplified.recursive_construct_jax_function()

        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            base = left_jax(parameters, one_row, the_draws, the_random_variables)
            exponent = right_jax(parameters, one_row, the_draws, the_random_variables)
            epsilon = jnp.finfo(JAX_FLOAT).eps
            is_zero = base == 0
            is_negative = base < 0
            safe_base = jnp.clip(base, a_min=epsilon)
            return jnp.where(
                is_zero,
                0.0,
                jnp.where(is_negative, jnp.nan, jnp.exp(exponent * jnp.log(safe_base))),
            )

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
        epsilon = np.finfo(float).eps
        if abs(exponent - round(exponent)) < epsilon:
            self.integer_exponent = int(round(exponent))
        else:
            self.integer_exponent = None

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
            is_zero = child_value == 0
            is_negative = child_value < 0
            abs_exponent = abs(self.exponent)
            epsilon = jnp.finfo(JAX_FLOAT).eps

            if self.integer_exponent is not None:
                abs_exponent = jnp.abs(self.integer_exponent)
                # This is a smooth approximation of the absolute value
                safe_value = jnp.sqrt(child_value**2 + epsilon)
                powered = safe_value**abs_exponent
                signed = jnp.where(child_value < 0, (-1) ** self.integer_exponent, 1.0)
                result = jnp.where(self.exponent < 0, 1.0 / powered, powered)

                near_zero = jnp.logical_and(
                    child_value >= -epsilon, child_value <= epsilon
                )
                if self.integer_exponent > 0:
                    return jnp.where(
                        near_zero, 0.0, jnp.where(is_zero, 0.0, result * signed)
                    )
                else:
                    return jnp.where(is_zero, 0.0, result * signed)
            else:

                return jnp.where(
                    is_zero,
                    0.0,
                    jnp.where(
                        is_negative,
                        jnp.nan,
                        jnp.exp(
                            self.exponent
                            * jnp.log(jnp.clip(child_value, a_min=epsilon))
                        ),
                    ),
                )

        return the_jax_function
