"""Arithmetic expressions accepted by Biogeme: power constant

Michel Bierlaire
Sat Jun 28 2025, 12:20:48
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from biogeme.exceptions import BiogemeError
from biogeme.floating_point import JAX_FLOAT

from .base_expressions import ExpressionOrNumeric
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


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

    def deep_flat_copy(self) -> PowerConstant:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_child = self.child.deep_flat_copy()
        return type(self)(child=copy_child, exponent=self.exponent)

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
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        child_jax = self.child.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        if numerically_safe:

            def the_jax_function(
                parameters: jnp.ndarray,
                one_row: jnp.ndarray,
                the_draws: jnp.ndarray,
                the_random_variables: jnp.ndarray,
            ) -> jnp.ndarray:
                child_value = child_jax(
                    parameters, one_row, the_draws, the_random_variables
                )
                epsilon = jnp.finfo(JAX_FLOAT).eps

                if self.integer_exponent is not None:
                    abs_exponent = jnp.abs(self.integer_exponent)
                    safe_value = jnp.sqrt(child_value**2 + epsilon)
                    powered = safe_value**abs_exponent
                    signed = jnp.where(
                        child_value < 0, (-1) ** self.integer_exponent, 1.0
                    )
                    result = jnp.where(self.exponent < 0, 1.0 / powered, powered)

                    near_zero = jnp.logical_and(
                        child_value >= -epsilon, child_value <= epsilon
                    )

                    def zero_case(_):
                        return jnp.array(0.0, dtype=JAX_FLOAT)

                    def nonzero_case(_):
                        return result * signed

                    return jax.lax.cond(
                        near_zero if self.integer_exponent > 0 else child_value == 0.0,
                        zero_case,
                        nonzero_case,
                        operand=None,
                    )
                else:

                    def nan_branch(_):
                        return jnp.nan

                    def safe_branch(_):
                        return jnp.exp(
                            self.exponent
                            * jnp.log(jnp.clip(child_value, a_min=epsilon))
                        )

                    return jax.lax.cond(
                        child_value == 0.0,
                        lambda _: jnp.array(0.0, dtype=JAX_FLOAT),
                        lambda _: jax.lax.cond(
                            child_value < 0.0, nan_branch, safe_branch, operand=None
                        ),
                        operand=None,
                    )

            return the_jax_function

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            child_value = child_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return jnp.power(child_value, self.exponent)

        return the_jax_function
