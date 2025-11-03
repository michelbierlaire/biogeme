"""Arithmetic expressions accepted by Biogeme: power

Michel Bierlaire
10.04.2025 15:56
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import pandas as pd
import pytensor.tensor as pt

from biogeme.floating_point import JAX_FLOAT
from .base_expressions import ExpressionOrNumeric
from .bayesian import PymcModelBuilderType
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType
from .numeric_expressions import Numeric

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

    def deep_flat_copy(self) -> Power:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_left = self.left.deep_flat_copy()
        copy_right = self.right.deep_flat_copy()
        return type(self)(left=copy_left, right=copy_right)

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
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.simplified is not None:
            return self.simplified.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )

        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        if numerically_safe:

            def the_jax_function(
                parameters: jnp.ndarray,
                one_row: jnp.ndarray,
                the_draws: jnp.ndarray,
                the_random_variables: jnp.ndarray,
            ) -> jnp.ndarray:
                base = left_jax(parameters, one_row, the_draws, the_random_variables)
                exponent = right_jax(
                    parameters, one_row, the_draws, the_random_variables
                )
                epsilon = jnp.finfo(JAX_FLOAT).eps

                def safe_power(_):
                    safe_base = jnp.clip(base, a_min=epsilon)
                    return jnp.exp(exponent * jnp.log(safe_base))

                def return_nan(_):
                    return jnp.nan

                def return_zero(_):
                    return jnp.array(0.0, dtype=JAX_FLOAT)

                return jax.lax.cond(
                    base == 0.0,
                    lambda _: return_zero(None),
                    lambda _: jax.lax.cond(
                        base < 0.0,
                        lambda _: return_nan(None),
                        safe_power,
                        operand=None,
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
            base = left_jax(parameters, one_row, the_draws, the_random_variables)
            exponent = right_jax(parameters, one_row, the_draws, the_random_variables)
            return jnp.power(base, exponent)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return left_value**right_value

        return builder
