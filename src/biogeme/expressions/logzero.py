"""Arithmetic expressions accepted by Biogeme: logzero

Michel Bierlaire
Sat Jun 28 2025, 12:17:05
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from biogeme.expressions.bayesian import PymcModelBuilderType
from biogeme.floating_point import EPSILON, JAX_FLOAT

from .base_expressions import ExpressionOrNumeric
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class logzero(UnaryOperator):
    """
    logarithm expression. Returns zero if the argument is zero.
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def deep_flat_copy(self) -> logzero:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_child = self.child.deep_flat_copy()
        return type(self)(child=copy_child)

    def __str__(self) -> str:
        return f'logzero({self.child})'

    def __repr__(self) -> str:
        return f'logzero({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        v = self.child.get_value()
        return 0 if v == 0 else np.log(v)

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded
            by each expression
        :return: the function takes two parameters: the parameters,
            and one row of the database.
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
                is_zero = child_value == 0.0
                slope = 1.0 / EPSILON
                intercept = jnp.log(EPSILON) - slope * EPSILON
                approx_log = slope * child_value + intercept

                return jax.lax.cond(
                    is_zero,
                    lambda _: jnp.array(0.0, dtype=JAX_FLOAT),
                    lambda _: jax.lax.cond(
                        child_value < EPSILON,
                        lambda _: approx_log,
                        lambda _: jnp.log(child_value),
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
            child_value = child_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            is_zero = child_value == 0.0

            return jax.lax.cond(
                is_zero,
                lambda _: jnp.array(0.0, dtype=JAX_FLOAT),
                lambda _: jnp.log(child_value),
                operand=None,
            )

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        child_pymc = self.child.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            x = child_pymc(dataframe=dataframe)

            # Boolean mask for zeros
            is_zero = pt.eq(x, 0.0)
            m = pt.cast(is_zero, x.dtype)  # 1 where x==0, else 0
            one_minus_m = 1.0 - m

            # Safe log: for x==0, we evaluate log(EPSILON), otherwise log(x)
            # (No branching: just add EPSILON exactly where x==0)
            safe_log = pt.log(x + m * EPSILON)

            # Enforce exact 0 when x==0, and log(x) otherwise (still branch-free)
            return one_minus_m * safe_log

        return builder
