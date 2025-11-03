"""Arithmetic expressions accepted by Biogeme: belongs to

Michel Bierlaire
Sat May 03 2025, 11:56:33
"""

from __future__ import annotations

import logging

import pandas as pd
import pytensor.tensor as pt
from jax import numpy as jnp

from .base_expressions import ExpressionOrNumeric
from .bayesian import PymcModelBuilderType
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class BelongsTo(UnaryOperator):
    """
    Check if a value belongs to a set
    """

    def __init__(self, child: ExpressionOrNumeric, the_set: set[float]):
        """Constructor

        :param child: arithmetic expression
        :type child: biogeme.expressions.Expression
        :param the_set: set of values
        :type the_set: set(float)
        """
        super().__init__(child)
        self.the_set: set[float] = the_set

    def deep_flat_copy(self) -> BelongsTo:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        child_copy = self.child.deep_flat_copy()
        return type(self)(child=child_copy, the_set=self.the_set)

    def __str__(self) -> str:
        return f'BelongsTo({self.child}, "{self.the_set}")'

    def __repr__(self) -> str:
        return f'BelongsTo({self.child}, "{self.the_set}")'

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
                jnp.isin(child_value, jnp.array(list(self.the_set))), 1.0, 0.0
            )

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        child_builder = self.child.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            child_val: pt.TensorVariable = child_builder(dataframe)

            # Make a constant tensor of the set values, matching dtype for safety
            set_vals = pt.constant(set_values_np, dtype=child_val.dtype)

            # Membership test:
            membership = pt.any(pt.eq(child_val[..., None], set_vals), axis=-1)

            # Return 1.0 where in set, else 0.0, with correct dtype/shape
            return pt.where(
                membership, pt.ones_like(child_val), pt.zeros_like(child_val)
            )

        return builder
