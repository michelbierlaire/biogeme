"""Arithmetic expressions accepted by Biogeme: Monte-Carlo integration

Michel Bierlaire
Thu Apr 24 2025, 18:47:01
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import pandas as pd
from biogeme.expressions.bayesian import PymcModelBuilderType
from jax import vmap
from pytensor.tensor import TensorVariable

from .base_expressions import ExpressionOrNumeric
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class MonteCarlo(UnaryOperator):
    """
    Monte Carlo integration
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def deep_flat_copy(self) -> MonteCarlo:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_child = self.child.deep_flat_copy()
        return type(self)(child=copy_child)

    def __str__(self) -> str:
        return f'MonteCarlo({self.child})'

    def __repr__(self) -> str:
        return f'MonteCarlo({repr(self.child)})'

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
        vectorized_function = vmap(
            lambda parameters, row, draws, random_variables: child_jax(
                parameters, row, draws, random_variables
            ),
            in_axes=(None, None, 0, None),
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            value_per_draw = vectorized_function(
                parameters, one_row, the_draws, the_random_variables
            )
            mean = jnp.mean(value_per_draw, axis=-1)
            return mean

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        In PyMC, the MonteCarlo integration must not be performed. It will be handled by the Gibbs sampling.
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        child_tensor = self.child.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            child_value = child_tensor(dataframe=dataframe)
            return child_value

        return builder
