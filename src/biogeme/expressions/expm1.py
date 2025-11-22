"""Arithmetic expressions accepted by Biogeme: expm1

Michel Bierlaire
Mon Nov 03 2025, 16:44:14
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from biogeme.floating_point import MAX_EXP_ARG, MIN_EXP_ARG
from jax import numpy as jnp

from .base_expressions import ExpressionOrNumeric
from .bayesian import PymcModelBuilderType
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class expm1(UnaryOperator):
    """
    exponential minus one expression, i.e. eˣ - 1, implemented in a numerically stable way.
    """

    def __init__(self, child: ExpressionOrNumeric) -> None:
        """Constructor

        :param child: arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)

    def deep_flat_copy(self) -> expm1:
        """Provides a deep copy of the expression."""
        copy_child = self.child.deep_flat_copy()
        return type(self)(child=copy_child)

    def __str__(self) -> str:
        return f'expm1({self.child})'

    def __repr__(self) -> str:
        return f'expm1({repr(self.child)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: eˣ - 1
        :rtype: float
        """
        return np.expm1(self.child.get_value())

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a JAX-compatible function for Biogeme-JAX.
        :return: callable(parameters, one_row, the_draws, the_random_variables)
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
                safe_value = jnp.clip(child_value, min=MIN_EXP_ARG, max=MAX_EXP_ARG)
                result = jnp.expm1(safe_value)
                return result

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
            result = jnp.expm1(child_value)
            return result

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function for PyMC representation.
        :return: the expression in TensorVariable format (PyTensor)
        """
        child_pymc = self.child.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            child_value = child_pymc(dataframe=dataframe)
            return pt.expm1(child_value)

        return builder
