"""Arithmetic expressions accepted by Biogeme: numeric expressions

Michel Bierlaire
Tue Mar 25 18:41:06 2025
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import pandas as pd
import pytensor.tensor as pt
from biogeme.bayesian_estimation import check_shape
from pytensor import config as pt_config
from pytensor.tensor import TensorVariable

from .base_expressions import Expression
from .bayesian import PymcModelBuilderType
from .jax_utils import JaxFunctionType
from ..floating_point import JAX_FLOAT

logger = logging.getLogger(__name__)


class Numeric(Expression):
    """
    Numerical expression for a simple number
    """

    def __init__(self, value: float | int | bool):
        """Constructor

        :param value: numerical value
        :type value: float
        """
        super().__init__()
        self.value = float(value)  #: numeric value

    def deep_flat_copy(self) -> Numeric:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        return type(self)(value=self.value)

    def __str__(self) -> str:
        return '`' + str(self.value) + '`'

    def __repr__(self):
        return f'<Numeric value={self.value}>'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.value

    def recursive_construct_jax_function(
        self, numerically_safe: bool
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
            return jnp.array(self.value, dtype=JAX_FLOAT)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """

        @check_shape
        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            # Produce a constant vector of length len(dataframe) with the numeric value
            n = len(dataframe)
            return pt.full((n,), self.value, dtype=pt_config.floatX)

        return builder
