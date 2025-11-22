"""Arithmetic expressions accepted by Biogeme: BoxCox

Michel Bierlaire
Mon Nov 03 2025, 17:16:46
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytensor.tensor as pt

from biogeme.floating_point import JAX_FLOAT
from .base_expressions import ExpressionOrNumeric
from .bayesian import PymcModelBuilderType
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType
from .numeric_expressions import Numeric

logger = logging.getLogger(__name__)


class BoxCox(BinaryOperator):
    """
    Box–Cox transform with robust control-flow for PyMC/JAX backends.

    For x > 0:
        BC(x, λ) = log(x) * expm1(λ log x) / (λ log x)
        with the z = λ log x -> 0 limit handled by a short Taylor expansion.

    For x = 0:
        BC(0, λ) = 0    (by convention)
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: x (the data/expression to transform)
        :type left: biogeme.expressions.Expression

        :param right: lambda (shape parameter)
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> BoxCox:
        """Provides a deep copy of the expression (flat semantics as in other operators)."""
        copy_left = self.left.deep_flat_copy()
        copy_right = self.right.deep_flat_copy()
        return type(self)(left=copy_left, right=copy_right)

    def __str__(self) -> str:
        return f'BoxCox({self.left}, {self.right})'

    def __repr__(self) -> str:
        return f'BoxCox({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression for the current row (NumPy path).

        :return: Box–Cox transform value
        :rtype: float
        """
        x = self.left.get_value()
        lam = self.right.get_value()

        if x == 0.0:
            return 0.0

        # Compute z = λ log x; handle the z→0 limit with a Taylor series
        lx = np.log(x)
        z = lam * lx

        az = abs(z)
        if az < 1e-6:
            ratio = 1.0 + z / 2.0 + (z * z) / 6.0 + (z * z * z) / 24.0
        else:
            ratio = np.expm1(z) / z

        return lx * ratio

    # ---------------------- JAX BACKEND ---------------------- #

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax.
        :return: callable(parameters, one_row, the_draws, the_random_variables) -> array
        """
        x_jax: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        lam_jax: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        tol = jnp.array(1e-6, dtype=JAX_FLOAT)

        if numerically_safe:
            eps = jnp.finfo(JAX_FLOAT).eps

            def the_jax_function(
                parameters: jnp.ndarray,
                one_row: jnp.ndarray,
                the_draws: jnp.ndarray,
                the_random_variables: jnp.ndarray,
            ) -> jnp.ndarray:
                x = x_jax(parameters, one_row, the_draws, the_random_variables)
                lam = lam_jax(parameters, one_row, the_draws, the_random_variables)

                # x==0 -> 0
                is_zero_x = x == 0.0

                # For log computation, replace zeros by 1.0 (log(1)=0), then mask later
                x_pos = jnp.where(is_zero_x, jnp.array(1.0, dtype=JAX_FLOAT), x)
                lx = jnp.log(jnp.clip(x_pos, a_min=eps))  # safe log
                z = lam * lx
                az = jnp.abs(z)

                # Taylor near zero (up to cubic term)
                taylor = 1.0 + z / 2.0 + (z * z) / 6.0 + (z * z * z) / 24.0
                ratio = jnp.where(az < tol, taylor, jnp.expm1(z) / z)

                bc = lx * ratio

                # Enforce BC(0, λ) = 0 exactly
                return jnp.where(is_zero_x, jnp.array(0.0, dtype=JAX_FLOAT), bc)

            return the_jax_function

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            x = x_jax(parameters, one_row, the_draws, the_random_variables)
            lam = lam_jax(parameters, one_row, the_draws, the_random_variables)

            is_zero_x = x == 0.0
            x_pos = jnp.where(is_zero_x, jnp.array(1.0, dtype=JAX_FLOAT), x)
            lx = jnp.log(x_pos)
            z = lam * lx
            az = jnp.abs(z)

            taylor = 1.0 + z / 2.0 + (z * z) / 6.0 + (z * z * z) / 24.0
            ratio = jnp.where(az < 1e-6, taylor, jnp.expm1(z) / z)

            bc = lx * ratio
            return jnp.where(is_zero_x, jnp.array(0.0, dtype=JAX_FLOAT), bc)

        return the_jax_function

    # ---------------------- PyMC BACKEND ---------------------- #

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMC.
        :return: the expression in TensorVariable format, suitable for PyMC
        """
        x_pymc = self.left.recursive_construct_pymc_model_builder()
        lam_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            x = x_pymc(dataframe=dataframe)  # expected (N,) or broadcastable
            lam = lam_pymc(dataframe=dataframe)  # scalar or (N,)

            # x == 0 -> 0 (use switch so the dangerous path is not evaluated)
            is_zero_x = pt.eq(x, 0.0)

            # For log, replace zeros by 1.0 (log(1)=0); we mask result later
            x_pos = pt.switch(is_zero_x, 1.0, x)
            lx = pt.log(x_pos)
            z = lam * lx
            az = pt.abs(z)

            # Near zero, use Taylor; else expm1(z)/z. Use switch to avoid 0/0 when z==0.
            taylor = 1.0 + z / 2.0 + (z * z) / 6.0 + (z * z * z) / 24.0
            ratio = pt.switch(pt.lt(az, 1.0e-6), taylor, pt.expm1(z) / z)

            bc = lx * ratio
            return pt.switch(is_zero_x, 0.0, bc)

        return builder
