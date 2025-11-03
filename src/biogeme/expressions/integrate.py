"""Arithmetic expressions accepted by Biogeme: numerical integration

Michel Bierlaire
10.04.2025 09:27
"""

from __future__ import annotations

import logging

import jax
from jax import numpy as jnp
from numpy.polynomial.hermite import hermgauss

from biogeme.floating_point import JAX_FLOAT
from .base_expressions import ExpressionOrNumeric
from .elementary_expressions import (
    TypeOfElementaryExpression,
)
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator
from ..exceptions import BiogemeError

logger = logging.getLogger(__name__)


class IntegrateNormal(UnaryOperator):
    """
    Numerical integration
    """

    def __init__(
        self,
        child: ExpressionOrNumeric,
        name: str,
        number_of_quadrature_points: int = 30,
    ):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        :param name: name of the random variable for the integration.
        :type name: string
        """
        super().__init__(child)
        self.random_variable_name: str = name
        self.random_variable_id: int | None = (
            None  # Index of the element in its own array.
        )
        self.number_of_quadrature_points: int = number_of_quadrature_points
        self._is_complex = True

    def deep_flat_copy(self) -> IntegrateNormal:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_child = self.child.deep_flat_copy()
        return type(self)(
            child=copy_child,
            name=self.random_variable_name,
            number_of_quadrature_points=self.number_of_quadrature_points,
        )

    def set_specific_id(self, name, specific_id, the_type: TypeOfElementaryExpression):
        """The elementary IDs identify the position of each element in the corresponding datab"""
        if name == self.random_variable_name:
            if the_type != TypeOfElementaryExpression.RANDOM_VARIABLE:
                error_msg = f'Elementary expression {name} is not a random variable to be used for integration.'
                raise BiogemeError(error_msg)
            self.random_variable_id = specific_id

        for child in self.get_children():
            child.set_specific_id(name, specific_id, the_type)

    @property
    def safe_rv_id(self) -> int:
        """Check the presence of the random variable ID before its usage"""
        if self.random_variable_id is None:
            raise BiogemeError(
                f"No id defined for random variable {self.random_variable_name} inside integration expression."
            )
        return self.random_variable_id

    def __str__(self) -> str:
        return f'Integrate({self.child}, "{self.random_variable_name}")'

    def __repr__(self) -> str:
        return f'Integrate({self.child}, "{self.random_variable_name}")'

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        child_jax = jax.checkpoint(
            self.child.recursive_construct_jax_function(
                numerically_safe=numerically_safe
            )
        )
        x, w = hermgauss(self.number_of_quadrature_points)
        x = jnp.asarray(x, dtype=JAX_FLOAT)
        w = jnp.asarray(w, dtype=JAX_FLOAT)

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:

            z_vals = jnp.sqrt(2.0) * x

            def integrand(z_val):
                updated_rv = the_random_variables.at[self.safe_rv_id].set(z_val)
                val = child_jax(parameters, one_row, the_draws, updated_rv)
                return val

            values = jax.vmap(integrand)(z_vals)  # shape: (n_points, sample_size)
            result = jnp.sum(values * w) / jnp.sqrt(jnp.pi)
            return result

        return the_jax_function
