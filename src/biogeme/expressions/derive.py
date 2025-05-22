"""Arithmetic expressions accepted by Biogeme: Derive

Michel Bierlaire
Fri May 02 2025, 13:24:27
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from .base_expressions import ExpressionOrNumeric
from .collectors import list_of_all_betas_in_expression, list_of_variables_in_expression
from .elementary_expressions import Variable
from .jax_utils import JaxFunctionType
from .unary_expressions import UnaryOperator

logger = logging.getLogger(__name__)


class Derive(UnaryOperator):
    """
    Derivative with respect to a variable
    """

    def __init__(self, child: ExpressionOrNumeric, name: str):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression
        """
        super().__init__(child)
        # Name of the elementary expression by which the derivative is taken
        self.name = name
        # Check if it is a variable or a parameter

        list_of_variables = list_of_variables_in_expression(child)
        self.variable: Variable | None = next(
            (variable for variable in list_of_variables if variable.name == name), None
        )
        if self.variable is not None:
            self.children.append(self.variable)
        list_of_betas = list_of_all_betas_in_expression(child)
        self.beta: Variable | None = next(
            (beta for beta in list_of_betas if beta.name == name), None
        )
        if self.beta is not None:
            self.children.append(self.beta)

        if self.beta is None and self.variable is None:
            logger.warning(
                f'{name} for not appear in expression {child}. Derivative is trivially zero.'
            )

    def __str__(self) -> str:
        return f'Derive({self.child}, "{self.name}")'

    def __repr__(self) -> str:
        return f'Derive({self.child}, "{self.name}")'

    def recursive_construct_jax_function_variable(
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
            # Compute gradient with respect to the data row (i.e., Variable values)
            grad_wrt_row = jax.grad(
                lambda p, row, d, rv: child_jax(p, row, d, rv), argnums=1
            )
            # Get derivative w.r.t. Variable 'X' (assuming it’s index i in row)
            index = self.variable.safe_variable_id
            value = grad_wrt_row(parameters, one_row, the_draws, the_random_variables)[
                index
            ]
            return value

        return the_jax_function

    def recursive_construct_jax_function_beta(
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
            # Compute gradient with respect to beta
            grad_wrt_beta = jax.grad(
                lambda p, row, d, rv: child_jax(p, row, d, rv), argnums=0
            )
            # Get derivative w.r.t. Variable 'X' (assuming it’s index i in row)
            index = self.beta.safe_beta_id
            value = grad_wrt_beta(parameters, one_row, the_draws, the_random_variables)[
                index
            ]
            return value

        return the_jax_function

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.beta is not None:
            return self.recursive_construct_jax_function_beta()
        if self.variable is not None:
            return self.recursive_construct_jax_function_variable()
        # Return zero function if neither beta nor variable is matched
        return lambda p, row, d, rv: jnp.array(0.0)
