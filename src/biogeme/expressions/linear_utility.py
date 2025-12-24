"""Arithmetic expressions accepted by Biogeme: nary operators

Michel Bierlaire
Sat Sep  9 15:29:36 2023
"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
import pytensor.tensor as pt

from biogeme.exceptions import BiogemeError
from biogeme.expressions.bayesian import PymcModelBuilderType
from .base_expressions import Expression
from .beta_parameters import Beta
from .jax_utils import JaxFunctionType
from .variable import Variable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LinearTermTuple(NamedTuple):
    beta: Beta
    x: Variable


class LinearUtility(Expression):
    """When the utility function is linear, it is expressed as a list of
    terms, where a parameter multiplies a variable.
    """

    def __init__(self, list_of_terms: list[LinearTermTuple]):
        """Constructor

        :param list_of_terms: a list of tuple. Each tuple contains first
             a Beta parameter, second the name of a variable.
        :type list_of_terms: list(biogeme.expressions.Expression,
            biogeme.expressions.Expression)

        :raises biogeme.exceptions.BiogemeError: if the object is not
                        a list of tuples (parameter, variable)

        """
        super().__init__()

        the_error = ''
        first = True

        for b, v in list_of_terms:
            if not isinstance(b, Beta) or not isinstance(v, Variable):
                raise BiogemeError(
                    f'Each term must be a (Beta, Variable) pair. Got: ({b}, {v})'
                )

        if not first or not list_of_terms:
            raise BiogemeError(the_error)

        self.betas, self.variables = zip(*list_of_terms)

        self.betas = list(self.betas)  #: list of parameters

        self.variables = list(self.variables)  #: list of variables

        self.list_of_terms = list_of_terms
        """ List of terms """

        self.children += self.betas + self.variables

    def deep_flat_copy(self) -> LinearUtility:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_list_of_terms = [
            LinearTermTuple(beta=term[0].deep_flat_copy(), x=term[1].deep_flat_copy())
            for term in self.list_of_terms
        ]
        return type(self)(list_of_terms=copy_list_of_terms)

    def __str__(self) -> str:
        return ' + '.join([f'{b} * {x}' for b, x in self.list_of_terms])

    def __repr__(self) -> str:
        return f"LinearUtility({repr(self.list_of_terms)})"

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        beta_fns = [
            b.recursive_construct_jax_function(numerically_safe=numerically_safe)
            for b in self.betas
        ]
        variable_fns = [
            v.recursive_construct_jax_function(numerically_safe=numerically_safe)
            for v in self.variables
        ]

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            beta_values = jnp.array(
                [
                    fn(parameters, one_row, the_draws, the_random_variables)
                    for fn in beta_fns
                ]
            )
            variable_values = jnp.array(
                [
                    fn(parameters, one_row, the_draws, the_random_variables)
                    for fn in variable_fns
                ]
            )
            return jnp.dot(beta_values, variable_values)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        PyMC builder for LinearUtility:
        - evaluate Beta (scalar) and Variable (per-observation) children
        - form elementwise products beta_k * x_k
        - stack along a new axis and sum to get the linear utility per observation
        """
        # Builders for each Beta and Variable term (preserve pairing order)
        beta_builders = [b.recursive_construct_pymc_model_builder() for b in self.betas]
        var_builders = [
            v.recursive_construct_pymc_model_builder() for v in self.variables
        ]

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            # Evaluate all terms on the dataframe
            betas = [bb(dataframe=dataframe) for bb in beta_builders]
            vars_ = [vb(dataframe=dataframe) for vb in var_builders]

            if len(betas) != len(vars_):
                raise BiogemeError(
                    f"LinearUtility mismatch: {len(betas)} betas for {len(vars_)} variables."
                )

            # Form beta*x for each pair; broadcasting handles scalar beta with vector x
            try:
                products = [b * x for b, x in zip(betas, vars_)]
                if len(products) == 1:
                    return products[0]
                return pt.sum(pt.stack(products, axis=0), axis=0)
            except (TypeError, ValueError) as e:
                shape_pairs = [
                    (
                        getattr(getattr(b, "type", None), "shape", None),
                        getattr(getattr(x, "type", None), "shape", None),
                    )
                    for b, x in zip(betas, vars_)
                ]
                raise BiogemeError(
                    "LinearUtility terms are not shape-compatible. "
                    f"Got (beta_shape, var_shape) pairs: {shape_pairs}. "
                    "Each product beta_k * x_k must broadcast to a common per-observation shape."
                ) from e

        return builder
