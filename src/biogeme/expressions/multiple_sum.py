"""Arithmetic expressions accepted by Biogeme: multiple sum

Michel Bierlaire
Sat Sep  9 15:29:36 2023
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
import pytensor.tensor as pt
from biogeme.exceptions import BiogemeError
from biogeme.expressions.bayesian import PymcModelBuilderType

from .base_expressions import Expression, ExpressionOrNumeric
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MultipleSum(Expression):
    """This expression returns the sum of several other expressions.

    It is a generalization of 'Plus' for more than two terms
    """

    def __init__(
        self,
        list_of_expressions: list[ExpressionOrNumeric] | dict[Any:ExpressionOrNumeric],
    ):
        """Constructor

        :param list_of_expressions: list of objects representing the
                                     terms of the sum.

        :type list_of_expressions: list(biogeme.expressions.Expression)

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        :raise BiogemeError: if the list of expressions is empty
        :raise BiogemeError: if the list of expressions is neither a dict nor a list
        """
        if not list_of_expressions:
            raise BiogemeError('The argument of MultipleSum cannot be empty')

        super().__init__()

        if isinstance(list_of_expressions, dict):
            items = list_of_expressions.values()
        elif isinstance(list_of_expressions, list):
            items = list_of_expressions
        else:
            raise BiogemeError('Argument of MultipleSum must be a dict or a list.')

        for expression in items:
            self.children.append(validate_and_convert(expression))

    def deep_flat_copy(self) -> MultipleSum:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_list_of_expressions = [
            expression.deep_flat_copy() for expression in self.children
        ]
        return type(self)(list_of_expressions=copy_list_of_expressions)

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        result = 0.0
        for e in self.get_children():
            result += e.get_value()
        return result

    def __str__(self) -> str:
        s = 'MultipleSum(' + ', '.join([f'{e}' for e in self.get_children()]) + ')'
        return s

    def __repr__(self) -> str:
        return f"MultipleSum({repr(self.get_children())})"

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        compiled_children = [
            child.recursive_construct_jax_function(numerically_safe=numerically_safe)
            for child in self.get_children()
        ]

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            terms = [
                fn(parameters, one_row, the_draws, the_random_variables)
                for fn in compiled_children
            ]
            return jnp.sum(jnp.stack(terms))

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        PyMC builder for MultipleSum:
        - evaluate all children
        - stack along a new axis
        - sum over that axis
        """
        child_builders = [
            c.recursive_construct_pymc_model_builder() for c in self.get_children()
        ]

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            terms = [cb(dataframe=dataframe) for cb in child_builders]
            if len(terms) == 1:
                return terms[0]
            try:
                return pt.sum(pt.stack(terms, axis=0), axis=0)
            except (TypeError, ValueError) as e:
                shapes = [
                    getattr(getattr(t, "type", None), "shape", None) for t in terms
                ]
                raise BiogemeError(
                    f"MultipleSum terms are not shape-compatible. Got shapes: {shapes}. "
                    "All terms must be broadcastable to a common shape."
                ) from e

        return builder
