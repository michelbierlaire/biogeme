"""Arithmetic expressions accepted by Biogeme: weighted log-sum-exp.

Michel Bierlaire
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import pandas as pd
import pytensor.tensor as pt

from biogeme.exceptions import BiogemeError
from biogeme.expressions.bayesian import PymcModelBuilderType
from biogeme.floating_point import JAX_FLOAT

from .base_expressions import Expression, ExpressionOrNumeric
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


def _compile_expression_list_to_jax_function(
    expressions: list[Expression],
    numerically_safe: bool,
) -> JaxFunctionType:
    """Compile a list of scalar expressions into one stacked JAX function.

    The generic expression infrastructure compiles each child expression into
    a scalar JAX function. This helper keeps that compilation local, but exposes
    a single function returning a stacked array. It reduces duplicated list
    handling in expression nodes that naturally operate on vectors of children.
    """
    compiled_expressions = tuple(
        expression.recursive_construct_jax_function(numerically_safe=numerically_safe)
        for expression in expressions
    )

    def the_jax_function(
        parameters: jnp.ndarray,
        one_row: jnp.ndarray,
        the_draws: jnp.ndarray,
        the_random_variables: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.stack(
            [
                compiled_expression(
                    parameters, one_row, the_draws, the_random_variables
                )
                for compiled_expression in compiled_expressions
            ],
            axis=0,
        )

    return the_jax_function


class WeightedLogSumExp(Expression):
    """Weighted log-sum-exp expression.

    It represents

    .. math::

        \\log\\left(\\sum_i w_i \\exp(x_i)\\right).

    By convention, if the weighted sum is exactly zero, the expression
    returns zero, consistently with ``logzero(MultipleSum(...))``.
    """

    def __init__(
        self,
        terms: list[ExpressionOrNumeric],
        weights: list[ExpressionOrNumeric],
    ):
        """Constructor.

        :param terms: expressions :math:`x_i`.
        :param weights: expressions :math:`w_i`.

        :raise BiogemeError: if the two lists do not have the same length.
        :raise BiogemeError: if the lists are empty.
        """
        if not terms:
            raise BiogemeError('The list of terms cannot be empty.')

        if len(terms) != len(weights):
            raise BiogemeError(
                f'WeightedLogSumExp received {len(terms)} terms '
                f'and {len(weights)} weights.'
            )

        super().__init__()

        self.number_of_terms = len(terms)

        for term in terms:
            self.children.append(validate_and_convert(term))

        for weight in weights:
            self.children.append(validate_and_convert(weight))

    @property
    def terms(self):
        """Terms :math:`x_i`."""
        return self.children[: self.number_of_terms]

    @property
    def weights(self):
        """Weights :math:`w_i`."""
        return self.children[self.number_of_terms :]

    def deep_flat_copy(self) -> WeightedLogSumExp:
        """Deep flat copy of the expression."""
        copied_terms = [term.deep_flat_copy() for term in self.terms]
        copied_weights = [weight.deep_flat_copy() for weight in self.weights]
        return type(self)(terms=copied_terms, weights=copied_weights)

    def get_value(self) -> float:
        """Evaluate the value of the expression."""
        weighted_sum = 0.0
        for term, weight in zip(self.terms, self.weights, strict=True):
            weighted_sum += weight.get_value() * jnp.exp(term.get_value())

        return 0.0 if weighted_sum == 0.0 else float(jnp.log(weighted_sum))

    def __str__(self) -> str:
        entries = [
            f'{weight} * exp({term})'
            for term, weight in zip(self.terms, self.weights, strict=True)
        ]
        return f'WeightedLogSumExp({", ".join(entries)})'

    def __repr__(self) -> str:
        return (
            f'WeightedLogSumExp(terms={repr(self.terms)}, weights={repr(self.weights)})'
        )

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        """Generate a JAX function for weighted log-sum-exp.

        Terms and weights are each compiled into one stacked JAX function.
        This keeps the weighted log-sum-exp node closer to a vector primitive
        and avoids maintaining one runtime list for all children together.
        """
        terms_jax = _compile_expression_list_to_jax_function(
            expressions=self.terms,
            numerically_safe=numerically_safe,
        )
        weights_jax = _compile_expression_list_to_jax_function(
            expressions=self.weights,
            numerically_safe=numerically_safe,
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            stacked_terms = terms_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            stacked_weights = weights_jax(
                parameters, one_row, the_draws, the_random_variables
            )

            log_sum = jax.nn.logsumexp(
                stacked_terms,
                axis=0,
                b=stacked_weights,
            )

            return jnp.where(
                jnp.isneginf(log_sum),
                jnp.array(0.0, dtype=JAX_FLOAT),
                log_sum,
            )

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """PyMC builder for WeightedLogSumExp."""
        child_builders = tuple(
            child.recursive_construct_pymc_model_builder() for child in self.children
        )
        number_of_terms = self.number_of_terms

        def builder(dataframe: pd.DataFrame) -> pt.TensorVariable:
            child_values = tuple(
                child_builder(dataframe=dataframe) for child_builder in child_builders
            )
            term_values = child_values[:number_of_terms]
            weight_values = child_values[number_of_terms:]

            weighted_terms = [
                weight * pt.exp(term)
                for term, weight in zip(term_values, weight_values, strict=True)
            ]

            weighted_sum = pt.sum(pt.stack(weighted_terms, axis=0), axis=0)

            return pt.switch(
                pt.eq(weighted_sum, 0.0),
                0.0,
                pt.log(weighted_sum),
            )

        return builder


def weighted_logsumexp(
    terms: list[ExpressionOrNumeric],
    weights: list[ExpressionOrNumeric],
) -> Expression:
    """Factory function for weighted log-sum-exp."""
    return WeightedLogSumExp(terms=terms, weights=weights)
