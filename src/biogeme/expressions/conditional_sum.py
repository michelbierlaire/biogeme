"""Arithmetic expressions accepted by Biogeme: ConditionalSum

Michel Bierlaire
Sat Sep  9 15:29:36 2023
"""

from __future__ import annotations

import logging
from typing import Iterable, NamedTuple

import jax
from biogeme.exceptions import BiogemeError

from .base_expressions import Expression, ExpressionOrNumeric
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class ConditionalTermTuple(NamedTuple):
    condition: ExpressionOrNumeric
    term: ExpressionOrNumeric


class ConditionalSum(Expression):
    """This expression returns the sum of a selected list of
    expressions. An expression is considered in the sum only if the
    corresponding key is True (that is, return a non-zero value).


    """

    def __init__(self, list_of_terms: Iterable[ConditionalTermTuple]):
        """Constructor

        :param list_of_terms: list containing the terms and the associated conditions

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        :raise BiogemeError: if the dict of expressions is empty
        :raise BiogemeError: if the dict of expressions is not a dict

        """
        if not list_of_terms:
            raise BiogemeError('The argument of ConditionalSum cannot be empty')

        super().__init__()

        self.list_of_terms = [
            the_term._replace(
                condition=validate_and_convert(the_term.condition),
                term=validate_and_convert(the_term.term),
            )
            for the_term in list_of_terms
        ]
        for the_term in self.list_of_terms:
            self.children.append(the_term.condition)
            self.children.append(the_term.term)

    def deep_flat_copy(self) -> ConditionalSum:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_list_of_terms: list[ConditionalTermTuple] = [
            ConditionalTermTuple(
                condition=the_term.condition.deep_flat_copy(),
                term=the_term.term.deep_flat_copy(),
            )
            for the_term in self.list_of_terms
        ]
        return type(self)(list_of_terms=copy_list_of_terms)

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        result = 0.0
        for the_term in self.list_of_terms:
            condition = the_term.condition.get_value()
            if condition != 0:
                result += the_term.term.get_value()
        return result

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        compiled_terms = [
            (
                cond.recursive_construct_jax_function(
                    numerically_safe=numerically_safe
                ),
                term.recursive_construct_jax_function(
                    numerically_safe=numerically_safe
                ),
            )
            for cond, term in self.list_of_terms
        ]

        def the_jax_function(parameters, one_row, the_draws, the_random_variables):
            result = 0.0
            for cond_fn, term_fn in compiled_terms:
                cond_val = cond_fn(parameters, one_row, the_draws, the_random_variables)

                def include_branch(_):
                    return term_fn(parameters, one_row, the_draws, the_random_variables)

                def skip_branch(_):
                    return 0.0

                result += jax.lax.cond(
                    cond_val != 0.0, include_branch, skip_branch, operand=None
                )

            return result

        return the_jax_function

    def __str__(self) -> str:
        s = (
            'ConditionalSum('
            + ', '.join([f'{k}: {v}' for k, v in self.list_of_terms])
            + ')'
        )
        return s

    def __repr__(self) -> str:
        return f"ConditionalSum({repr(self.list_of_terms)})"
