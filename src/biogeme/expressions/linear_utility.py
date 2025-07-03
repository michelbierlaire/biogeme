"""Arithmetic expressions accepted by Biogeme: nary operators

Michel Bierlaire
Sat Sep  9 15:29:36 2023
"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp
from biogeme.exceptions import BiogemeError

from .base_expressions import Expression
from .beta_parameters import Beta
from .elementary_expressions import Elementary, TypeOfElementaryExpression
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

        if not first:
            raise BiogemeError(the_error)

        self.betas, self.variables = zip(*list_of_terms)

        self.betas = list(self.betas)  #: list of parameters

        self.variables = list(self.variables)  #: list of variables

        self.list_of_terms = list(zip(self.betas, self.variables))
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

    def dict_of_elementary_expression(
        self, the_type: TypeOfElementaryExpression
    ) -> dict[str, Elementary]:
        """Extract a dict with all elementary expressions of a specific type

        :param the_type: the type of expression
        :type  the_type: TypeOfElementaryExpression

        :return: returns a dict with the variables appearing in the
               expression the keys being their names.
        :rtype: dict(string:biogeme.expressions.Expression)

        """
        if the_type == TypeOfElementaryExpression.BETA:
            return {x.name: x for x in self.betas}

        if the_type == TypeOfElementaryExpression.FREE_BETA:
            return {x.name: x for x in self.betas if x.status == 0}

        if the_type == TypeOfElementaryExpression.FIXED_BETA:
            return {x.name: x for x in self.betas if x.status != 0}

        if the_type == TypeOfElementaryExpression.VARIABLE:
            return {x.name: x for x in self.variables}

        return {}

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
