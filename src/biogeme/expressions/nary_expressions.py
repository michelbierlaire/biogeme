"""Arithmetic expressions accepted by Biogeme: nary operators

Michel Bierlaire
Sat Sep  9 15:29:36 2023
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Iterable, TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from .base_expressions import Expression
from .beta_parameters import Beta
from .convert import validate_and_convert
from .elementary_expressions import (
    Variable,
    TypeOfElementaryExpression,
    Elementary,
)
from .jax_utils import JaxFunctionType

if TYPE_CHECKING:
    from . import ExpressionOrNumeric

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
            raise BiogemeError('The argument of bioMultSum cannot be empty')

        super().__init__()

        if isinstance(list_of_expressions, dict):
            items = list_of_expressions.values()
        elif isinstance(list_of_expressions, list):
            items = list_of_expressions
        else:
            raise BiogemeError('Argument of bioMultSum must be a dict or a list.')

        for expression in items:
            self.children.append(validate_and_convert(expression))

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
        s = 'bioMultSum(' + ', '.join([f'{e}' for e in self.get_children()]) + ')'
        return s

    def __repr__(self) -> str:
        return f"MultipleSum({repr(self.get_children())})"

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        compiled_children = [
            child.recursive_construct_jax_function() for child in self.get_children()
        ]

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.array:
            result = 0.0
            for fn in compiled_children:
                result += fn(parameters, one_row, the_draws, the_random_variables)
            return result

        return the_jax_function


class Elem(Expression):
    """This returns the element of a dictionary. The key is evaluated
    from an expression and must return an integer, possibly negative.
    """

    def __init__(
        self,
        dict_of_expressions: dict[int, ExpressionOrNumeric],
        key_expression: ExpressionOrNumeric,
    ):
        """Constructor

        :param dict_of_expressions: dict of objects with numerical keys.
        :type dict_of_expressions: dict(int: biogeme.expressions.Expression)

        :param key_expression: object providing the key of the element
                              to be evaluated.
        :type key_expression: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.
        """
        super().__init__()

        self.key_expression = validate_and_convert(key_expression)
        self.children.append(self.key_expression)

        self.dict_of_expressions = {}  #: dict of expressions
        for k, v in dict_of_expressions.items():
            self.dict_of_expressions[k] = validate_and_convert(v)
            self.children.append(self.dict_of_expressions[k])

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float

        :raise BiogemeError: if the calculated key is not present in
            the dictionary.
        """
        try:
            key = int(self.key_expression.get_value())
            return self.dict_of_expressions[key].get_value()
        except (ValueError, KeyError):
            raise BiogemeError(
                f'Invalid or missing key: {key}. Available keys: {self.dict_of_expressions.keys()}'
            )

    def __str__(self) -> str:
        s = '{{'
        first = True
        for k, v in self.dict_of_expressions.items():
            if first:
                s += f'{k}:{v}'
                first = False
            else:
                s += f', {k}:{v}'
        s += f'}}[{self.key_expression}]'
        return s

    def __repr__(self) -> str:
        return f"Elem({repr(self.dict_of_expressions)}, {repr(self.key_expression)})"

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        compiled_dict = {
            k: v.recursive_construct_jax_function()
            for k, v in self.dict_of_expressions.items()
        }
        key_fn = self.key_expression.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            key = key_fn(parameters, one_row, the_draws, the_random_variables)
            key_int = jnp.asarray(key, dtype=jnp.int32)

            def dispatch(k):
                return compiled_dict[k](
                    parameters, one_row, the_draws, the_random_variables
                )

            branches = [
                lambda parameters, one_row, the_draws, the_random_variables, fn=compiled_dict[
                    k
                ]: fn(
                    parameters, one_row, the_draws, the_random_variables
                )
                for k in sorted(compiled_dict)
            ]
            return jax.lax.switch(
                key_int, branches, parameters, one_row, the_draws, the_random_variables
            )

        return the_jax_function


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
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        beta_fns = [b.recursive_construct_jax_function() for b in self.betas]
        variable_fns = [v.recursive_construct_jax_function() for v in self.variables]

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
