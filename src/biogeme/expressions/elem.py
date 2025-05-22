"""Arithmetic expressions accepted by Biogeme: Elem

Michel Bierlaire
Fri Apr 25 2025, 10:33:58
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from .base_expressions import Expression
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType

if TYPE_CHECKING:
    from . import ExpressionOrNumeric

logger = logging.getLogger(__name__)


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
        self._key_depends_on_parameters = self.key_expression.embed_expression('Beta')
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

        sorted_keys = sorted(compiled_dict)
        key_array = jnp.array(sorted_keys)

        branches = [
            lambda p, r, d, rv, fn=compiled_dict[k]: fn(p, r, d, rv)
            for k in sorted_keys
        ]

        def the_jax_function(parameters, one_row, the_draws, the_random_variables):
            key_value = key_fn(parameters, one_row, the_draws, the_random_variables)
            # if self._key_depends_on_parameters and isinstance(
            #    key_value, jax.core.Tracer
            # ):
            #    logger.warning(
            #        "The key expression depends on a parameter and cannot be used "
            #        "when computing derivatives, as it defines a non-differentiable control flow."
            #    )
            key_int = jnp.asarray(key_value, dtype=jnp.int32)
            matches = key_array == key_int
            branch_index = jnp.argmax(matches)

            return jax.lax.switch(
                branch_index,
                branches,
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )

        return the_jax_function
