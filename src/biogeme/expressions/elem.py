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
from .beta_parameters import Beta
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
        self._key_depends_on_parameters = self.key_expression.embed_expression(Beta)
        self.children.append(self.key_expression)

        self.dict_of_expressions = {}  #: dict of expressions
        for k, v in dict_of_expressions.items():
            self.dict_of_expressions[k] = validate_and_convert(v)
            self.children.append(self.dict_of_expressions[k])

    def deep_flat_copy(self) -> Elem:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        copy_dict_of_expressions = {
            key: expression.deep_flat_copy()
            for key, expression in self.dict_of_expressions.items()
        }
        copy_key = self.key_expression.deep_flat_copy()
        return type(self)(
            dict_of_expressions=copy_dict_of_expressions, key_expression=copy_key
        )

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
                f'Invalid or missing key: {key}. '
                f'Available keys: {self.dict_of_expressions.keys()}'
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

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        compiled_dict = {
            k: v.recursive_construct_jax_function(numerically_safe=numerically_safe)
            for k, v in self.dict_of_expressions.items()
        }
        key_fn = self.key_expression.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        sorted_keys = sorted(compiled_dict)
        key_array = jnp.array(sorted_keys)

        def make_branch(fn, k):
            def wrapped(*args):

                result = fn(*args)
                return result

            return wrapped

        branches = [make_branch(compiled_dict[k], k) for k in sorted_keys]

        # branches = [
        #    lambda p, r, d, rv, fn=compiled_dict[k]: fn(p, r, d, rv)
        #    for k in sorted_keys
        # ]

        def the_jax_function(parameters, one_row, the_draws, the_random_variables):
            key_value = key_fn(parameters, one_row, the_draws, the_random_variables)
            key_int = jnp.asarray(key_value, dtype=jnp.int32)
            matches = key_array == key_int
            branch_index = jnp.argmax(matches)
            result = jax.lax.switch(
                branch_index,
                branches,
                parameters,
                one_row,
                the_draws,
                the_random_variables,
            )
            return result

        return the_jax_function
