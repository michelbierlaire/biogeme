"""Arithmetic expressions accepted by Biogeme: Elem

Michel Bierlaire
Fri Apr 25 2025, 10:33:58
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
from biogeme.exceptions import BiogemeError
from jax import numpy as jnp

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

    def recursive_construct_pymc_model_builder(self):
        """Return a PyTensor builder that selects the expression associated with
        the evaluated integer key.

        Implementation detail: build a stack of branch tensors (one per key)
        and use an index derived from `argmax(key_vec == key)` to pick the
        correct slice. If the key does not match any provided key, the result
        is zero-like (same shape/dtype as a branch).
        """
        compiled_dict = {
            k: v.recursive_construct_pymc_model_builder()
            for k, v in self.dict_of_expressions.items()
        }
        key_builder = self.key_expression.recursive_construct_pymc_model_builder()

        # Fixed order of keys for stacking and indexing
        sorted_keys = sorted(compiled_dict)

        def builder(dataframe):
            import pytensor.tensor as pt
            import numpy as np

            N = len(dataframe)

            # 1) Evaluate key: must be a 1-D per-observation vector of length N
            key_val = key_builder(dataframe)
            key_shape = getattr(getattr(key_val, "type", None), "shape", None)
            key_ndim = getattr(key_val, "ndim", None)
            if key_ndim != 1:
                raise BiogemeError(
                    f"Elem key expression must be a 1-D vector (N,), got shape {key_shape}."
                )
            # If static length is known and mismatches N, fail with a clear message
            if key_shape and key_shape[0] is not None and key_shape[0] != N:
                raise BiogemeError(
                    f"Elem key expression length mismatch: expected ({N},), got {key_shape}."
                )
            key_vec = pt.cast(key_val, "int32")  # shape: (N,)

            # 2) Build each branch and verify shapes
            terms = []
            ref_shape = None
            for k in sorted_keys:
                term = compiled_dict[k](dataframe)
                t_shape = getattr(getattr(term, "type", None), "shape", None)
                t_ndim = getattr(term, "ndim", None)
                if t_ndim != 1:
                    raise BiogemeError(
                        f"Elem branch {k} must return a 1-D vector (N,), got shape {t_shape}."
                    )
                if t_shape and t_shape[0] is not None and t_shape[0] != N:
                    raise BiogemeError(
                        f"Elem branch {k} length mismatch: expected ({N},), got {t_shape}."
                    )
                if ref_shape is None:
                    ref_shape = t_shape
                terms.append(term)

            if not terms:
                raise BiogemeError("Elem has no branches to select from.")

            # 3) Map key values to branch indices via broadcasting comparison (N,K)
            keys_vec = pt.constant(np.asarray(sorted_keys, dtype=np.int32))  # (K,)
            matches = pt.eq(key_vec[:, None], keys_vec[None, :])  # (N,K)
            idx = pt.argmax(matches, axis=1)  # (N,)

            # 4) Stack branches and select per observation
            try:
                terms_stack = pt.stack(terms, axis=0)  # (K,N)
            except (TypeError, ValueError) as e:
                shapes = [
                    getattr(getattr(t, "type", None), "shape", None) for t in terms
                ]
                raise BiogemeError(
                    "Elem branches are not shape-compatible. "
                    f"Expected each branch to return (N,), got {shapes}."
                ) from e

            out = terms_stack[idx, pt.arange(N)]  # (N,)
            return out

        return builder
