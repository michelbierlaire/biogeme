"""Arithmetic expressions accepted by Biogeme: comparison operators

Michel Bierlaire
Wed Mar 26 13:17:40 2025
"""

from __future__ import annotations

import logging

import pandas as pd
import pytensor.tensor as pt
from jax import Array, lax
from pytensor.tensor import TensorVariable

from .base_expressions import ExpressionOrNumeric
from .bayesian import PymcModelBuilderType
from .binary_expressions import BinaryOperator
from .jax_utils import JaxFunctionType

logger = logging.getLogger(__name__)


class ComparisonOperator(BinaryOperator):
    """Base class for comparison expressions."""

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)


class Equal(ComparisonOperator):
    """
    Logical equal
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> Equal:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(left=left_copy, right=right_copy)

    def __str__(self) -> str:
        return f'({self.left} == {self.right})'

    def __repr__(self) -> str:
        return f'Equal({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.get_value() == self.right.get_value() else 0
        return r

    def recursive_construct_jax_function(self, numerically_safe: bool):
        from jax import numpy as jnp

        left_fn = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_fn = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            is_equal = left_fn(
                parameters, one_row, the_draws, the_random_variables
            ) == right_fn(parameters, one_row, the_draws, the_random_variables)
            return lax.cond(is_equal, lambda _: 1.0, lambda _: 0.0, operand=None)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return pt.eq(left_value, right_value)

        return builder


class NotEqual(ComparisonOperator):
    """
    Logical not equal
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> NotEqual:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(left=left_copy, right=right_copy)

    def __str__(self) -> str:
        return f'({self.left} != {self.right})'

    def __repr__(self) -> str:
        return f'NotEqual({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.get_value() != self.right.get_value() else 0
        return r

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        from jax import numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> Array:
            is_not_equal = left_fn(
                parameters, one_row, the_draws, the_random_variables
            ) != right_fn(parameters, one_row, the_draws, the_random_variables)
            return lax.cond(is_not_equal, lambda _: 1.0, lambda _: 0.0, operand=None)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return pt.neq(left_value, right_value)

        return builder


class LessOrEqual(ComparisonOperator):
    """
    Logical less or equal
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> LessOrEqual:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(left=left_copy, right=right_copy)

    def __str__(self) -> str:
        return f'({self.left} <= {self.right})'

    def __repr__(self) -> str:
        return f'LessOrEqual({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.get_value() <= self.right.get_value() else 0
        return r

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        from jax import numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            is_less_or_equal = left_fn(
                parameters, one_row, the_draws, the_random_variables
            ) <= right_fn(parameters, one_row, the_draws, the_random_variables)
            return lax.cond(
                is_less_or_equal, lambda _: 1.0, lambda _: 0.0, operand=None
            )

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return pt.le(left_value, right_value)

        return builder


class GreaterOrEqual(ComparisonOperator):
    """
    Logical greater or equal
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> GreaterOrEqual:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(left=left_copy, right=right_copy)

    def __str__(self) -> str:
        return f'({self.left} >= {self.right})'

    def __repr__(self) -> str:
        return f'GreaterOrEqual({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.get_value() >= self.right.get_value() else 0
        return r

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        from jax import numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            is_greater_or_equal = left_fn(
                parameters, one_row, the_draws, the_random_variables
            ) >= right_fn(parameters, one_row, the_draws, the_random_variables)
            return lax.cond(
                is_greater_or_equal, lambda _: 1.0, lambda _: 0.0, operand=None
            )

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return pt.ge(left_value, right_value)

        return builder


class Less(ComparisonOperator):
    """
    Logical less
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> Less:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(left=left_copy, right=right_copy)

    def __str__(self) -> str:
        return f'({self.left} < {self.right})'

    def __repr__(self) -> str:
        return f'Less({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.get_value() < self.right.get_value() else 0
        return r

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        from jax import numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            is_less = left_fn(
                parameters, one_row, the_draws, the_random_variables
            ) < right_fn(parameters, one_row, the_draws, the_random_variables)
            return lax.cond(is_less, lambda _: 1.0, lambda _: 0.0, operand=None)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return pt.lt(left_value, right_value)

        return builder


class Greater(ComparisonOperator):
    """
    Logical greater
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)

    def deep_flat_copy(self) -> Greater:
        """Provides a copy of the expression. It is deep in the sense that it generates copies of the children.
        It is flat in the sense that any `MultipleExpression` is transformed into the currently selected expression.
        The flat part is irrelevant for this expression.
        """
        left_copy = self.left.deep_flat_copy()
        right_copy = self.right.deep_flat_copy()
        return type(self)(left=left_copy, right=right_copy)

    def __str__(self) -> str:
        return f'({self.left} > {self.right})'

    def __repr__(self) -> str:
        return f'Greater({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.get_value() > self.right.get_value() else 0
        return r

    def recursive_construct_jax_function(
        self, numerically_safe: bool
    ) -> JaxFunctionType:
        from jax import numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function(
            numerically_safe=numerically_safe
        )

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            is_greater = left_fn(
                parameters, one_row, the_draws, the_random_variables
            ) > right_fn(parameters, one_row, the_draws, the_random_variables)
            return lax.cond(is_greater, lambda _: 1.0, lambda _: 0.0, operand=None)

        return the_jax_function

    def recursive_construct_pymc_model_builder(self) -> PymcModelBuilderType:
        """
        Generates recursively a function to be used by PyMc. Must be overloaded by each expression
        :return: the expression in TensorVariable format, suitable for PyMc
        """
        left_pymc = self.left.recursive_construct_pymc_model_builder()
        right_pymc = self.right.recursive_construct_pymc_model_builder()

        def builder(dataframe: pd.DataFrame) -> TensorVariable:
            left_value = left_pymc(dataframe=dataframe)
            right_value = right_pymc(dataframe=dataframe)
            return pt.gt(left_value, right_value)

        return builder
