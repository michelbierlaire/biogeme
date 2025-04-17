"""Arithmetic expressions accepted by Biogeme: comparison operators

Michel Bierlaire
Wed Mar 26 13:17:40 2025
"""

from __future__ import annotations

import logging

from jax import Array

from .base_expressions import ExpressionOrNumeric
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

    def recursive_construct_jax_function(self):
        import jax.numpy as jnp

        left_fn = self.left.recursive_construct_jax_function()
        right_fn = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            return jnp.where(
                left_fn(parameters, one_row, the_draws, the_random_variables)
                == right_fn(parameters, one_row, the_draws, the_random_variables),
                1.0,
                0.0,
            )

        return the_jax_function


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

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        import jax.numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> Array:
            return jnp.where(
                left_fn(parameters, one_row, the_draws, the_random_variables)
                != right_fn(parameters, one_row, the_draws, the_random_variables),
                1.0,
                0.0,
            )

        return the_jax_function


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

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        import jax.numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            return jnp.where(
                left_fn(parameters, one_row, the_draws, the_random_variables)
                <= right_fn(parameters, one_row, the_draws, the_random_variables),
                1.0,
                0.0,
            )

        return the_jax_function


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

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        import jax.numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            return jnp.where(
                left_fn(parameters, one_row, the_draws, the_random_variables)
                >= right_fn(parameters, one_row, the_draws, the_random_variables),
                1.0,
                0.0,
            )

        return the_jax_function


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

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        import jax.numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            return jnp.where(
                left_fn(parameters, one_row, the_draws, the_random_variables)
                < right_fn(parameters, one_row, the_draws, the_random_variables),
                1.0,
                0.0,
            )

        return the_jax_function


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

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        import jax.numpy as jnp

        left_fn: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_fn: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ):
            return jnp.where(
                left_fn(parameters, one_row, the_draws, the_random_variables)
                > right_fn(parameters, one_row, the_draws, the_random_variables),
                1.0,
                0.0,
            )

        return the_jax_function
