"""Arithmetic expressions accepted by Biogeme: comparison operators

Michel Bierlaire
Wed Mar 26 13:17:40 2025
"""

from __future__ import annotations

import logging

from jax import Array, lax

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
