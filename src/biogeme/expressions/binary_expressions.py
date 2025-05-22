"""Arithmetic expressions accepted by Biogeme: binary operators

Michel Bierlaire
Wed Mar 26 09:55:46 2025
"""

from __future__ import annotations

import logging

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from biogeme.floating_point import JAX_FLOAT
from .base_expressions import Expression, ExpressionOrNumeric
from .convert import validate_and_convert
from .jax_utils import JaxFunctionType
from .numeric_expressions import Numeric

logger = logging.getLogger(__name__)


class BinaryOperator(Expression):
    """
    Base class for arithmetic expressions that are binary operators.
    This expression is the result of the combination of two expressions,
    typically addition, subtraction, multiplication or division.
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        super().__init__()
        self.left = validate_and_convert(left)
        self.right = validate_and_convert(right)

        self.children.append(self.left)
        self.children.append(self.right)


class Plus(BinaryOperator):
    """
    Addition expression
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)
        self.simplified = None
        if isinstance(self.left, Numeric) and self.left.get_value() == 0:
            self.simplified = self.right
        elif isinstance(self.right, Numeric) and self.right.get_value() == 0:
            self.simplified = self.left

    def __str__(self) -> str:
        return f'({self.left} + {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} + {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.simplified is not None:
            return self.simplified.get_value()
        return self.left.get_value() + self.right.get_value()

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.simplified is not None:
            return self.simplified.recursive_construct_jax_function()
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> float:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return left_value + right_value

        return the_jax_function


class Minus(BinaryOperator):
    """
    Subtraction expression
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)
        self.simplified = None
        if isinstance(self.right, Numeric) and self.right.get_value() == 0:
            self.simplified = self.left

    def __str__(self) -> str:
        return f'({self.left} - {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} - {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.simplified is not None:
            return self.simplified.get_value()
        return self.left.get_value() - self.right.get_value()

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.simplified is not None:
            return self.simplified.recursive_construct_jax_function()
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> float:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return left_value - right_value

        return the_jax_function


class Times(BinaryOperator):
    """
    Multiplication expression
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        super().__init__(left, right)
        self.simplified = None
        if isinstance(self.left, Numeric) and self.left.get_value() == 0:
            self.simplified = Numeric(0)
        elif isinstance(self.right, Numeric) and self.right.get_value() == 0:
            self.simplified = Numeric(0)
        elif isinstance(self.left, Numeric) and self.left.get_value() == 1:
            self.simplified = self.right
        elif isinstance(self.right, Numeric) and self.right.get_value() == 1:
            self.simplified = self.left

    def __str__(self) -> str:
        return f'({self.left} * {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} * {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.simplified is not None:
            return self.simplified.get_value()
        return self.left.get_value() * self.right.get_value()

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.simplified is not None:
            return self.simplified.recursive_construct_jax_function()
        left_jax = self.left.recursive_construct_jax_function()
        right_jax = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> float:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return left_value * right_value

        return the_jax_function


class Divide(BinaryOperator):
    """
    Division expression
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        super().__init__(left, right)
        self.simplified = None
        if isinstance(self.left, Numeric) and self.left.get_value() == 0:
            self.simplified = Numeric(0)
        elif isinstance(self.right, Numeric) and self.right.get_value() == 1:
            self.simplified = self.left

    def __str__(self) -> str:
        return f'({self.left} / {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} / {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.simplified is not None:
            return self.simplified.get_value()
        left_val = self.left.get_value()
        right_val = self.right.get_value()
        if right_val == 0.0:
            raise BiogemeError("Division by zero in Divide expression.")
        return left_val / right_val

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        if self.simplified is not None:
            return self.simplified.recursive_construct_jax_function()
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> float:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return left_value / right_value

        return the_jax_function


class BinaryMin(BinaryOperator):
    """
    Minimum of two expressions
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
        return f'BinaryMin({self.left}, {self.right})'

    def __repr__(self) -> str:
        return f'BinaryMin({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() <= self.right.get_value():
            return self.left.get_value()

        return self.right.get_value()

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return jnp.minimum(left_value, right_value)

        return the_jax_function


class BinaryMax(BinaryOperator):
    """
    Maximum of two expressions
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
        return f'BinaryMax({self.left}, {self.right})'

    def __repr__(self) -> str:
        return f'BinaryMax({repr(self.left)}, {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() >= self.right.get_value():
            return self.left.get_value()

        return self.right.get_value()

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            return jnp.maximum(left_value, right_value)

        return the_jax_function


class And(BinaryOperator):
    """
    Logical and
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
        return f'({self.left} and {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} and {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() == 0.0:
            return 0.0
        if self.right.get_value() == 0.0:
            return 0.0
        return 1.0

    def recursive_construct_jax_function(self) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression.
        :return: the function takes three parameters: the parameters, one row of the database, and the draws.
        """
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> jnp.ndarray:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            condition = (left_value != 0.0).astype(JAX_FLOAT) * (
                right_value != 0.0
            ).astype(JAX_FLOAT)
            return jnp.where(
                condition != 0.0,
                jnp.array(1.0, dtype=JAX_FLOAT),
                jnp.array(0.0, dtype=JAX_FLOAT),
            )

        return the_jax_function


class Or(BinaryOperator):
    """
    Logical or
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
        return f'({self.left} or {self.right})'

    def __repr__(self) -> str:
        return f'({repr(self.left)} or {repr(self.right)})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() != 0.0:
            return 1.0
        if self.right.get_value() != 0.0:
            return 1.0
        return 0.0

    def recursive_construct_jax_function(
        self,
    ) -> JaxFunctionType:
        """
        Generates a function to be used by biogeme_jax. Must be overloaded by each expression
        :return: the function takes two parameters: the parameters, and one row of the database.
        """
        left_jax: JaxFunctionType = self.left.recursive_construct_jax_function()
        right_jax: JaxFunctionType = self.right.recursive_construct_jax_function()

        def the_jax_function(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            the_draws: jnp.ndarray,
            the_random_variables: jnp.ndarray,
        ) -> float:
            left_value = left_jax(parameters, one_row, the_draws, the_random_variables)
            right_value = right_jax(
                parameters, one_row, the_draws, the_random_variables
            )
            condition = (left_value != 0.0) | (right_value != 0.0)
            return condition.astype(JAX_FLOAT)

        return the_jax_function
