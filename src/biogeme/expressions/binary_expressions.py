""" Arithmetic expressions accepted by Biogeme: binary operators

:author: Michel Bierlaire

:date: Sat Sep  9 15:18:27 2023
"""

from __future__ import annotations

import logging

from . import validate_and_convert
from .base_expressions import Expression, ExpressionOrNumeric
from ..deprecated import deprecated

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
        Expression.__init__(self)
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
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'({self.left} + {self.right})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.get_value() + self.right.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


class Minus(BinaryOperator):
    """
    Substraction expression
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'({self.left} - {self.right})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.get_value() - self.right.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


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
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'({self.left} * {self.right})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.get_value() * self.right.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


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
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'({self.left} / {self.right})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.get_value() / self.right.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


class Power(BinaryOperator):
    """
    Power expression
    """

    def __init__(self, left: ExpressionOrNumeric, right: ExpressionOrNumeric):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'({self.left} ** {self.right})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.get_value() ** self.right.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


class bioMin(BinaryOperator):
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
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'bioMin({self.left}, {self.right})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() <= self.right.get_value():
            return self.left.get_value()

        return self.right.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


class bioMax(BinaryOperator):
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
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'bioMax({self.left}, {self.right})'

    def get_value(self) -> float:
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.get_value() >= self.right.get_value():
            return self.left.get_value()

        return self.right.get_value()

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


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
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'({self.left} and {self.right})'

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

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass


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
        BinaryOperator.__init__(self, left, right)

    def __str__(self) -> str:
        return f'({self.left} or {self.right})'

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

    @deprecated(get_value)
    def getValue(self) -> float:
        """Kept for backward compatibility"""
        pass
