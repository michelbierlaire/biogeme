""" Arithmetic expressions accepted by Biogeme: binary operators

:author: Michel Bierlaire

:date: Sat Sep  9 15:18:27 2023
"""
import logging
import biogeme.exceptions as excep

from .base_expressions import Expression
from .numeric_tools import is_numeric

logger = logging.getLogger(__name__)


class BinaryOperator(Expression):
    """
    Base class for arithmetic expressions that are binary operators.
    This expression is the result of the combination of two expressions,
    typically addition, substraction, multiplication or division.
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value or a
            biogeme.expressions.Expression object.

        """
        Expression.__init__(self)
        if is_numeric(left):
            from .numeric_expressions import Numeric

            self.left = Numeric(left)  #: left child
        else:
            if not isinstance(left, Expression):
                raise excep.BiogemeError(f'This is not a valid expression: {left}')
            self.left = left
        if is_numeric(right):
            from .numeric_expressions import Numeric

            self.right = Numeric(right)  #: right child
        else:
            if not isinstance(right, Expression):
                raise excep.BiogemeError(f'This is not a valid expression: {right}')
            self.right = right
        self.children.append(self.left)
        self.children.append(self.right)


class Plus(BinaryOperator):
    """
    Addition expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} + {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() + self.right.getValue()


class Minus(BinaryOperator):
    """
    Substraction expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} - {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() - self.right.getValue()


class Times(BinaryOperator):
    """
    Multiplication expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} * {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() * self.right.getValue()


class Divide(BinaryOperator):
    """
    Division expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} / {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() / self.right.getValue()


class Power(BinaryOperator):
    """
    Power expression
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} ** {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        return self.left.getValue() ** self.right.getValue()


class bioMin(BinaryOperator):
    """
    Minimum of two expressions
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'bioMin({self.left}, {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() <= self.right.getValue():
            return self.left.getValue()

        return self.right.getValue()


class bioMax(BinaryOperator):
    """
    Maximum of two expressions
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'bioMax({self.left}, {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() >= self.right.getValue():
            return self.left.getValue()

        return self.right.getValue()


class And(BinaryOperator):
    """
    Logical and
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} and {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() == 0.0:
            return 0.0
        if self.right.getValue() == 0.0:
            return 0.0
        return 1.0


class Or(BinaryOperator):
    """
    Logical or
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} or {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        if self.left.getValue() != 0.0:
            return 1.0
        if self.right.getValue() != 0.0:
            return 1.0
        return 0.0
