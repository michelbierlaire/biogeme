""" Arithmetic expressions accepted by Biogeme: comparison operators

:author: Michel Bierlaire

:date: Sat Sep  9 15:20:12 2023Tue Mar 26 16:47:49 2019
"""
import logging

from .binary_expressions import BinaryOperator

logger = logging.getLogger(__name__)


class ComparisonOperator(BinaryOperator):
    """Base class for comparison expressions."""

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        BinaryOperator.__init__(self, left, right)

    def audit(self, database=None):
        """Performs various checks on the expression."""
        list_of_errors = []
        list_of_warnings = []
        if isinstance(self.left, ComparisonOperator) or isinstance(
            self.right, ComparisonOperator
        ):
            the_warning = (
                f'The following expression may potentially be ambiguous: [{self}] '
                f'if it contains the chaining of two comparisons expressions. '
                f'Keep in mind that, for Biogeme (like for Pandas), the '
                f'expression (a <= x <= b) is not equivalent to (a <= x) '
                f'and (x <= b).'
            )
            list_of_warnings.append(the_warning)
        return list_of_errors, list_of_warnings


class Equal(ComparisonOperator):
    """
    Logical equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} == {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() == self.right.getValue() else 0
        return r


class NotEqual(ComparisonOperator):
    """
    Logical not equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} != {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() != self.right.getValue() else 0
        return r


class LessOrEqual(ComparisonOperator):
    """
    Logical less or equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression

        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} <= {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() <= self.right.getValue() else 0
        return r


class GreaterOrEqual(ComparisonOperator):
    """
    Logical greater or equal
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} >= {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() >= self.right.getValue() else 0
        return r


class Less(ComparisonOperator):
    """
    Logical less
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} < {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() < self.right.getValue() else 0
        return r


class Greater(ComparisonOperator):
    """
    Logical greater
    """

    def __init__(self, left, right):
        """Constructor

        :param left: first arithmetic expression
        :type left: biogeme.expressions.Expression

        :param right: second arithmetic expression
        :type right: biogeme.expressions.Expression
        """
        ComparisonOperator.__init__(self, left, right)

    def __str__(self):
        return f'({self.left} > {self.right})'

    def getValue(self):
        """Evaluates the value of the expression

        :return: value of the expression
        :rtype: float
        """
        r = 1 if self.left.getValue() > self.right.getValue() else 0
        return r
