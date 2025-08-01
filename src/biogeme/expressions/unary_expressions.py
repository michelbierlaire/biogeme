"""Arithmetic expressions accepted by Biogeme: unary operators

:author: Michel Bierlaire
:date: Sat Sep  9 15:51:53 2023
"""

from __future__ import annotations

import logging

from .base_expressions import Expression, ExpressionOrNumeric
from .convert import validate_and_convert

logger = logging.getLogger(__name__)


class UnaryOperator(Expression):
    """
    Base class for arithmetic expressions that are unary operators.

    Such an expression is the result of the modification of another
    expressions, typically changing its sign.
    """

    def __init__(self, child: ExpressionOrNumeric):
        """Constructor

        :param child: first arithmetic expression
        :type child: biogeme.expressions.Expression

        :raise BiogemeError: if one of the expressions is invalid, that is
            neither a numeric value nor a
            biogeme.expressions.Expression object.

        """
        super().__init__()
        self.child = validate_and_convert(child)
        self.children.append(self.child)
