"""Arithmetic expressions accepted by Biogeme: binary operators

Michel Bierlaire
Wed Mar 26 09:55:46 2025
"""

from __future__ import annotations

import logging

from .base_expressions import Expression, ExpressionOrNumeric
from .convert import validate_and_convert

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
