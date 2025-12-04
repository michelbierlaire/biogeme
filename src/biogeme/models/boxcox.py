"""Implements the Box-Cox model

:author: Michel Bierlaire
:date: Wed Oct 25 08:52:44 2023
"""

from biogeme.expressions import (
    BoxCox,
    Expression,
    ExpressionOrNumeric,
    validate_and_convert,
)


def boxcox(x: ExpressionOrNumeric, ell: ExpressionOrNumeric) -> Expression:
    """
    Box-Cox transform
    :param x: a variable to transform.
    :param ell: parameter of the transformation.

    :return: the Box-Cox transform
    """
    return BoxCox(validate_and_convert(x), validate_and_convert(ell))
