""" Implements the Box-Cox model

:author: Michel Bierlaire
:date: Wed Oct 25 08:52:44 2023
"""
import logging
from biogeme.expressions import Expression, Elem, Numeric, log, Beta

logger = logging.getLogger(__name__)


def boxcox(x: Expression, ell: Expression) -> Expression:
    """Box-Cox transform

    .. math:: B(x, \\ell) = \\frac{x^{\\ell}-1}{\\ell}.

    It has the property that

    .. math:: \\lim_{\\ell \\to 0} B(x,\\ell)=\\log(x).

    To avoid numerical difficulties, if :math:`\\ell < 10^{-5}`,
    the McLaurin approximation is used:

    .. math:: \\log(x) + \\ell \\log(x)^2 + \\frac{1}{6} \\ell^2 \\log(x)^3
              + \\frac{1}{24} \\ell^3 \\log(x)^4.

    :param x: a variable to transform.
    :param ell: parameter of the transformation.

    :return: the Box-Cox transform
    """
    if isinstance(ell, Beta) and (ell.ub is None or ell.lb is None):
        warning_msg = (
            f'It is advised to set the bounds on parameter {ell.name}. '
            f'A value of -10 and 10 should be appropriate: Beta("{ell.name}", '
            f'{ell.initValue}, -10, 10, {ell.status})'
        )
        logger.warning(warning_msg)

    regular = (x**ell - 1.0) / ell
    mclaurin = (
        log(x)
        + ell * log(x) ** 2
        + ell**2 * log(x) ** 3 / 6.0
        + ell**3 * log(x) ** 4 / 24.0
    )
    close_to_zero = (ell < Numeric(1.0e-5)) * (ell > -Numeric(1.0e-5))
    smooth = Elem({0: regular, 1: mclaurin}, close_to_zero)
    return Elem({0: smooth, 1: Numeric(0)}, x == 0)
