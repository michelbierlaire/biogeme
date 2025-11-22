"""Implements the Box-Cox model

:author: Michel Bierlaire
:date: Wed Oct 25 08:52:44 2023
"""

import logging

from biogeme.expressions import (
    Beta,
    Elem,
    Expression,
    ExpressionOrNumeric,
    Numeric,
    expm1,
    log,
)

logger = logging.getLogger(__name__)


def boxcox_old(x: ExpressionOrNumeric, ell: ExpressionOrNumeric) -> Expression:
    """Box-Cox transform. Old implementation, with Elem. Does not work for Bayesian estimation

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
    if isinstance(ell, Beta) and (ell.upper_bound is None or ell.lower_bound is None):
        warning_msg = (
            f'It is advised to set the bounds on parameter {ell.name}. '
            f'A value of -10 and 10 should be appropriate: Beta("{ell.name}", '
            f'{ell.init_value}, -10, 10, {ell.status})'
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


def boxcox(x: ExpressionOrNumeric, ell: ExpressionOrNumeric) -> Expression:
    """Smooth approximate Box–Cox transform without piecewise control flow.

    B(x, λ) = log(x) * [expm1(λ log x)] / (λ log x)

    Implementation details (all smooth; no Elem / no pt.switch):
      • Use log(x + ε_x) to avoid log(0).
      • Use expm1(z) / (z + ε_z) to avoid 0/0 at z ≈ 0.
      • Multiply by 1[x > 0] so the value is exactly 0 at x == 0.

    Parameters
    ----------
    x : ExpressionOrNumeric
        Variable to transform (assumed nonnegative; zeros are allowed).
    ell : ExpressionOrNumeric
        Box–Cox shape parameter.

    Returns
    -------
    Expression
        Smooth approximation to the Box–Cox transform.
    """
    # (Optional) keep the same advisory as the original implementation
    if isinstance(ell, Beta) and (ell.upper_bound is None or ell.lower_bound is None):
        warning_msg = (
            f'It is advised to set the bounds on parameter {ell.name}. '
            f'A value of -10 and 10 should be appropriate: Beta("{ell.name}", '
            f'{ell.init_value}, -10, 10, {ell.status})'
        )
        logger.warning(warning_msg)

    # Tiny epsilons to avoid singularities (introduces a tiny bias near 0)
    eps_x = Numeric(1.0e-12)  # for log(x + eps_x)
    eps_z = Numeric(1.0e-12)  # for division by (z + eps_z)

    # Core quantities
    lx = log(x + eps_x)  # finite even when x == 0
    z = ell * lx
    ratio = expm1(z) / (z + eps_z)  # finite even when z ≈ 0

    # Indicator for x > 0 ensures exact 0 at x == 0 (but no NaNs are formed anyway)
    is_pos = x > Numeric(0)

    # Final smooth approximation
    return lx * ratio * is_pos
