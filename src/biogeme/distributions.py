"""Implementation of the pdf and CDF of common distributions

:author: Michel Bierlaire

:date: Thu Apr 23 12:01:49 2015

"""

import math
from typing import Union

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    MultipleSum,
    Numeric,
    exp,
    log,
    validate_and_convert,
)


def normalpdf(
    x: Union[float, Expression],
    mu: Union[float, Expression] = Numeric(0.0),
    s: Union[float, Expression] = Numeric(1.0),
) -> Expression:
    """
    Normal pdf

    Probability density function of a normal distribution

    .. math:: f(x;\\mu, \\sigma) =
        \\frac{1}{\\sigma \\sqrt{2\\pi}} \\exp{-\\frac{(x-\\mu)^2}{2\\sigma^2}}

    :param x: value at which the pdf is evaluated.

    :param mu: location parameter :math:`\\mu` of the Normal distribution.

    :param s: scale parameter :math:`\\sigma` of the Normal distribution.

    :note: It is assumed that :math:`\\sigma > 0`.

    :return: value of the Normal pdf.

    :raise ValueError: if :math:`\\sigma \\leq 0`.
    """
    x_expr = validate_and_convert(x)
    mu_expr = validate_and_convert(mu)
    s_expr = validate_and_convert(s)
    try:
        s_value = s_expr.get_value()
    except NotImplementedError:
        s_value = None

    if (s_value is not None) and (s_value <= 0):
        raise ValueError(f'Scale parameter must be positive and not {s_value}')

    d = -(x_expr - mu_expr) * (x_expr - mu_expr)
    n = Numeric(2.0) * s_expr * s_expr
    a = d / n
    num = exp(a)
    den = s_expr * Numeric(2.506628275)
    p = num / den
    return p


def normal_logpdf(
    x: Union[float, Expression],
    mu: Union[float, Expression] = Numeric(0.0),
    s: Union[float, Expression] = Numeric(1.0),
) -> Expression:
    """
    Logarithm of the Normal pdf

    .. math:: \\log f(x;\\mu, \\sigma) =
        -\\tfrac{1}{2}\\log(2\\pi) - \\log(\\sigma)
        - \\frac{(x-\\mu)^2}{2\\sigma^2}

    :param x: value at which the log-pdf is evaluated.

    :param mu: location parameter :math:`\\mu` of the Normal distribution.

    :param s: scale parameter :math:`\\sigma` of the Normal distribution.

    :note: It is assumed that :math:`\\sigma > 0`.

    :return: value of the log of the Normal pdf.

    :raise ValueError: if :math:`\\sigma \\leq 0`.
    """
    x_expr = validate_and_convert(x)
    mu_expr = validate_and_convert(mu)
    s_expr = validate_and_convert(s)

    try:
        s_value = s_expr.get_value()
    except NotImplementedError:
        s_value = None

    if (s_value is not None) and (s_value <= 0):
        raise ValueError(f'Scale parameter must be positive and not {s_value}')

    # (x - mu)^2
    diff_sq = (x_expr - mu_expr) * (x_expr - mu_expr)

    # -0.5 * log(2π)
    c = Numeric(-0.5) * log(Numeric(2.0 * math.pi))

    # - log(s)
    term_scale = -log(s_expr)

    # - (x - mu)^2 / (2 * s^2)
    term_quad = -diff_sq / (Numeric(2.0) * s_expr * s_expr)

    return c + term_scale + term_quad


def lognormalpdf(
    x: Union[float, Expression],
    mu: Union[float, Expression] = Numeric(0.0),
    s: Union[float, Expression] = Numeric(1.0),
) -> Expression:
    """
    Log normal pdf

    Probability density function of a log normal distribution

    .. math:: f(x;\\mu, \\sigma) =
              \\frac{1}{x\\sigma \\sqrt{2\\pi}}
              \\exp{-\\frac{(\\ln x-\\mu)^2}{2\\sigma^2}}


    :param x: value at which the pdf is evaluated.

    :param mu: location parameter :math:`\\mu` of the lognormal distribution.

    :param s: scale parameter :math:`\\sigma` of the lognormal distribution.

    :note: It is assumed that :math:`\\sigma > 0`, but it is not
        verified by the code.

    :return: value of the lognormal pdf.

    """
    x_expr = validate_and_convert(x)
    mu_expr = validate_and_convert(mu)
    s_expr = validate_and_convert(s)

    try:
        x_value = x_expr.get_value()
    except (NotImplementedError, BiogemeError):
        x_value = None

    if (x_value is not None) and (x_value <= 0):
        raise ValueError(f'Argument must be positive and not {x_value}')

    try:
        s_value = s_expr.get_value()
    except (NotImplementedError, BiogemeError):
        s_value = None

    if (s_value is not None) and (s_value <= 0):
        raise ValueError(f'Scale parameter must be positive and not {s_value}')

    d = -(log(x_expr) - mu_expr) * (log(x_expr) - mu_expr)
    n = Numeric(2.0) * s_expr * s_expr
    a = d / n
    num = exp(a)
    den = x_expr * s_expr * Numeric(2.506628275)
    p = (x_expr > Numeric(0)) * num / den
    return p


def uniformpdf(
    x: Union[float, Expression],
    a: Union[float, Expression] = Numeric(-1),
    b: Union[float, Expression] = Numeric(1.0),
) -> Expression:
    """
    Uniform pdf

    Probability density function of a uniform distribution.

    .. math::  f(x;a, b) = \\left\\{ \\begin{array}{ll}
              \\frac{1}{b-a} & \\text{for } x \\in [a, b] \\\\
              0 & \\text{otherwise}\\end{array} \\right.

    :param x: argument of the pdf
    :param a: lower bound :math:`a` of the distribution. Default: -1.

    :param b: upper bound :math:`b` of the distribution. Default: 1.

    :note: It is assumed that :math:`a < b`, but it is
        not verified by the code.
    :return: value of the uniform pdf.

 """
    x_expr = validate_and_convert(x)
    a_expr = validate_and_convert(a)
    b_expr = validate_and_convert(b)
    try:
        a_value = a_expr.get_value()
    except NotImplementedError:
        a_value = None
    try:
        b_value = b_expr.get_value()
    except NotImplementedError:
        b_value = None

    if a_value is not None and b_value is not None:
        if a_value > b_value:
            raise ValueError(f'Condition {a_value} <= {b_value} is not verified.')
    result = (
        (x_expr < a_expr) * Numeric(0.0)
        + (x_expr > b_expr) * Numeric(0.0)
        + (x_expr >= a_expr) * (x_expr <= b_expr) / (b_expr - a_expr)
    )
    return result


def triangularpdf(
    x: Union[float, Expression],
    a: Union[float, Expression] = Numeric(-1.0),
    b: Union[float, Expression] = Numeric(1.0),
    c: Union[float, Expression] = Numeric(0.0),
) -> Expression:
    """
    Triangular pdf

    Probability density function of a triangular distribution

    .. math:: f(x;a, b, c) = \\left\\{ \\begin{array}{ll} 0 &
             \\text{if } x < a \\\\\\frac{2(x-a)}{(b-a)(c-a)} &
             \\text{if } a \\leq x < c \\\\\\frac{2(b-x)}{(b-a)(b-c)} &
             \\text{if } c \\leq x < b \\\\0 & \\text{if } x \\geq b.
             \\end{array} \\right.

    :param x: argument of the pdf

    :param a: lower bound :math:`a` of the distribution. Default: -1.

    :param b: upper bound :math:`b` of the distribution. Default: 1.

    :param c: mode :math:`c` of the distribution. Default: 0.

    :note: It is assumed that :math:`a <  c < b`, but it is
        not verified by the code.
    :return: value of the triangular pdf.

    """
    x_expr = validate_and_convert(x)
    a_expr = validate_and_convert(a)
    b_expr = validate_and_convert(b)
    c_expr = validate_and_convert(c)
    try:
        a_value = a_expr.get_value()
    except (NotImplementedError, BiogemeError):
        a_value = None
    try:
        b_value = b_expr.get_value()
    except (NotImplementedError, BiogemeError):
        b_value = None
    try:
        c_value = c_expr.get_value()
    except (NotImplementedError, BiogemeError):
        c_value = None

    if all(var is not None for var in (a_value, b_value, c_value)):
        if c_value <= a_value or c_value >= b_value:
            error_msg = (
                f'The following condition is not verified: a [{a_value}] < '
                f'c [{c_value} < b [{b_value}]]'
            )
            raise ValueError(error_msg)

    # x < a
    r1 = (x_expr < a_expr) * Numeric(0.0)

    # a <= x < c
    r2 = (
        (x_expr >= a_expr)
        * (x_expr < c_expr)
        * Numeric(2.0)
        * ((x_expr - a_expr) / ((b_expr - a_expr) * (c_expr - a_expr)))
    )
    #  x == c
    r3 = (x_expr == c_expr) * Numeric(2.0) / (b_expr - a_expr)

    # c < x <= b
    r4 = (
        (x_expr > c_expr)
        * (x_expr <= b_expr)
        * Numeric(2.0)
        * (b_expr - x_expr)
        / ((b_expr - a_expr) * (b_expr - c_expr))
    )

    # b < x
    r5 = (x_expr > b_expr) * Numeric(0.0)
    return MultipleSum([r1, r2, r3, r4, r5])


def logisticcdf(
    x: Union[float, Expression],
    mu: Union[float, Expression] = Numeric(0.0),
    s: Union[float, Expression] = Numeric(1.0),
) -> Expression:
    """
    Logistic CDF

    Cumulative distribution function of a logistic distribution

    .. math:: f(x;\\mu, \\sigma) = \\frac{1}
        {1+\\exp\\left(-\\frac{x-\\mu}{\\sigma} \\right)}

    :param x: value at which the CDF is evaluated.

    :param mu: location parameter :math:`\\mu` of the logistic distribution.
        Default: 0.

    :param s: scale parameter :math:`\\sigma` of the logistic distribution.
        Default: 1.

    :note: It is assumed that :math:`\\sigma > 0`, but it is
        not verified by the code.

    :return: value of the logistic CDF.

    """
    x_expr = validate_and_convert(x)
    mu_expr = validate_and_convert(mu)
    s_expr = validate_and_convert(s)
    try:
        s_value = s_expr.get_value()
    except (NotImplementedError, BiogemeError):
        s_value = None
    if (s_value is not None) and (s_value <= 0):
        raise ValueError(f'Scale parameter must be positive and not {s_value}')

    result = Numeric(1.0) / (Numeric(1.0) + exp(-(x_expr - mu_expr) / s_expr))
    return result
