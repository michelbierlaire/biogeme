""" Implementation of the pdf and CDF of common distributions

:author: Michel Bierlaire

:date: Thu Apr 23 12:01:49 2015

"""

# There seems to be a bug in PyLint.
# pylint: disable=invalid-unary-operand-type, too-many-function-args

# Too constraining
# pylint: disable=invalid-name, too-many-arguments, too-many-locals, too-many-statements,
# pylint: disable=too-many-branches, too-many-instance-attributes, too-many-lines,
# pylint: disable=too-many-public-methods

from biogeme.expressions import log, exp, Numeric


def normalpdf(x, mu=Numeric(0.0), s=Numeric(1.0)):
    """
    Normal pdf

    Probability density function of a normal distribution

    .. math:: f(x;\\mu, \\sigma) =
        \\frac{1}{\\sigma \\sqrt{2\\pi}} \\exp{-\\frac{(x-\\mu)^2}{2\\sigma^2}}

    :param x: value at which the pdf is evaluated.
    :type x: float or biogeme.expression
    :param mu: location parameter :math:`\\mu` of the Normal distribution.
        Default: 0.
    :type mu: float or biogeme.expression
    :param s: scale parameter :math:`\\sigma` of the Normal distribution.
        Default: 1.
    :type s: float or biogeme.expression

    :note: It is assumed that :math:`\\sigma > 0`, but it is not
        verified by the code.

    :return: value of the Normal pdf.
    :rtype: float or biogeme.expression"""
    d = -(x - mu) * (x - mu)
    n = 2.0 * s * s
    a = d / n
    num = exp(a)
    den = s * 2.506628275
    p = num / den
    return p


def lognormalpdf(x, mu=0.0, s=1.0):
    """
    Log normal pdf

    Probability density function of a log normal distribution

    .. math:: f(x;\\mu, \\sigma) =
              \\frac{1}{x\\sigma \\sqrt{2\\pi}}
              \\exp{-\\frac{(\\ln x-\\mu)^2}{2\\sigma^2}}


    :param x: value at which the pdf is evaluated.
    :type x: float or biogeme.expression
    :param mu: location parameter :math:`\\mu` of the lognormal distribution.
        Default: 0.
    :type mu: float or biogeme.expression
    :param s: scale parameter :math:`\\sigma` of the lognormal distribution.
        Default: 1.
    :type s: float or biogeme.expression

    :note: It is assumed that :math:`\\sigma > 0`, but it is not
        verified by the code.

    :return: value of the lognormal pdf.
    :rtype: float or biogeme.expression

    """
    d = -(log(x) - mu) * (log(x) - mu)
    n = 2.0 * s * s
    a = d / n
    num = exp(a)
    den = x * s * 2.506628275
    p = (x > 0) * num / den
    return p


def uniformpdf(x, a=-1, b=1.0):
    """
    Uniform pdf

    Probability density function of a uniform distribution.

    .. math::  f(x;a, b) = \\left\\{ \\begin{array}{ll}
              \\frac{1}{b-a} & \\text{for } x \\in [a, b] \\\\
              0 & \\text{otherwise}\\end{array} \\right.

    :param x: argument of the pdf
    :type x: float or biogeme.expression
    :param a: lower bound :math:`a` of the distribution. Default: -1.
    :type a: float or biogeme.expression
    :param b: upper bound :math:`b` of the distribution. Default: 1.
    :type b: float or biogeme.expression
    :note: It is assumed that :math:`a < b`, but it is
        not verified by the code.
    :return: value of the uniform pdf.
    :rtype: float or biogeme.expression
 """
    result = (x < a) * 0.0 + (x >= b) * 0.0 + (x >= a) * (x < b) / (b - a)
    return result


def triangularpdf(x, a=-1.0, b=1.0, c=0.0):
    """
    Triangular pdf

    Probability density function of a triangular distribution

    .. math:: f(x;a, b, c) = \\left\\{ \\begin{array}{ll} 0 &
             \\text{if } x < a \\\\\\frac{2(x-a)}{(b-a)(c-a)} &
             \\text{if } a \\leq x < c \\\\\\frac{2(b-x)}{(b-a)(b-c)} &
             \\text{if } c \\leq x < b \\\\0 & \\text{if } x \\geq b.
             \\end{array} \\right.

    :param x: argument of the pdf
    :type x: float or biogeme.expression
    :param a: lower bound :math:`a` of the distribution. Default: -1.
    :type a: float or biogeme.expression
    :param b: upper bound :math:`b` of the distribution. Default: 1.
    :type b: float or biogeme.expression
    :param c: mode :math:`c` of the distribution. Default: 0.
    :type c: float or biogeme.expression
    :note: It is assumed that :math:`a <  c < b`, but it is
        not verified by the code.
    :return: value of the triangular pdf.
    :rtype: float or biogeme.expression

    """
    result = (
        (x < a) * 0.0
        + (x >= b) * 0.0
        + (x >= a)
        * (x < c)
        * 2.0
        * ((x - a) / ((b - a) * (c - a)))
        * (x >= c)
        * (x < b)
        * 2.0
        * (b - x)
        / ((b - a) * (b - c))
    )
    return result


def logisticcdf(x, mu=Numeric(0.0), s=Numeric(1.0)):
    """
    Logistic CDF

    Cumulative distribution function of a logistic distribution

    .. math:: f(x;\\mu, \\sigma) = \\frac{1}
        {1+\\exp\\left(-\\frac{x-\\mu}{\\sigma} \\right)}

    :param x: value at which the CDF is evaluated.
    :type x: float or biogeme.expression
    :param mu: location parameter :math:`\\mu` of the logistic distribution.
        Default: 0.
    :type mu: float or biogeme.expression
    :param s: scale parameter :math:`\\sigma` of the logistic distribution.
        Default: 1.
    :type s: float or biogeme.expression
    :note: It is assumed that :math:`\\sigma > 0`, but it is
        not verified by the code.

    :return: value of the logistic CDF.
    :rtype: float or biogeme.expression

    """
    result = 1.0 / (1.0 + exp(-(x - mu) / s))
    return result
