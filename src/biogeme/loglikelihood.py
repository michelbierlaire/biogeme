""" Functions to calculate the log likelihood

:author: Michel Bierlaire
:date: Fri Mar 29 17:11:44 2019

"""

# Too constraining
# pylint: disable=invalid-name,

from biogeme.expressions import Expression, exp, log, MonteCarlo


def loglikelihood(prob: Expression) -> Expression:
    """
    Simply computes the log of the probability

    :param prob: An expression providing the value of the probability.

    :return: the logarithm of the probability.

    """
    return log(prob)


def mixedloglikelihood(prob: Expression) -> Expression:
    """Compute a simulated loglikelihood function

    :param prob: An expression providing the value of the
                 probability. Although it is not formally necessary,
                 the expression should contain one or more random
                 variables of a given distribution, and therefore
                 is defined as

    .. math:: P(i|\\xi_1,\\ldots,\\xi_L)


    :return: the simulated loglikelihood, given by

        .. math:: \\ln\\left(\\sum_{r=1}^R
            P(i|\\xi^r_1,\\ldots,\\xi^r_L) \\right)

        where :math:`R` is the number of draws, and :math:`\\xi_j^r`
        is the rth draw of the random variable :math:`\\xi_j`.

    """
    ell = MonteCarlo(prob)
    return log(ell)


def likelihoodregression(
    meas: Expression, model: Expression, sigma: Expression
) -> Expression:
    """Computes likelihood function of a regression model.

    :param meas: An expression providing the value :math:`y` of the measure
                 for the current observation.

    :param model: An expression providing the output :math:`m` of the model
                  for the current observation.

    :param sigma: An expression (typically, a parameter) providing the
                  standard error :math:`\\sigma` of the error term.
    :type sigma: biogeme.expressions.Expression
    :return: The likelihood of the regression, assuming a normal distribution,
        that is

        .. math:: \\frac{1}{\\sigma} \\phi\\left( \\frac{y-m}{\\sigma} \\right)

        where :math:`\\phi(\\cdot)` is the pdf of the normal distribution.

    """
    return exp(loglikelihoodregression(meas, model, sigma))


def loglikelihoodregression(
    meas: Expression, model: Expression, sigma: Expression
) -> Expression:
    """Computes log likelihood function of a regression model.

    :param meas: An expression providing the value :math:`y` of the
                 measure for the current observation.

    :param model: An expression providing the output :math:`m` of the
                  model for the current observation.

    :param sigma: An expression (typically, a parameter) providing
                  the standard error :math:`\\sigma` of the error term.

    :return: the likelihood of the regression, assuming a normal distribution,
        that is

    .. math:: -\\left( \\frac{(y-m)^2}{2\\sigma^2} \\right) -
              \\frac{1}{2}\\log(\\sigma^2) - \\frac{1}{2}\\log(2\\pi)

    """
    t = (meas - model) / sigma
    f = -(t**2) / 2 - log(sigma**2) / 2 - 0.9189385332
    return f
