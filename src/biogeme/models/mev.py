"""Implements the Multivariate Extreme Value models.

:author: Michel Bierlaire
:date: Wed Oct 25 11:35:34 2023
"""

import logging

from biogeme.deprecated import deprecated
from biogeme.expressions import Expression, ExpressionOrNumeric, LogLogit, exp

logger = logging.getLogger(__name__)


def logmev(
    util: dict[int, ExpressionOrNumeric],
    log_gi: dict[int, ExpressionOrNumeric],
    av: dict[int, ExpressionOrNumeric],
    choice: ExpressionOrNumeric,
) -> Expression:
    """Log of the choice probability for a MEV model.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
            (e^{V_1},\\ldots,e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of the choice probability of the MEV model, given by

    .. math:: V_i + \\ln G_i(e^{V_1},\\ldots,e^{V_J}) -
              \\ln\\left(\\sum_j e^{V_j + \\ln G_j(e^{V_1},
              \\ldots,e^{V_J})}\\right)

    """
    h = {i: v + log_gi[i] for i, v in util.items()}
    log_p = LogLogit(h, av, choice=choice)
    log_p._is_complex = True
    return log_p


def mev(
    util: dict[int, ExpressionOrNumeric],
    log_gi: dict[int, Expression],
    av: dict[int, ExpressionOrNumeric] | None,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Choice probability for a MEV model.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type util: dict(int:biogeme.expressions.expr.Expression)


    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
              (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type log_gi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: Choice probability of the   MEV model, given by

    .. math:: \\frac{e^{V_i + \\ln G_i(e^{V_1},
              \\ldots,e^{V_J})}}{\\sum_j e^{V_j +
              \\ln G_j(e^{V_1},\\ldots,e^{V_J})}}

    :rtype: biogeme.expressions.expr.Expression
    """
    return exp(logmev(util, log_gi, av, choice))


def logmev_endogenous_sampling(
    util: dict[int, ExpressionOrNumeric],
    log_gi: dict[int, Expression],
    av: dict[int, ExpressionOrNumeric] | None,
    correction: dict[int, ExpressionOrNumeric],
    choice: ExpressionOrNumeric,
) -> Expression:
    """Log of choice probability for a MEV model, including the
    correction for endogenous sampling as proposed by `Bierlaire, Bolduc
    and McFadden (2008)`_.

    .. _`Bierlaire, Bolduc and McFadden (2008)`:
       http://dx.doi.org/10.1016/j.trb.2007.09.003

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type util: dict(int:biogeme.expressions.expr.Expression)

    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
                  (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type log_gi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)


    :param correction: a dict of expressions for the correction terms
                       of each alternative.
    :type correction: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of the choice probability of the MEV model, given by

    .. math:: V_i + \\ln G_i(e^{V_1}, \\ldots,e^{V_J}) + \\omega_i -
              \\ln\\left(\\sum_j e^{V_j +
              \\ln G_j(e^{V_1}, \\ldots, e^{V_J})+ \\omega_j}\\right)

    where :math:`\\omega_i` is the correction term for alternative :math:`i`.

    :rtype: biogeme.expressions.expr.Expression

    """
    h = {i: v + log_gi[i] + correction[i] for i, v in util.items()}
    log_p = LogLogit(h, av, choice)
    log_p._is_complex = True
    return log_p


@deprecated(new_func=logmev_endogenous_sampling)
def logmev_endogenousSampling(
    util: dict[int, ExpressionOrNumeric],
    log_gi: dict[int, Expression],
    av: dict[int, ExpressionOrNumeric] | None,
    correction: dict[int, ExpressionOrNumeric],
    choice: ExpressionOrNumeric,
):
    pass


def mev_endogenous_sampling(
    util: dict[int, ExpressionOrNumeric],
    log_gi: dict[int, Expression],
    av: dict[int, ExpressionOrNumeric] | None,
    correction: dict[int, ExpressionOrNumeric],
    choice: ExpressionOrNumeric,
):
    """Choice probability for a MEV model, including the correction
    for endogenous sampling as proposed by
    `Bierlaire, Bolduc and McFadden (2008)`_.

    .. _`Bierlaire, Bolduc and McFadden (2008)`:
           http://dx.doi.org/10.1016/j.trb.2007.09.003

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type util: dict(int:biogeme.expressions.expr.Expression)

    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
              (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type log_gi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)


    :param correction: a dict of expressions for the correction terms
                       of each alternative.
    :type correction: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: log of the choice probability of the MEV model, given by

    .. math:: V_i + \\ln G_i(e^{V_1}, \\ldots, e^{V_J}) + \\omega_i -
              \\ln\\left(\\sum_j e^{V_j + \\ln G_j(e^{V_1},\\ldots,e^{V_J})+
              \\omega_j}\\right)

    where :math:`\\omega_i` is the correction term for alternative :math:`i`.

    :rtype: biogeme.expressions.expr.Expression

    """
    the_expression = exp(
        logmev_endogenous_sampling(util, log_gi, av, correction, choice)
    )
    the_expression._is_complex = True
    return the_expression


@deprecated(new_func=mev_endogenous_sampling)
def mev_endogenousSampling(
    util: dict[int, ExpressionOrNumeric],
    log_gi: dict[int, Expression],
    av: dict[int, ExpressionOrNumeric] | None,
    correction: dict[int, ExpressionOrNumeric],
    choice: ExpressionOrNumeric,
):
    pass
