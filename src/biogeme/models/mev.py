""" Implements the Multivariate Extreme Value models.

:author: Michel Bierlaire
:date: Wed Oct 25 11:35:34 2023
"""
import logging
from typing import Mapping
from biogeme.expressions import Expression, _bioLogLogitFullChoiceSet, _bioLogLogit, exp

logger = logging.getLogger(__name__)


def logmev(
    V: Mapping[int, Expression],
    log_gi: Mapping[int, Expression],
    av: Mapping[int, Expression],
    choice: Expression,
) -> Expression:
    """Log of the choice probability for a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
            (e^{V_1},\\ldots,e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of the choice probability of the MEV model, given by

    .. math:: V_i + \\ln G_i(e^{V_1},\\ldots,e^{V_J}) -
              \\ln\\left(\\sum_j e^{V_j + \\ln G_j(e^{V_1},
              \\ldots,e^{V_J})}\\right)

    """
    H = {i: v + log_gi[i] for i, v in V.items()}
    if av is None:
        log_p = _bioLogLogitFullChoiceSet(H, av=None, choice=choice)
    else:
        log_p = _bioLogLogit(H, av, choice)
    return log_p


def mev(V, log_gi, av, choice):
    """Choice probability for a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)


    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
              (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type log_gi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)

    :param choice: id of the alternative for which the probability must be
              calculated.
    :type choice: biogeme.expressions.expr.Expression

    :return: Choice probability of the MEV model, given by

    .. math:: \\frac{e^{V_i + \\ln G_i(e^{V_1},
              \\ldots,e^{V_J})}}{\\sum_j e^{V_j +
              \\ln G_j(e^{V_1},\\ldots,e^{V_J})}}

    :rtype: biogeme.expressions.expr.Expression
    """
    return exp(logmev(V, log_gi, av, choice))


def logmev_endogenousSampling(V, log_gi, av, correction, choice):
    """Log of choice probability for a MEV model, including the
    correction for endogenous sampling as proposed by `Bierlaire, Bolduc
    and McFadden (2008)`_.

    .. _`Bierlaire, Bolduc and McFadden (2008)`:
       http://dx.doi.org/10.1016/j.trb.2007.09.003

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
                  (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type log_gi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)


    :param correction: a dict of expressions for the correstion terms
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
    H = {i: v + log_gi[i] + correction[i] for i, v in V.items()}
    log_p = _bioLogLogit(H, av, choice)
    return log_p


def mev_endogenousSampling(V, log_gi, av, correction, choice):
    """Choice probability for a MEV model, including the correction
    for endogenous sampling as proposed by
    `Bierlaire, Bolduc and McFadden (2008)`_.

    .. _`Bierlaire, Bolduc and McFadden (2008)`:
           http://dx.doi.org/10.1016/j.trb.2007.09.003

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :type V: dict(int:biogeme.expressions.expr.Expression)

    :param log_gi: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}
              (e^{V_1}, \\ldots, e^{V_J})

        where :math:`G` is the MEV generating function. If an alternative
        :math:`i` is not available, then :math:`G_i = 0`.

    :type log_gi: dict(int:biogeme.expressions.expr.Expression)

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :type av: dict(int:biogeme.expressions.expr.Expression)


    :param correction: a dict of expressions for the correstion terms
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
    return exp(logmev_endogenousSampling(V, log_gi, av, correction, choice))
