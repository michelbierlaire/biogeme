""" Implements the logit model.

:author: Michel Bierlaire
:date: Wed Oct 25 08:43:26 2023
"""

import logging

from biogeme.expressions import (
    Expression,
    LogLogit,
    exp,
    ExpressionOrNumeric,
)

logger = logging.getLogger(__name__)


def loglogit(
    util: dict[int, ExpressionOrNumeric],
    av: dict[int, ExpressionOrNumeric] | None,
    i: ExpressionOrNumeric,
) -> Expression:
    """The logarithm of the logit model

    The model is defined as

    .. math:: \\frac{a_i e^{V_i}}{\\sum_{i=1}^J a_j e^{V_j}}

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param i: id of the alternative for which the probability must be
              calculated.

    :return: choice probability of alternative number i.
    """

    return LogLogit(util, av, i)


def logit(
    util: dict[int, ExpressionOrNumeric],
    av: dict[int, ExpressionOrNumeric] | None,
    i: ExpressionOrNumeric,
) -> Expression:
    """The logit model

    The model is defined as

    .. math:: \\frac{a_i e^{V_i}}{\\sum_{i=1}^J a_j e^{V_j}}

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param i: id of the alternative for which the probability must be
              calculated.

    :return: choice probability of alternative number i.

    """
    return exp(LogLogit(util, av, i))
