""" Implements the logit model.

:author: Michel Bierlaire
:date: Wed Oct 25 08:43:26 2023
"""
import logging
from typing import Optional
from biogeme.expressions import Expression, _bioLogLogit, _bioLogLogitFullChoiceSet, exp

logger = logging.getLogger(__name__)


def loglogit(
    V: dict[int, Expression], av: Optional[dict[int, Expression]], i: int
) -> Expression:
    """The logarithm of the logit model

    The model is defined as

    .. math:: \\frac{a_i e^{V_i}}{\\sum_{i=1}^J a_j e^{V_j}}

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param i: id of the alternative for which the probability must be
              calculated.

    :return: choice probability of alternative number i.
    """

    if av is None:
        return _bioLogLogitFullChoiceSet(V, av=None, choice=i)

    return _bioLogLogit(V, av, i)


def logit(
    V: dict[int, Expression], av: Optional[dict[int, Expression]], i: int
) -> Expression:
    """The logit model

    The model is defined as

    .. math:: \\frac{a_i e^{V_i}}{\\sum_{i=1}^J a_j e^{V_j}}

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param av: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param i: id of the alternative for which the probability must be
              calculated.

    :return: choice probability of alternative number i.

    """
    if av is None:
        return exp(_bioLogLogitFullChoiceSet(V, av=None, choice=i))

    return exp(_bioLogLogit(V, av, i))
