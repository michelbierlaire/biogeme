"""Implements the cross-nested logit model.

:author: Michel Bierlaire
:date: Wed Oct 25 11:08:59 2023
"""

import logging

from biogeme.deprecated import deprecated
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Expression,
    ExpressionOrNumeric,
    MultipleSum,
    exp,
    log,
    logzero,
)
from biogeme.nests import NestsForCrossNestedLogit, OldNestsForCrossNestedLogit

from .mev import logmev

logger = logging.getLogger(__name__)


def cnl(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Implements the cross-nested logit model as a MEV model.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: choice probability for the cross-nested logit model.
    """
    return exp(logcnl(util, availability, nests, choice))


@deprecated(cnl)
def cnl_avail(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric],
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Same as cnl. Maintained for backward compatibility"""
    pass


@deprecated(cnl)
def logcnl_avail(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Same as logcnl. Maintained for backward compatibility

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of choice probability for the cross-nested logit model.
    """
    pass


def get_mev_for_cross_nested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
) -> dict[int, Expression]:
    """Implements the MEV generating function for the cross nested logit
    model as a MEV model.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
        alternative, indexed
        by numerical ids. Must be consistent with util, or
        None. In this case, all alternatives are supposed to be
        always available.

    :param nests: object describing the nesting structure

    :return: log of the choice probability for the cross-nested logit model.
    """
    if not isinstance(nests, NestsForCrossNestedLogit):
        logger.warning(
            'It is recommended to define the nests of the cross-nested logit model using '
            'the objects OneNestForNestedLogit and NestsForCrossNestedLogit defined '
            'in biogeme.nests.'
        )
        nests = NestsForCrossNestedLogit(choice_set=list(util), tuple_of_nests=nests)

    ok, message = nests.check_validity()
    if not ok:
        raise BiogemeError(message)

    gi_terms: dict[int, Expression] = {}
    if nests.alone is None:
        log_gi = {}
        for i in util:
            gi_terms[i] = []
    else:
        log_gi = {i: 0 for i in nests.alone}
        for i in set(util).difference(set(nests.alone)):
            gi_terms[i] = []
    for m in nests:
        if availability is None:
            biosum = MultipleSum(
                [
                    a**m.nest_param * exp(m.nest_param * (util[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        else:
            biosum = MultipleSum(
                [
                    availability[i] * a**m.nest_param * exp(m.nest_param * (util[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        for i, a in m.dict_of_alpha.items():
            gi_terms[i] += [
                a**m.nest_param
                * exp((m.nest_param - 1) * (util[i]))
                * biosum ** ((1.0 - m.nest_param) / m.nest_param)
            ]
    for k, G in gi_terms.items():
        log_gi[k] = logzero(MultipleSum(G))
    return log_gi


@deprecated(new_func=get_mev_for_cross_nested)
def getMevForCrossNested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
) -> Expression:
    pass


def logcnl(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Implements the log of the cross-nested logit model as a MEV model.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure
    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of the choice probability for the cross-nested logit model.

    :raise BiogemeError: if the definition of the nests is invalid.
    """

    log_gi = get_mev_for_cross_nested(util, availability, nests)
    log_p = logmev(util, log_gi, availability, choice)
    return log_p


def cnlmu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric,
) -> Expression:
    """Implements the cross-nested logit model as a MEV model with
    the homogeneity parameters is explicitly involved

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :param mu: Homogeneity parameter :math:`\\mu`.

    :return: choice probability for the cross-nested logit model.
    """
    return exp(logcnlmu(util, availability, nests, choice, mu))


def get_mev_for_cross_nested_mu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    mu: ExpressionOrNumeric,
) -> dict[int, Expression]:
    """Implements the MEV generating function for the cross-nested logit
    model as a MEV model with the homogeneity parameters is explicitly
    involved.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param mu: Homogeneity parameter :math:`\\mu`.

    :return: log of the choice probability for the cross-nested logit model.

    """

    if not isinstance(nests, NestsForCrossNestedLogit):
        logger.warning(
            'It is recommended to define the nests of the cross-nested logit model using '
            'the objects OneNestForNestedLogit and NestsForCrossNestedLogit defined '
            'in biogeme.nests.'
        )
        nests = NestsForCrossNestedLogit(choice_set=list(util), tuple_of_nests=nests)

    ok, message = nests.check_validity()
    if not ok:
        raise BiogemeError(message)

    gi_terms = {}
    if nests.alone is None:
        log_gi = {}
        for i in util:
            gi_terms[i] = []
    else:
        log_gi = {i: log(mu) + (mu - 1) * util[i] for i in nests.alone}
        for i in set(util).difference(set(nests.alone)):
            gi_terms[i] = []
    for m in nests:
        if availability is None:
            biosum = MultipleSum(
                [
                    a ** (m.nest_param / mu) * exp(m.nest_param * (util[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        else:
            biosum = MultipleSum(
                [
                    availability[i]
                    * a ** (m.nest_param / mu)
                    * exp(m.nest_param * (util[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        for i, a in m.dict_of_alpha.items():
            gi_terms[i] += [
                a ** (m.nest_param / mu)
                * exp((m.nest_param - 1) * (util[i]))
                * biosum ** ((mu / m.nest_param) - 1.0)
            ]
    for k, G in gi_terms.items():
        log_gi[k] = log(mu * MultipleSum(G))
    return log_gi


@deprecated(get_mev_for_cross_nested_mu)
def getMevForCrossNestedMu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    mu: ExpressionOrNumeric,
) -> dict[int, Expression]:
    pass


def logcnlmu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric,
) -> Expression:
    """Implements the log of the cross-nested logit model as a MEV model
    with the homogeneity parameters is explicitly involved.


    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with util, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :param mu: Homogeneity parameter :math:`\\mu`.

    :return: log of the choice probability for the cross-nested logit model.

    :raise BiogemeError: if the definition of the nests is invalid.

    """
    log_gi = get_mev_for_cross_nested_mu(util, availability, nests, mu)
    log_p = logmev(util, log_gi, availability, choice)
    return log_p
