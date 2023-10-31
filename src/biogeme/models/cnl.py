""" Implements the cross-nested logit model.

:author: Michel Bierlaire
:date: Wed Oct 25 11:08:59 2023
"""
import logging
import warnings
from typing import Mapping
import biogeme.exceptions as excep
from biogeme.expressions import Expression, exp, bioMultSum, logzero, log
from biogeme.nests import NestsForCrossNestedLogit
from biogeme.models import logmev

logger = logging.getLogger(__name__)


def cnl_avail(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
    choice: Expression,
) -> Expression:
    """Same as cnl. Maintained for backward compatibility"""
    warnings.warn(
        'The function cnl_avail is deprecated. It has been replaced by the function cnl',
        DeprecationWarning,
        stacklevel=2,
    )
    return cnl(V, availability, nests, choice)


def cnl(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
    choice: Expression,
) -> Expression:
    """Implements the cross-nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: choice probability for the cross-nested logit model.
    """
    return exp(logcnl(V, availability, nests, choice))


def logcnl_avail(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
    choice: Expression,
) -> Expression:
    """Same as logcnl. Maintained for backward compatibility

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of choice probability for the cross-nested logit model.
    """
    warnings.warn(
        'The function logcnl_avail is deprecated. It has been replaced by the function logcnl',
        DeprecationWarning,
        stacklevel=2,
    )
    return logcnl(V, availability, nests, choice)


def getMevForCrossNested(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
) -> Expression:
    """Implements the MEV generating function for the cross nested logit
    model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
        alternative, indexed
        by numerical ids. Must be consistent with V, or
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
        nests = NestsForCrossNestedLogit(choice_set=list(V), tuple_of_nests=nests)

    ok, message = nests.check_validity()
    if not ok:
        raise excep.BiogemeError(message)

    gi_terms: Mapping[int, Expression] = {}
    if nests.alone is None:
        log_gi = {}
        for i in V:
            gi_terms[i] = []
    else:
        log_gi = {i: 0 for i in nests.alone}
        for i in set(V).difference(set(nests.alone)):
            gi_terms[i] = []
    biosum = {}
    for m in nests:
        if availability is None:
            biosum = bioMultSum(
                [
                    a ** (m.nest_param) * exp(m.nest_param * (V[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        else:
            biosum = bioMultSum(
                [
                    availability[i] * a ** (m.nest_param) * exp(m.nest_param * (V[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        for i, a in m.dict_of_alpha.items():
            gi_terms[i] += [
                a ** (m.nest_param)
                * exp((m.nest_param - 1) * (V[i]))
                * biosum ** ((1.0 / m.nest_param) - 1.0)
            ]
    for k, G in gi_terms.items():
        log_gi[k] = logzero(bioMultSum(G))
    return log_gi


def logcnl(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
    choice: Expression,
) -> Expression:
    """Implements the log of the cross-nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure
    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of the choice probability for the cross-nested logit model.

    :raise BiogemeError: if the definition of the nests is invalid.
    """

    log_gi = getMevForCrossNested(V, availability, nests)
    log_p = logmev(V, log_gi, availability, choice)
    return log_p


def cnlmu(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
    choice: Expression,
    mu: Expression,
) -> Expression:
    """Implements the cross-nested logit model as a MEV model with
    the homogeneity parameters is explicitly involved

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :param mu: Homogeneity parameter :math:`\\mu`.

    :return: choice probability for the cross-nested logit model.
    """
    return exp(logcnlmu(V, availability, nests, choice, mu))


def getMevForCrossNestedMu(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
    mu: Expression,
) -> Expression:
    """Implements the MEV generating function for the cross-nested logit
    model as a MEV model with the homogeneity parameters is explicitly
    involved.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
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
        nests = NestsForCrossNestedLogit(choice_set=list(V), tuple_of_nests=nests)

    ok, message = nests.check_validity()
    if not ok:
        raise excep.BiogemeError(message)

    gi_terms = {}
    if nests.alone is None:
        log_gi = {}
        for i in V:
            gi_terms[i] = []
    else:
        log_gi = {i: log(mu) + (mu - 1) * V[i] for i in nests.alone}
        for i in set(V).difference(set(nests.alone)):
            gi_terms[i] = []
    biosum = {}
    for m in nests:
        if availability is None:
            biosum = bioMultSum(
                [
                    a ** (m.nest_param / mu) * exp(m.nest_param * (V[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        else:
            biosum = bioMultSum(
                [
                    availability[i]
                    * a ** (m.nest_param / mu)
                    * exp(m.nest_param * (V[i]))
                    for i, a in m.dict_of_alpha.items()
                ]
            )
        for i, a in m.dict_of_alpha.items():
            gi_terms[i] += [
                a ** (m.nest_param / mu)
                * exp((m.nest_param - 1) * (V[i]))
                * biosum ** ((mu / m.nest_param) - 1.0)
            ]
    for k, G in gi_terms.items():
        log_gi[k] = log(mu * bioMultSum(G))
    return log_gi


def logcnlmu(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForCrossNestedLogit,
    choice: Expression,
    mu: Expression,
) -> Expression:
    """Implements the log of the cross-nested logit model as a MEV model
    with the homogeneity parameters is explicitly involved.


    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :param mu: Homogeneity parameter :math:`\\mu`.

    :return: log of the choice probability for the cross-nested logit model.

    :raise BiogemeError: if the definition of the nests is invalid.

    """
    log_gi = getMevForCrossNestedMu(V, availability, nests, mu)
    log_p = logmev(V, log_gi, availability, choice)
    return log_p
