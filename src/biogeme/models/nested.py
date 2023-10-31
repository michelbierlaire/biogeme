""" Implements various models.

:author: Michel Bierlaire
:date: Fri Mar 29 17:13:14 2019
"""
import logging
from typing import Mapping
import biogeme.exceptions as excep
from biogeme.models import logmev, mev
from biogeme.expressions import (
    Expression,
    exp,
    log,
    bioMultSum,
    Numeric,
    ConditionalTermTuple,
    ConditionalSum,
)
from biogeme.nests import NestsForNestedLogit

logger = logging.getLogger(__name__)


def getMevGeneratingForNested(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForNestedLogit,
) -> Expression:
    """Implements the  MEV generating function for the nested logit model

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: an object describing the nests

    :return: a dictionary mapping each alternative id with the function

    .. math:: G(e^{V_1},
              \\ldots,e^{V_J}) =  \\sum_m \\left( \\sum_{\\ell \\in C_m}
              y_\\ell^{\\mu_m}\\right)^{\\frac{\\mu}{\\mu_m}}

    where :math:`G` is the MEV generating function.

    """
    if not isinstance(nests, NestsForNestedLogit):
        logger.warning(
            'It is recommended to define the nests of the nested logit model using '
            'the objects OneNestForNestedLogit and NestsForNestedLogit defined '
            'in biogeme.nests.'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=nests)

    ok, message = nests.check_partition()
    if not ok:
        raise excep.BiogemeError(message)

    terms_for_nests = []
    for m in nests:
        if availability is None:
            sum_terms = [exp(m.nest_param * V[i]) for i in m.list_of_alternatives]
            the_sum = bioMultSum(sum_terms)
        else:
            sum_terms = [
                ConditionalTermTuple(
                    condition=availability[i] != Numeric(0),
                    term=exp(m.nest_param * V[i]),
                )
                for i in m.list_of_alternatives
            ]
            the_sum = ConditionalSum(list_of_terms=sum_terms)
        terms_for_nests.append(the_sum ** (1.0 / m.nest_param))
    if nests.alone is not None:
        for i in nests.alone:
            terms_for_nests.append(V[i])
    return bioMultSum(terms_for_nests)


def getMevForNested(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForNestedLogit,
) -> Mapping[int, Expression]:
    """Implements the derivatives of MEV generating function for the
    nested logit model

    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object containing the description of the nests.

    :return: a dictionary mapping each alternative id with the function

        .. math:: \\ln \\frac{\\partial G}{\\partial y_i}(e^{V_1},
              \\ldots,e^{V_J}) = e^{(\\mu_m-1)V_i}
              \\left(\\sum_{i=1}^{J_m} e^{\\mu_m V_i}\\right)^
              {\\frac{1}{\\mu_m}-1}

        where :math:`m` is the (only) nest containing alternative :math:`i`,
        and :math:`G` is the MEV generating function.

    """
    if not isinstance(nests, NestsForNestedLogit):
        logger.warning(
            'It is recommended to define the nests of the nested logit model using '
            'the objects OneNestForNestedLogit and NestsForNestedLogit defined '
            'in biogeme.nests.'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=nests)

    ok, message = nests.check_partition()
    if not ok:
        raise excep.BiogemeError(message)

    if nests.alone is None:
        log_gi = {}
    else:
        log_gi = {i: Numeric(0) for i in nests.alone}
    for m in nests:
        if availability is None:
            sum_terms = [exp(m.nest_param * V[i]) for i in m.list_of_alternatives]
            the_sum = bioMultSum(sum_terms)
        else:
            sum_terms = [
                ConditionalTermTuple(
                    condition=availability[i] != Numeric(0),
                    term=exp(m.nest_param * V[i]),
                )
                for i in m.list_of_alternatives
            ]
            the_sum = ConditionalSum(list_of_terms=sum_terms)

        for i in m.list_of_alternatives:
            log_gi[i] = (m.nest_param - 1.0) * V[i] + (1.0 / m.nest_param - 1.0) * log(
                the_sum
            )
    return log_gi


def getMevForNestedMu(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForNestedLogit,
    mu: Expression,
) -> Mapping[int, Expression]:
    """Implements the MEV generating function for the nested logit model,
    including the scale parameter

    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability
        of each alternative, indexed
        by numerical ids. Must be consistent with V, or
        None. In this case, all alternatives are supposed to be
        always available.

    :param nests: object describing the nesting structure

    :param mu: scale parameter

    :return: a dictionary mapping each alternative id with the function

        .. math:: \\frac{\\partial G}{\\partial y_i}(e^{V_1},\\ldots,e^{V_J}) =
                  \\mu e^{(\\mu_m-1)V_i} \\left(\\sum_{i=1}^{J_m}
                  e^{\\mu_m V_i}\\right)^{\\frac{\\mu}{\\mu_m}-1}

        where :math:`m` is the (only) nest containing alternative :math:`i`,
        and :math:`G` is the MEV generating function.

    """
    if not isinstance(nests, NestsForNestedLogit):
        logger.warning(
            'It is recommended to define the nests of the nested logit model using '
            'the objects OneNestForNestedLogit and NestsForNestedLogit defined '
            'in biogeme.nests.'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=nests)

    ok, message = nests.check_partition()
    if not ok:
        raise excep.BiogemeError(message)

    if nests.alone is None:
        log_gi = {}
    else:
        log_gi = {i: log(mu) + (mu - 1) * V[i] for i in nests.alone}
    for m in nests:
        if availability is None:
            sum_terms = [exp(m.nest_param * V[i]) for i in m.list_of_alternatives]
            the_sum = bioMultSum(sum_terms)

        else:
            sum_terms = [
                ConditionalTermTuple(
                    condition=availability[i] != Numeric(0),
                    term=exp(m.nest_param * V[i]),
                )
                for i in m.list_of_alternatives
            ]
            the_sum = ConditionalSum(list_of_terms=sum_terms)
        for i in m.list_of_alternatives:
            log_gi[i] = (
                log(mu)
                + (m.nest_param - 1.0) * V[i]
                + (mu / m.nest_param - 1.0) * log(the_sum)
            )
    return log_gi


def nested(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForNestedLogit,
    choice: Expression,
) -> Expression:
    """Implements the nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability
                         of each alternative, indexed by numerical
                         ids. Must be consistent with V, or None. In
                         this case, all alternatives are supposed to
                         be always available.

    :param nests: object containing the description of the nests.

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: choice probability for the nested logit model,
             based on the derivatives of the MEV generating function produced
             by the function getMevForNested

    :raise BiogemeError: if the definition of the nests is invalid.
    """

    log_gi = getMevForNested(V, availability, nests)
    P = mev(V, log_gi, availability, choice)
    return P


def lognested(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForNestedLogit,
    choice: Expression,
) -> Expression:
    """Implements the log of a nested logit model as a MEV model.

    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
        alternative (:math:`a_i` in the above formula), indexed
        by numerical ids. Must be consistent with V, or
        None. In this case, all alternatives are supposed to be
        always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of choice probability for the nested logit model,
             based on the derivatives of the MEV generating function produced
             by the function getMevForNested

    :raise BiogemeError: if the definition of the nests is invalid.
    """
    if not isinstance(nests, NestsForNestedLogit):
        logger.warning(
            'It is recommended to define the nests of the nested logit model using '
            'the objects OneNestForNestedLogit and NestsForNestedLogit defined '
            'in biogeme.nests.'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=nests)

    ok, message = nests.check_partition()
    if not ok:
        raise excep.BiogemeError(message)
    log_gi = getMevForNested(
        V,
        availability,
        nests,
    )
    log_p = logmev(V, log_gi, availability, choice)
    return log_p


def nestedMevMu(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForNestedLogit,
    choice: Expression,
    mu: Expression,
) -> Expression:
    """Implements the nested logit model as a MEV model, where mu is also
    a parameter, if the user wants to test different normalization
    schemes.

    :param V: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :param mu: expression producing the value of the top-level scale parameter.

    :return: the nested logit choice probability based on the following
             derivatives of the MEV generating function:

    .. math:: \\frac{\\partial G}{\\partial y_i}(e^{V_1},\\ldots,e^{V_J}) =
              \\mu e^{(\\mu_m-1)V_i} \\left(\\sum_{i=1}^{J_m}
              e^{\\mu_m V_i}\\right)^{\\frac{\\mu}{\\mu_m}-1}

    Where :math:`m` is the (only) nest containing alternative :math:`i`, and
    :math:`G` is the MEV generating function.


    """
    return exp(lognestedMevMu(V, availability, nests, choice, mu))


def lognestedMevMu(
    V: Mapping[int, Expression],
    availability: Mapping[int, Expression],
    nests: NestsForNestedLogit,
    choice: Expression,
    mu: Expression,
) -> Expression:
    """Implements the log of the nested logit model as a MEV model, where
    mu is also a parameter, if the user wants to test different
    normalization schemes.


    :param V: dict of objects representing the utility functions of
        each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative (:math:`a_i` in the above formula), indexed
               by numerical ids. Must be consistent with V, or
               None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure

    :param choice: id of the alternative for which the probability must be
              calculated.

    :param mu: expression producing the value of the top-level scale parameter.

    :return: the log of the nested logit choice probability based on the
        following derivatives of the MEV generating function:

        .. math:: \\frac{\\partial G}{\\partial y_i}(e^{V_1},\\ldots,e^{V_J}) =
                  \\mu e^{(\\mu_m-1)V_i} \\left(\\sum_{i=1}^{J_m}
                  e^{\\mu_m V_i}\\right)^{\\frac{\\mu}{\\mu_m}-1}

        where :math:`m` is the (only) nest containing alternative :math:`i`,
        and :math:`G` is the MEV generating function.


    """

    log_gi = getMevForNestedMu(V, availability, nests, mu)
    log_p = logmev(V, log_gi, availability, choice)
    return log_p
