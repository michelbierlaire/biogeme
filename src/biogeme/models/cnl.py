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
    LogCrossNested,
    MultipleSum,
    SparseLogCrossNested,
    exp,
    log,
    logzero,
)
from biogeme.nests import NestsForCrossNestedLogit, OldNestsForCrossNestedLogit

logger = logging.getLogger(__name__)


def cnl(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Implements the cross-nested logit model.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed by numerical ids. Must be consistent with
               util, or None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure.

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: choice probability for the cross-nested logit model.
    """
    return exp(logcnl(util, availability, nests, choice))


def sparse_cnl(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric | None = None,
) -> Expression:
    """Return a CNL probability using structurally sparse memberships.

    Literal zero allocation parameters are omitted from the JAX calculation.
    Parameter-dependent allocations remain active. The optional ``mu`` selects
    the explicit-homogeneity formulation.
    """
    return exp(log_sparse_cnl(util, availability, nests, choice, mu=mu))


@deprecated(cnl)
def cnl_avail(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric],
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Same as cnl. Maintained for backward compatibility."""
    return cnl(util, availability, nests, choice)


@deprecated(cnl)
def logcnl_avail(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Same as logcnl. Maintained for backward compatibility.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed by numerical ids. Must be consistent with
               util, or None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure.

    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of choice probability for the cross-nested logit model.
    """
    return logcnl(util, availability, nests, choice)


def _normalize_and_check_nests(
    util: dict[int, ExpressionOrNumeric],
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
) -> NestsForCrossNestedLogit:
    """Convert old nest syntax if needed and check validity."""
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

    return nests


def get_mev_for_cross_nested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
) -> dict[int, Expression]:
    """Implements the MEV derivative terms for the cross-nested logit model.

    This function is kept for backward compatibility and for users who build
    generic MEV expressions explicitly. The public ``logcnl`` function now uses
    the dedicated ``LogCrossNested`` expression for better backend performance.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
        alternative, indexed by numerical ids. Must be consistent with util, or
        None. In this case, all alternatives are supposed to be always available.

    :param nests: object describing the nesting structure.

    :return: dictionary mapping each alternative to
        :math:`\\log G_i(e^{V_1},\\ldots,e^{V_J})`.
    """
    nests = _normalize_and_check_nests(util, nests)

    gi_terms: dict[int, list[Expression]] = {}
    if nests.alone is None:
        log_gi = {}
        for i in util:
            gi_terms[i] = []
    else:
        log_gi = {i: 0 for i in nests.alone}
        for i in set(util).difference(set(nests.alone)):
            gi_terms[i] = []

    for nest in nests:
        if availability is None:
            biosum = MultipleSum(
                [
                    alpha**nest.nest_param * exp(nest.nest_param * util[i])
                    for i, alpha in nest.dict_of_alpha.items()
                ]
            )
        else:
            biosum = MultipleSum(
                [
                    availability[i]
                    * alpha**nest.nest_param
                    * exp(nest.nest_param * util[i])
                    for i, alpha in nest.dict_of_alpha.items()
                ]
            )

        for i, alpha in nest.dict_of_alpha.items():
            gi_terms[i] += [
                alpha**nest.nest_param
                * exp((nest.nest_param - 1) * util[i])
                * biosum ** ((1.0 - nest.nest_param) / nest.nest_param)
            ]

    for k, G in gi_terms.items():
        log_gi[k] = logzero(MultipleSum(G))

    return log_gi


@deprecated(new_func=get_mev_for_cross_nested)
def getMevForCrossNested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
) -> dict[int, Expression]:
    """Deprecated name for get_mev_for_cross_nested."""
    return get_mev_for_cross_nested(util, availability, nests)


def logcnl(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Implements the log of the cross-nested logit model.

    The implementation uses the dedicated ``LogCrossNested`` expression, which
    preserves the CNL structure and allows efficient backend-specific code.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed by numerical ids. Must be consistent with
               util, or None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure.
    :param choice: id of the alternative for which the probability must be
              calculated.

    :return: log of the choice probability for the cross-nested logit model.

    :raise BiogemeError: if the definition of the nests is invalid.
    """
    return LogCrossNested(
        util=util,
        av=availability,
        nests=nests,
        choice=choice,
    )


def log_sparse_cnl(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric | None = None,
) -> Expression:
    """Return a CNL log probability using structurally sparse memberships.

    Only literal zero allocation parameters are treated as inactive. Any
    expression involving a parameter is retained, even if its current value is
    zero. The optional ``mu`` selects the explicit-homogeneity formulation.
    """
    return SparseLogCrossNested(
        util=util,
        av=availability,
        nests=nests,
        choice=choice,
        mu=mu,
    )


def cnlmu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric,
) -> Expression:
    """Implements the cross-nested logit model with explicit homogeneity.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed by numerical ids. Must be consistent with
               util, or None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure.

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
    """Implements the MEV derivative terms for explicit-mu CNL.

    This function is kept for backward compatibility and for users who build
    generic MEV expressions explicitly. The public ``logcnlmu`` function now
    uses the dedicated ``LogCrossNested`` expression with its optional ``mu``
    parameter for better backend performance.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed by numerical ids. Must be consistent with
               util, or None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure.

    :param mu: Homogeneity parameter :math:`\\mu`.

    :return: dictionary mapping each alternative to
        :math:`\\log G_i(e^{V_1},\\ldots,e^{V_J})`.
    """
    nests = _normalize_and_check_nests(util, nests)

    gi_terms: dict[int, list[Expression]] = {}
    if nests.alone is None:
        log_gi = {}
        for i in util:
            gi_terms[i] = []
    else:
        log_gi = {i: log(mu) + (mu - 1) * util[i] for i in nests.alone}
        for i in set(util).difference(set(nests.alone)):
            gi_terms[i] = []

    for nest in nests:
        if availability is None:
            biosum = MultipleSum(
                [
                    alpha ** (nest.nest_param / mu) * exp(nest.nest_param * util[i])
                    for i, alpha in nest.dict_of_alpha.items()
                ]
            )
        else:
            biosum = MultipleSum(
                [
                    availability[i]
                    * alpha ** (nest.nest_param / mu)
                    * exp(nest.nest_param * util[i])
                    for i, alpha in nest.dict_of_alpha.items()
                ]
            )

        for i, alpha in nest.dict_of_alpha.items():
            gi_terms[i] += [
                alpha ** (nest.nest_param / mu)
                * exp((nest.nest_param - 1) * util[i])
                * biosum ** ((mu / nest.nest_param) - 1.0)
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
    """Deprecated name for get_mev_for_cross_nested_mu."""
    return get_mev_for_cross_nested_mu(util, availability, nests, mu)


def logcnlmu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForCrossNestedLogit | OldNestsForCrossNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric,
) -> Expression:
    """Implements the log of the explicit-mu cross-nested logit model.

    The implementation uses the dedicated ``LogCrossNested`` expression with
    its optional global homogeneity parameter ``mu``.

    :param util: dict of objects representing the utility functions of
              each alternative, indexed by numerical ids.

    :param availability: dict of objects representing the availability of each
               alternative, indexed by numerical ids. Must be consistent with
               util, or None. In this case, all alternatives are supposed to be
               always available.

    :param nests: object describing the nesting structure.

    :param choice: id of the alternative for which the probability must be
              calculated.

    :param mu: Homogeneity parameter :math:`\\mu`.

    :return: log of the choice probability for the cross-nested logit model.

    :raise BiogemeError: if the definition of the nests is invalid.
    """
    return LogCrossNested(
        util=util,
        av=availability,
        nests=nests,
        choice=choice,
        mu=mu,
    )
