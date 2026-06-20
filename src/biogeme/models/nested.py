"""Simplified nested logit model using the dedicated LogNested expression."""

from __future__ import annotations

import logging

import biogeme.exceptions as excep
from biogeme.deprecated import deprecated
from biogeme.expressions import (
    ConditionalSum,
    ConditionalTermTuple,
    Expression,
    ExpressionOrNumeric,
    LogNested,
    MultipleSum,
    Numeric,
    exp,
    log,
)
from biogeme.nests import NestsForNestedLogit, OldNestsForNestedLogit

logger = logging.getLogger(__name__)


def _normalize_and_check_nests(
    util: dict[int, ExpressionOrNumeric],
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
) -> NestsForNestedLogit:
    """Convert old nest syntax if needed and check partition validity."""
    if not isinstance(nests, NestsForNestedLogit):
        logger.warning(
            'It is recommended to define the nests of the nested logit model using '
            'the objects OneNestForNestedLogit and NestsForNestedLogit defined '
            'in biogeme.nests.'
        )
        nests = NestsForNestedLogit(choice_set=list(util), tuple_of_nests=nests)

    ok, message = nests.check_partition()
    if not ok:
        raise excep.BiogemeError(message)

    return nests


def get_mev_generating_for_nested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
) -> Expression:
    """Implements the MEV generating function for the nested logit model.

    Kept for backward compatibility.
    """
    nests = _normalize_and_check_nests(util, nests)

    terms_for_nests = []
    for nest in nests:
        if availability is None:
            sum_terms = [
                exp(nest.nest_param * util[i]) for i in nest.list_of_alternatives
            ]
            nest_sum = MultipleSum(sum_terms)
        else:
            sum_terms = [
                ConditionalTermTuple(
                    condition=availability[i] != Numeric(0),
                    term=exp(nest.nest_param * util[i]),
                )
                for i in nest.list_of_alternatives
            ]
            nest_sum = ConditionalSum(list_of_terms=sum_terms)

        terms_for_nests.append(nest_sum ** (1.0 / nest.nest_param))

    if nests.alone is not None:
        for i in nests.alone:
            terms_for_nests.append(util[i])

    return MultipleSum(terms_for_nests)


@deprecated(get_mev_generating_for_nested)
def getMevGeneratingForNested(
    util: dict[int, Expression],
    availability: dict[int, Expression],
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
) -> Expression:
    """Deprecated name for get_mev_generating_for_nested."""
    return get_mev_generating_for_nested(util, availability, nests)


def get_mev_for_nested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
) -> dict[int, Expression]:
    """Implements the derivatives of the MEV generating function.

    Kept for backward compatibility.
    """
    nests = _normalize_and_check_nests(util, nests)

    if nests.alone is None:
        log_gi = {}
    else:
        log_gi = {i: Numeric(0) for i in nests.alone}

    for nest in nests:
        if availability is None:
            nest_sum = MultipleSum(
                [exp(nest.nest_param * util[i]) for i in nest.list_of_alternatives]
            )
        else:
            nest_sum = ConditionalSum(
                list_of_terms=[
                    ConditionalTermTuple(
                        condition=availability[i] != Numeric(0),
                        term=exp(nest.nest_param * util[i]),
                    )
                    for i in nest.list_of_alternatives
                ]
            )

        for i in nest.list_of_alternatives:
            log_gi[i] = (nest.nest_param - 1.0) * util[i] + (
                1.0 / nest.nest_param - 1.0
            ) * log(nest_sum)

    return log_gi


@deprecated(get_mev_for_nested)
def getMevForNested(
    V: dict[int, Expression],
    availability: dict[int, Expression] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
) -> dict[int, Expression]:
    """Deprecated name for get_mev_for_nested."""
    return get_mev_for_nested(V, availability, nests)


def get_mev_for_nested_mu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    mu: ExpressionOrNumeric,
) -> dict[int, Expression]:
    """Implements the MEV derivative terms for explicit-mu nested logit.

    Kept for backward compatibility.
    """
    nests = _normalize_and_check_nests(util, nests)

    if nests.alone is None:
        log_gi = {}
    else:
        log_gi = {i: log(mu) + (mu - 1) * util[i] for i in nests.alone}

    for nest in nests:
        if availability is None:
            nest_sum = MultipleSum(
                [exp(nest.nest_param * util[i]) for i in nest.list_of_alternatives]
            )
        else:
            nest_sum = ConditionalSum(
                list_of_terms=[
                    ConditionalTermTuple(
                        condition=availability[i] != Numeric(0),
                        term=exp(nest.nest_param * util[i]),
                    )
                    for i in nest.list_of_alternatives
                ]
            )

        for i in nest.list_of_alternatives:
            log_gi[i] = (
                log(mu)
                + (nest.nest_param - 1.0) * util[i]
                + (mu / nest.nest_param - 1.0) * log(nest_sum)
            )

    return log_gi


@deprecated(get_mev_for_nested_mu)
def getMevForNestedMu(
    util: dict[int, Expression],
    availability: dict[int, Expression] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    mu: Expression,
) -> dict[int, Expression]:
    """Deprecated name for get_mev_for_nested_mu."""
    return get_mev_for_nested_mu(util, availability, nests, mu)


def nested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Choice probability for the nested logit model."""
    return exp(lognested(util, availability, nests, choice))


def lognested(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    choice: ExpressionOrNumeric,
) -> Expression:
    """Log probability for the nested logit model."""
    return LogNested(
        util=util,
        av=availability,
        nests=nests,
        choice=choice,
    )


def nested_mev_mu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric,
) -> Expression:
    """Choice probability for the nested logit model with explicit mu."""
    return exp(lognested_mev_mu(util, availability, nests, choice, mu))


@deprecated(nested_mev_mu)
def nestedMevMu(
    util: dict[int, Expression],
    availability: dict[int, Expression] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    choice: Expression,
    mu: Expression,
) -> Expression:
    """Deprecated name for nested_mev_mu."""
    return nested_mev_mu(util, availability, nests, choice, mu)


def lognested_mev_mu(
    util: dict[int, ExpressionOrNumeric],
    availability: dict[int, ExpressionOrNumeric] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    choice: ExpressionOrNumeric,
    mu: ExpressionOrNumeric,
) -> Expression:
    """Log probability for the nested logit model with explicit mu."""
    return LogNested(
        util=util,
        av=availability,
        nests=nests,
        choice=choice,
        mu=mu,
    )


@deprecated(lognested_mev_mu)
def lognestedMevMu(
    util: dict[int, Expression],
    availability: dict[int, Expression] | None,
    nests: NestsForNestedLogit | OldNestsForNestedLogit,
    choice: Expression,
    mu: Expression,
) -> Expression:
    """Deprecated name for lognested_mev_mu."""
    return lognested_mev_mu(util, availability, nests, choice, mu)
