"""Tests for CNL implementation recommendations and the sparse public API."""

from __future__ import annotations

import logging

from biogeme.expressions import Beta, SparseLogCrossNested
from biogeme.models import log_sparse_cnl, logcnl, sparse_cnl
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit

LOGGER_NAME = 'biogeme.expressions.log_cross_nested'


def build_nests(alternatives: int, nests_count: int, memberships: int):
    allocations: list[dict[int, float]] = [dict() for _ in range(nests_count)]
    for alternative in range(alternatives):
        for offset in range(memberships):
            nest = (alternative + offset) % nests_count
            allocations[nest][alternative + 1] = 1.0 / memberships
    return NestsForCrossNestedLogit(
        choice_set=list(range(1, alternatives + 1)),
        tuple_of_nests=tuple(
            OneNestForCrossNestedLogit(
                nest_param=1.2,
                dict_of_alpha=allocation,
                name=f'nest_{index}',
            )
            for index, allocation in enumerate(allocations)
        ),
    )


def recommendation_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.INFO
    ]


def test_dense_cnl_suggests_sparse_once_at_construction(caplog):
    alternatives = 50
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        expression = logcnl(
            util={i: 0.01 * i for i in range(1, alternatives + 1)},
            availability=None,
            nests=build_nests(alternatives, nests_count=10, memberships=2),
            choice=1,
        )
        messages_after_construction = recommendation_messages(caplog)
        expression.deep_flat_copy()
        expression.get_value()
        expression.get_value()

    assert len(messages_after_construction) == 1
    assert 'log_sparse_cnl or sparse_cnl' in messages_after_construction[0]
    assert recommendation_messages(caplog) == messages_after_construction


def test_sparse_cnl_suggests_dense_once_at_construction(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        expression = log_sparse_cnl(
            util={1: 0.1, 2: 0.2, 3: 0.3},
            availability=None,
            nests=build_nests(3, nests_count=2, memberships=2),
            choice=1,
        )
        messages_after_construction = recommendation_messages(caplog)
        expression.get_value()

    assert isinstance(expression, SparseLogCrossNested)
    assert len(messages_after_construction) == 1
    assert 'logcnl or cnl' in messages_after_construction[0]
    assert recommendation_messages(caplog) == messages_after_construction


def test_parameter_dependent_zero_membership_remains_active():
    alpha = Beta('alpha', 0.0, 0.0, 1.0, 0)
    nests = NestsForCrossNestedLogit(
        choice_set=[1, 2],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=1.2,
                dict_of_alpha={1: alpha, 2: 1.0},
                name='first',
            ),
            OneNestForCrossNestedLogit(
                nest_param=1.3,
                dict_of_alpha={1: 1.0 - alpha, 2: 0.0},
                name='second',
            ),
        ),
    )
    expression = log_sparse_cnl(
        util={1: 0.1, 2: 0.2},
        availability=None,
        nests=nests,
        choice=1,
    )
    assert expression.number_of_dense_memberships == 4
    assert expression.number_of_active_memberships == 3


def test_sparse_probability_public_api():
    probability = sparse_cnl(
        util={1: 0.1, 2: 0.2},
        availability=None,
        nests=build_nests(2, nests_count=2, memberships=1),
        choice=1,
    )
    assert probability.get_value() > 0.0
