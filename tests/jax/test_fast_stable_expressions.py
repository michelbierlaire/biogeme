"""Complete-evaluator tests for routine fast likelihood paths."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.expressions.log_cross_nested import LogCrossNested
from biogeme.expressions.log_nested import LogNested
from biogeme.expressions.sparse_log_cross_nested import SparseLogCrossNested
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.nests import (
    NestsForCrossNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    OneNestForNestedLogit,
)
from biogeme.second_derivatives import SecondDerivativesMode


def database(*, all_available: bool = False, extreme: bool = False) -> Database:
    rows = np.arange(80)
    scale = 2_000.0 if extreme else 1.0
    av3 = np.ones_like(rows) if all_available else rows % 3 != 0
    av4 = np.ones_like(rows) if all_available else rows % 5 != 0
    choices = []
    for row, third, fourth in zip(rows, av3, av4, strict=True):
        available = [1, 2]
        if third:
            available.append(3)
        if fourth:
            available.append(4)
        choices.append(available[row % len(available)])
    return Database(
        'experimental_fast_stable',
        pd.DataFrame(
            {
                'x1': scale * np.sin(rows / 5.0),
                'x2': scale * np.cos(rows / 7.0),
                'x3': scale * np.sin(rows / 9.0 + 0.4),
                'x4': scale * np.cos(rows / 11.0 - 0.2),
                'av1': np.ones_like(rows),
                'av2': np.ones_like(rows),
                'av3': av3,
                'av4': av4,
                'choice': choices,
            }
        ),
    )


def utilities():
    beta = Beta('beta', -0.7, None, None, 0)
    asc2 = Beta('asc2', 0.1, None, None, 0)
    return {
        1: beta * Variable('x1'),
        2: asc2 + beta * Variable('x2'),
        3: beta * Variable('x3'),
        4: beta * Variable('x4'),
    }


def availability():
    return {alternative: Variable(f'av{alternative}') for alternative in range(1, 5)}


def nested_expression(expression_class, *, explicit_mu: bool):
    nests = NestsForNestedLogit(
        choice_set=[1, 2, 3, 4],
        tuple_of_nests=(
            OneNestForNestedLogit(
                Beta('mu12', 1.3, 1.0, 4.0, 0), [1, 2], 'first'
            ),
            OneNestForNestedLogit(1.0, [3], 'third'),
            OneNestForNestedLogit(1.0, [4], 'fourth'),
        ),
    )
    global_mu = Beta('global_mu', 1.1, 0.5, 3.0, 0) if explicit_mu else None
    return expression_class(
        utilities(),
        availability(),
        nests,
        Variable('choice'),
        mu=global_mu,
    )


def cnl_expression(expression_class, *, explicit_mu: bool):
    nests = NestsForCrossNestedLogit(
        choice_set=[1, 2, 3, 4],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                Beta('mu12', 1.3, 1.0, 4.0, 0),
                {1: 1.0, 2: 1.0},
                'first',
            ),
            OneNestForCrossNestedLogit(1.0, {3: 1.0}, 'third'),
            OneNestForCrossNestedLogit(1.0, {4: 1.0}, 'fourth'),
        ),
    )
    global_mu = Beta('global_mu', 1.1, 0.5, 3.0, 0) if explicit_mu else None
    return expression_class(
        utilities(),
        availability(),
        nests,
        Variable('choice'),
        mu=global_mu,
    )


Case = tuple[Callable, type]
CASES: tuple[Case, ...] = (
    (nested_expression, LogNested),
    (cnl_expression, LogCrossNested),
    (cnl_expression, SparseLogCrossNested),
)


def evaluator(expression, the_database, *, numerically_safe):
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=the_database),
        use_jit=True,
    )
    return CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=numerically_safe,
    )


def evaluate(the_evaluator):
    betas = (
        the_evaluator.model_elements.expressions_registry.free_betas_init_values
    )
    return (
        the_evaluator.evaluate(
            betas, gradient=True, hessian=True, bhhh=True
        ),
        the_evaluator.evaluate_individual(betas),
    )


def assert_equal(left, right):
    assert left.function == pytest.approx(
        right.function, rel=1.0e-11, abs=1.0e-11
    )
    for name in ('gradient', 'hessian', 'bhhh'):
        np.testing.assert_allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )


@pytest.mark.parametrize(
    ('expression_factory', 'expression_class'), CASES
)
@pytest.mark.parametrize('explicit_mu', (False, True))
def test_fast_matches_safe_evaluator_with_empty_nests(
    expression_factory, expression_class, explicit_mu
):
    the_database = database()
    safe, safe_individual = evaluate(
        evaluator(
            expression_factory(expression_class, explicit_mu=explicit_mu),
            the_database,
            numerically_safe=True,
        )
    )
    fast, fast_individual = evaluate(
        evaluator(
            expression_factory(expression_class, explicit_mu=explicit_mu),
            the_database,
            numerically_safe=False,
        )
    )

    assert_equal(fast, safe)
    np.testing.assert_allclose(
        fast_individual, safe_individual, rtol=1.0e-11, atol=1.0e-11
    )
    for output in (fast, safe):
        assert np.isfinite(output.function)
        assert np.isfinite(output.gradient).all()
        assert np.isfinite(output.hessian).all()
        assert np.isfinite(output.bhhh).all()


@pytest.mark.parametrize(
    ('expression_factory', 'expression_class'), CASES
)
@pytest.mark.parametrize('explicit_mu', (False, True))
def test_fast_matches_safe_evaluator_when_everything_is_available(
    expression_factory, expression_class, explicit_mu
):
    the_database = database(all_available=True)
    safe, safe_individual = evaluate(
        evaluator(
            expression_factory(expression_class, explicit_mu=explicit_mu),
            the_database,
            numerically_safe=True,
        )
    )
    fast, fast_individual = evaluate(
        evaluator(
            expression_factory(expression_class, explicit_mu=explicit_mu),
            the_database,
            numerically_safe=False,
        )
    )

    assert_equal(fast, safe)
    np.testing.assert_allclose(
        fast_individual,
        safe_individual,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


@pytest.mark.parametrize(
    ('expression_factory', 'expression_class'), CASES
)
@pytest.mark.parametrize('explicit_mu', (False, True))
def test_fast_deliberately_does_not_replace_extreme_utility_safe_path(
    expression_factory, expression_class, explicit_mu
):
    the_database = database(all_available=True, extreme=True)
    safe, _ = evaluate(
        evaluator(
            expression_factory(expression_class, explicit_mu=explicit_mu),
            the_database,
            numerically_safe=True,
        )
    )
    fast, _ = evaluate(
        evaluator(
            expression_factory(expression_class, explicit_mu=explicit_mu),
            the_database,
            numerically_safe=False,
        )
    )

    assert np.isfinite(safe.function)
    assert np.isfinite(safe.gradient).all()
    assert np.isfinite(safe.hessian).all()
    assert not (
        np.isfinite(fast.function)
        and np.isfinite(fast.gradient).all()
        and np.isfinite(fast.hessian).all()
    )
