"""Regression tests for numerically safe likelihood availability handling."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.expressions.log_cross_nested import LogCrossNested
from biogeme.expressions.log_nested import LogNested
from biogeme.expressions.logit_expressions import LogLogit
from biogeme.expressions.sparse_log_cross_nested import SparseLogCrossNested
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.jax_calculator.function_call import function_from_compiled_formula
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.nests import (
    NestsForCrossNestedLogit,
    NestsForNestedLogit,
    OneNestForCrossNestedLogit,
    OneNestForNestedLogit,
)
from biogeme.parameters import Parameters
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.tools.derivatives import findiff_g, findiff_h

ExpressionFactory = Callable[[dict[int, object], dict[int, object]], object]
EXPECTED_NEGATIVE_LARGE = -1.0e30


def _database(availability_dtype: type = float) -> Database:
    """Swissmetro-shaped data with a sometimes-empty singleton Car nest."""
    return Database(
        'availability_safety',
        pd.DataFrame(
            {
                'x1': [0.2, 0.7, -0.4, 1.1, 0.5],
                'x2': [0.8, -0.2, 1.0, 0.1, -0.6],
                'x3': [-0.3, 0.5, 0.2, -0.8, 1.2],
                'av1': np.asarray([1, 1, 1, 1, 1], dtype=availability_dtype),
                'av2': np.asarray([1, 1, 1, 0, 1], dtype=availability_dtype),
                'av3': np.asarray([0, 1, 0, 1, 0], dtype=availability_dtype),
                'choice': [1, 2, 1, 3, 2],
            }
        ),
    )


def _utilities() -> dict[int, object]:
    beta = Beta('beta', -0.7, None, None, 0)
    asc2 = Beta('asc2', 0.1, None, None, 0)
    asc3 = Beta('asc3', -0.2, None, None, 0)
    return {
        1: beta * Variable('x1'),
        2: asc2 + beta * Variable('x2'),
        3: asc3 + beta * Variable('x3'),
    }


def _availability() -> dict[int, object]:
    return {alternative: Variable(f'av{alternative}') for alternative in (1, 2, 3)}


def _logit_expression(utilities, availability):
    return LogLogit(utilities, availability, Variable('choice'))


def _nested_expression(utilities, availability):
    public_transport = OneNestForNestedLogit(
        nest_param=Beta('mu_pt', 1.3, 1.0, 5.0, 0),
        list_of_alternatives=[1, 2],
        name='public_transport',
    )
    car = OneNestForNestedLogit(
        nest_param=1.0,
        list_of_alternatives=[3],
        name='car',
    )
    nests = NestsForNestedLogit(
        choice_set=[1, 2, 3], tuple_of_nests=(public_transport, car)
    )
    return LogNested(utilities, availability, nests, Variable('choice'))


def _cnl_nests() -> NestsForCrossNestedLogit:
    return NestsForCrossNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=Beta('mu_pt', 1.3, 1.0, 5.0, 0),
                dict_of_alpha={1: 1.0, 2: 1.0},
                name='public_transport',
            ),
            OneNestForCrossNestedLogit(
                nest_param=1.0,
                dict_of_alpha={3: 1.0},
                name='car',
            ),
        ),
    )


def _dense_cnl_expression(utilities, availability):
    return LogCrossNested(
        utilities, availability, _cnl_nests(), Variable('choice')
    )


def _sparse_cnl_expression(utilities, availability):
    return SparseLogCrossNested(
        utilities, availability, _cnl_nests(), Variable('choice')
    )


EXPRESSION_FACTORIES = (
    pytest.param(_logit_expression, id='logit'),
    pytest.param(_nested_expression, id='nested'),
    pytest.param(_dense_cnl_expression, id='dense-cnl'),
    pytest.param(_sparse_cnl_expression, id='sparse-cnl'),
)


def _evaluator(
    expression_factory: ExpressionFactory,
    *,
    numerically_safe: bool,
    availability_dtype: type = float,
    database: Database | None = None,
) -> CompiledFormulaEvaluator:
    database = _database(availability_dtype) if database is None else database
    expression = expression_factory(_utilities(), _availability())
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=True,
    )
    return CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=numerically_safe,
    )


def _beta_values(evaluator: CompiledFormulaEvaluator) -> dict[str, float]:
    return evaluator.model_elements.expressions_registry.free_betas_init_values


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
@pytest.mark.parametrize('availability_dtype', [bool, int, float])
@pytest.mark.parametrize('numerically_safe', [False, True])
def test_derivatives_are_finite_with_varying_availability(
    expression_factory: ExpressionFactory,
    availability_dtype: type,
    numerically_safe: bool,
):
    evaluator = _evaluator(
        expression_factory,
        numerically_safe=numerically_safe,
        availability_dtype=availability_dtype,
    )
    betas = _beta_values(evaluator)

    result = evaluator.evaluate(
        betas, gradient=True, hessian=True, bhhh=True
    )
    individual = evaluator.evaluate_individual(betas)

    assert np.isfinite(result.function)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()
    assert np.isfinite(individual).all()


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
@pytest.mark.parametrize('numerically_safe', [False, True])
def test_automatic_derivatives_match_finite_differences(
    expression_factory: ExpressionFactory,
    numerically_safe: bool,
):
    evaluator = _evaluator(
        expression_factory, numerically_safe=numerically_safe
    )
    betas = _beta_values(evaluator)
    x = np.asarray(list(betas.values()), dtype=float)
    callable_expression = function_from_compiled_formula(evaluator, betas.copy())

    analytical = callable_expression(
        x, gradient=True, hessian=True, bhhh=False
    )
    finite_difference_gradient = findiff_g(callable_expression, x)
    finite_difference_hessian = findiff_h(callable_expression, x)

    np.testing.assert_allclose(
        analytical.gradient,
        finite_difference_gradient,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        analytical.hessian,
        finite_difference_hessian,
        rtol=2.0e-4,
        atol=2.0e-5,
    )


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
def test_unavailable_chosen_alternative_returns_finite_sentinel_and_derivatives(
    expression_factory: ExpressionFactory,
):
    database = Database(
        'invalid_choice',
        pd.DataFrame(
            {
                'x1': [0.0],
                'x2': [0.0],
                'x3': [0.0],
                'av1': [1],
                'av2': [0],
                'av3': [0],
                'choice': [2],
            }
        ),
    )
    expression = expression_factory(_utilities(), _availability())
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=True,
    )
    evaluator = CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=True,
    )
    betas = _beta_values(evaluator)

    result = evaluator.evaluate(
        betas, gradient=True, hessian=True, bhhh=True
    )

    assert result.function == pytest.approx(EXPECTED_NEGATIVE_LARGE)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()


def test_fast_nested_matches_safe_with_empty_singleton_nest():
    fast = _evaluator(_nested_expression, numerically_safe=False)
    safe = _evaluator(_nested_expression, numerically_safe=True)
    fast_result = fast.evaluate(
        _beta_values(fast), gradient=True, hessian=True, bhhh=True
    )
    safe_result = safe.evaluate(
        _beta_values(safe), gradient=True, hessian=True, bhhh=True
    )

    assert fast_result.function == pytest.approx(
        safe_result.function, rel=1.0e-11, abs=1.0e-11
    )
    for name in ('gradient', 'hessian', 'bhhh'):
        np.testing.assert_allclose(
            getattr(fast_result, name),
            getattr(safe_result, name),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        assert np.isfinite(getattr(fast_result, name)).all()


def test_actual_swissmetro_empty_singleton_car_nest_regression():
    """Reproduce the original failure on identified Swissmetro observations."""
    from biogeme.data.swissmetro import read_data

    full_data = read_data().dataframe
    affected_mask = (full_data['CAR_AV_SP'] == 0) & full_data['CHOICE'].isin([1, 2])
    affected_rows = tuple(full_data.index[affected_mask])
    assert len(affected_rows) == 1683
    assert affected_rows[:18] == (
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
    )

    # Include affected rows plus ordinary rows in the same compiled objective.
    selected_rows = (0, 1, 2, *affected_rows[:18])
    database = Database(
        'swissmetro_empty_car_nest', full_data.loc[list(selected_rows)].copy()
    )
    asc_train = Beta('asc_train', 0.0, None, None, 0)
    asc_car = Beta('asc_car', 0.0, None, None, 0)
    b_time = Beta('b_time', -1.0, None, None, 0)
    b_cost = Beta('b_cost', -1.0, None, None, 0)
    utilities = {
        1: asc_train
        + b_time * Variable('TRAIN_TT_SCALED')
        + b_cost * Variable('TRAIN_COST_SCALED'),
        2: b_time * Variable('SM_TT_SCALED')
        + b_cost * Variable('SM_COST_SCALED'),
        3: asc_car
        + b_time * Variable('CAR_TT_SCALED')
        + b_cost * Variable('CAR_CO_SCALED'),
    }
    availability = {
        1: Variable('TRAIN_AV_SP'),
        2: Variable('SM_AV'),
        3: Variable('CAR_AV_SP'),
    }
    nests = NestsForNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(
            OneNestForNestedLogit(
                nest_param=Beta('mu_pt', 1.2, 1.0, 5.0, 0),
                list_of_alternatives=[1, 2],
                name='public_transport',
            ),
            OneNestForNestedLogit(
                nest_param=1.0,
                list_of_alternatives=[3],
                name='car',
            ),
        ),
    )
    expression = LogNested(
        utilities, availability, nests, Variable('CHOICE')
    )
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=True,
    )

    def evaluate(safe: bool):
        evaluator = CompiledFormulaEvaluator(
            model_elements=elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=safe,
        )
        return evaluator, evaluator.evaluate(
            evaluator.model_elements.expressions_registry.free_betas_init_values,
            gradient=True,
            hessian=True,
            bhhh=True,
        )

    _, safe_result = evaluate(True)
    _, fast_result = evaluate(False)

    assert fast_result.function == pytest.approx(
        safe_result.function, rel=1.0e-11, abs=1.0e-11
    )
    for name in ('gradient', 'hessian', 'bhhh'):
        fast_value = getattr(fast_result, name)
        safe_value = getattr(safe_result, name)
        assert np.isfinite(fast_value).all()
        np.testing.assert_allclose(
            fast_value,
            safe_value,
            rtol=1.0e-10,
            atol=1.0e-10,
        )


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
@pytest.mark.parametrize('numerically_safe', [False, True])
def test_paths_handle_all_alternatives_unavailable(
    expression_factory: ExpressionFactory,
    numerically_safe: bool,
):
    database = Database(
        'all_unavailable',
        pd.DataFrame(
            {
                'x1': [0.0],
                'x2': [0.0],
                'x3': [0.0],
                'av1': [0],
                'av2': [0],
                'av3': [0],
                'choice': [1],
            }
        ),
    )
    evaluator = _evaluator(
        expression_factory,
        numerically_safe=numerically_safe,
        database=database,
    )
    result = evaluator.evaluate(
        _beta_values(evaluator), gradient=True, hessian=True, bhhh=True
    )

    assert result.function == pytest.approx(EXPECTED_NEGATIVE_LARGE)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()


def test_fast_logit_handles_all_alternatives_unavailable():
    database = Database(
        'fast_logit_all_unavailable',
        pd.DataFrame(
            {
                'x1': [0.0],
                'x2': [0.0],
                'x3': [0.0],
                'av1': [0],
                'av2': [0],
                'av3': [0],
                'choice': [1],
            }
        ),
    )
    evaluator = _evaluator(
        _logit_expression, numerically_safe=False, database=database
    )
    result = evaluator.evaluate(
        _beta_values(evaluator), gradient=True, hessian=True, bhhh=True
    )

    assert result.function == pytest.approx(EXPECTED_NEGATIVE_LARGE)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
def test_fast_paths_treat_nonzero_availability_as_boolean(
    expression_factory: ExpressionFactory,
):
    reference_database = _database()
    nonbinary_database = _database()
    for column in ('av1', 'av2', 'av3'):
        reference_database.dataframe[column] = 1.0
    nonbinary_database.dataframe['av1'] = 2.0
    nonbinary_database.dataframe['av2'] = -3.0
    nonbinary_database.dataframe['av3'] = 0.5

    reference = _evaluator(
        expression_factory,
        numerically_safe=False,
        database=reference_database,
    )
    nonbinary = _evaluator(
        expression_factory,
        numerically_safe=False,
        database=nonbinary_database,
    )
    betas = _beta_values(reference)
    expected = reference.evaluate(
        betas, gradient=True, hessian=True, bhhh=True
    )
    actual = nonbinary.evaluate(
        betas, gradient=True, hessian=True, bhhh=True
    )

    assert actual.function == pytest.approx(
        expected.function, rel=1.0e-12, abs=1.0e-12
    )
    np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=1.0e-12)
    np.testing.assert_allclose(actual.hessian, expected.hessian, rtol=1.0e-12)
    np.testing.assert_allclose(actual.bhhh, expected.bhhh, rtol=1.0e-12)


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
@pytest.mark.parametrize(
    ('availability', 'choice'),
    [
        ({'av1': 1, 'av2': 1, 'av3': 1}, 99),
        ({'av1': 1, 'av2': 0, 'av3': 1}, 2),
    ],
    ids=('unknown-choice', 'unavailable-choice'),
)
def test_fast_paths_return_common_sentinel_for_invalid_choices(
    expression_factory: ExpressionFactory,
    availability: dict[str, int],
    choice: int,
):
    database = Database(
        'fast_invalid_choice',
        pd.DataFrame(
            {
                'x1': [0.1],
                'x2': [0.2],
                'x3': [0.3],
                **{name: [value] for name, value in availability.items()},
                'choice': [choice],
            }
        ),
    )
    evaluator = _evaluator(
        expression_factory, numerically_safe=False, database=database
    )
    result = evaluator.evaluate(
        _beta_values(evaluator), gradient=True, hessian=True, bhhh=True
    )

    assert result.function == pytest.approx(EXPECTED_NEGATIVE_LARGE)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
@pytest.mark.parametrize('numerically_safe', [False, True])
def test_paths_handle_an_empty_non_singleton_nest(
    expression_factory: ExpressionFactory,
    numerically_safe: bool,
):
    database = Database(
        'empty_public_transport_nest',
        pd.DataFrame(
            {
                'x1': [0.1],
                'x2': [0.2],
                'x3': [0.3],
                'av1': [0],
                'av2': [0],
                'av3': [1],
                'choice': [3],
            }
        ),
    )
    evaluator = _evaluator(
        expression_factory,
        numerically_safe=numerically_safe,
        database=database,
    )
    result = evaluator.evaluate(
        _beta_values(evaluator), gradient=True, hessian=True, bhhh=True
    )

    assert np.isfinite(result.function)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()


@pytest.mark.parametrize(
    'expression_class', [LogNested, LogCrossNested, SparseLogCrossNested]
)
@pytest.mark.parametrize('numerically_safe', [False, True])
def test_paths_handle_several_empty_nests(
    expression_class, numerically_safe
):
    database = Database(
        'several_empty_nests',
        pd.DataFrame(
            {
                'x1': [0.1],
                'x2': [0.2],
                'x3': [0.3],
                'x4': [0.4],
                'av1': [0],
                'av2': [0],
                'av3': [0],
                'av4': [1],
                'choice': [4],
            }
        ),
    )
    beta = Beta('beta', -0.5, None, None, 0)
    utilities = {
        alternative: beta * Variable(f'x{alternative}')
        for alternative in (1, 2, 3, 4)
    }
    availability = {
        alternative: Variable(f'av{alternative}')
        for alternative in (1, 2, 3, 4)
    }
    if expression_class is LogNested:
        nests = NestsForNestedLogit(
            choice_set=[1, 2, 3, 4],
            tuple_of_nests=(
                OneNestForNestedLogit(1.1, [1], 'first'),
                OneNestForNestedLogit(1.2, [2, 3], 'second'),
                OneNestForNestedLogit(1.0, [4], 'active'),
            ),
        )
    else:
        nests = NestsForCrossNestedLogit(
            choice_set=[1, 2, 3, 4],
            tuple_of_nests=(
                OneNestForCrossNestedLogit(
                    1.1, {1: 0.5, 2: 1.0}, 'first'
                ),
                OneNestForCrossNestedLogit(
                    1.2, {1: 0.5, 3: 1.0}, 'second'
                ),
                OneNestForCrossNestedLogit(1.0, {4: 1.0}, 'active'),
            ),
        )
    expression = expression_class(
        utilities, availability, nests, Variable('choice')
    )
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        use_jit=True,
    )
    evaluator = CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=numerically_safe,
    )
    result = evaluator.evaluate(
        {'beta': -0.5}, gradient=True, hessian=True, bhhh=True
    )

    assert np.isfinite(result.function)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()


@pytest.mark.parametrize('mu_pt', [1.0, 1.0 + 1.0e-8, 1.3, 4.9])
def test_nested_safe_derivatives_are_finite_across_nest_parameter_range(mu_pt):
    evaluator = _evaluator(_nested_expression, numerically_safe=True)
    betas = _beta_values(evaluator)
    betas['mu_pt'] = mu_pt

    result = evaluator.evaluate(
        betas, gradient=True, hessian=True, bhhh=True
    )

    assert np.isfinite(result.function)
    assert np.isfinite(result.gradient).all()
    assert np.isfinite(result.hessian).all()
    assert np.isfinite(result.bhhh).all()


def test_nested_with_unit_nest_parameters_equals_logit():
    logit_evaluator = _evaluator(_logit_expression, numerically_safe=True)
    nested_evaluator = _evaluator(_nested_expression, numerically_safe=True)
    logit_result = logit_evaluator.evaluate(
        _beta_values(logit_evaluator), gradient=False, hessian=False, bhhh=False
    )
    nested_betas = _beta_values(nested_evaluator)
    nested_betas['mu_pt'] = 1.0
    nested_result = nested_evaluator.evaluate(
        nested_betas, gradient=False, hessian=False, bhhh=False
    )

    assert nested_result.function == pytest.approx(
        logit_result.function, rel=1.0e-12, abs=1.0e-12
    )


@pytest.mark.parametrize(
    'cnl_factory', [_dense_cnl_expression, _sparse_cnl_expression]
)
def test_zero_one_cnl_allocations_equal_nested(cnl_factory):
    nested_evaluator = _evaluator(_nested_expression, numerically_safe=True)
    cnl_evaluator = _evaluator(cnl_factory, numerically_safe=True)
    nested_result = nested_evaluator.evaluate(
        _beta_values(nested_evaluator), gradient=False, hessian=False, bhhh=False
    )
    cnl_result = cnl_evaluator.evaluate(
        _beta_values(cnl_evaluator), gradient=False, hessian=False, bhhh=False
    )

    assert cnl_result.function == pytest.approx(
        nested_result.function, rel=1.0e-12, abs=1.0e-12
    )


@pytest.mark.parametrize(
    'expression_class', [LogNested, LogCrossNested, SparseLogCrossNested]
)
def test_retaining_an_unavailable_nest_equals_removing_it(expression_class):
    """An empty nest and an analytically removed nest are equivalent."""
    database = Database(
        'removed_nest_equivalence',
        pd.DataFrame(
            {
                'x1': [0.2, -0.4],
                'x2': [0.7, 0.1],
                'x3': [1.2, -0.8],
                'av1': [1, 1],
                'av2': [1, 1],
                'av3': [0, 0],
                'choice': [1, 2],
            }
        ),
    )
    beta = Beta('beta', -0.6, None, None, 0)
    full_utilities = {
        alternative: beta * Variable(f'x{alternative}')
        for alternative in (1, 2, 3)
    }
    reduced_utilities = {
        alternative: full_utilities[alternative] for alternative in (1, 2)
    }
    full_availability = _availability()
    reduced_availability = {
        alternative: full_availability[alternative] for alternative in (1, 2)
    }

    if expression_class is LogNested:
        full_nests = NestsForNestedLogit(
            choice_set=[1, 2, 3],
            tuple_of_nests=(
                OneNestForNestedLogit(1.3, [1, 2], 'public_transport'),
                OneNestForNestedLogit(1.0, [3], 'car'),
            ),
        )
        reduced_nests = NestsForNestedLogit(
            choice_set=[1, 2],
            tuple_of_nests=(
                OneNestForNestedLogit(1.3, [1, 2], 'public_transport'),
            ),
        )
    else:
        full_nests = NestsForCrossNestedLogit(
            choice_set=[1, 2, 3],
            tuple_of_nests=(
                OneNestForCrossNestedLogit(
                    1.3, {1: 1.0, 2: 1.0}, 'public_transport'
                ),
                OneNestForCrossNestedLogit(1.0, {3: 1.0}, 'car'),
            ),
        )
        reduced_nests = NestsForCrossNestedLogit(
            choice_set=[1, 2],
            tuple_of_nests=(
                OneNestForCrossNestedLogit(
                    1.3, {1: 1.0, 2: 1.0}, 'public_transport'
                ),
            ),
        )

    full_expression = expression_class(
        full_utilities, full_availability, full_nests, Variable('choice')
    )
    reduced_expression = expression_class(
        reduced_utilities,
        reduced_availability,
        reduced_nests,
        Variable('choice'),
    )

    def compile_and_evaluate(expression):
        elements = ModelElements.from_expression_and_weight(
            log_like=expression,
            weight=None,
            adapter=RegularAdapter(database=database),
            use_jit=True,
        )
        evaluator = CompiledFormulaEvaluator(
            model_elements=elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=True,
        )
        beta_values = {'beta': -0.6}
        return (
            evaluator.evaluate(
                beta_values, gradient=True, hessian=True, bhhh=True
            ),
            evaluator.evaluate_individual(beta_values),
        )

    (full_result, full_individual) = compile_and_evaluate(full_expression)
    (reduced_result, reduced_individual) = compile_and_evaluate(
        reduced_expression
    )

    differences = (
        abs(full_result.function - reduced_result.function),
        np.max(np.abs(full_result.gradient - reduced_result.gradient)),
        np.max(np.abs(full_result.hessian - reduced_result.hessian)),
        np.max(np.abs(full_result.bhhh - reduced_result.bhhh)),
        np.max(np.abs(full_individual - reduced_individual)),
    )
    assert max(differences) < 1.0e-12


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
def test_safe_and_unsafe_paths_agree_when_everything_is_available(
    expression_factory: ExpressionFactory,
):
    database = _database()
    for column in ('av1', 'av2', 'av3'):
        database.dataframe[column] = 1
    safe = _evaluator(
        expression_factory, numerically_safe=True, database=database
    )
    unsafe = _evaluator(
        expression_factory, numerically_safe=False, database=database
    )
    safe_result = safe.evaluate(
        _beta_values(safe), gradient=True, hessian=True, bhhh=True
    )
    unsafe_result = unsafe.evaluate(
        _beta_values(unsafe), gradient=True, hessian=True, bhhh=True
    )

    assert safe_result.function == pytest.approx(
        unsafe_result.function, rel=1.0e-11, abs=1.0e-11
    )
    np.testing.assert_allclose(
        safe_result.gradient, unsafe_result.gradient, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        safe_result.hessian, unsafe_result.hessian, rtol=1.0e-9, atol=1.0e-9
    )
    np.testing.assert_allclose(
        safe_result.bhhh, unsafe_result.bhhh, rtol=1.0e-10, atol=1.0e-10
    )


def _estimation_database() -> Database:
    rows = np.arange(60)
    av2 = rows % 7 != 0
    av3 = rows % 3 != 0
    choices = []
    for row, second_available, third_available in zip(
        rows, av2, av3, strict=True
    ):
        available = [1]
        if second_available:
            available.append(2)
        if third_available:
            available.append(3)
        choices.append(available[row % len(available)])
    return Database(
        'safe_estimation',
        pd.DataFrame(
            {
                'x1': np.sin(rows / 5.0),
                'x2': np.cos(rows / 7.0),
                'x3': np.sin(rows / 9.0 + 0.4),
                'av1': np.ones_like(rows),
                'av2': av2,
                'av3': av3,
                'choice': choices,
            }
        ),
    )


def test_numerically_safe_configuration_reaches_evaluation_and_simulation():
    database = _database()
    expression = _nested_expression(_utilities(), _availability())
    biogeme = BIOGEME(
        database,
        {'log_like': expression, 'probability': expression},
        parameters=Parameters(),
        numerically_safe=True,
        calculating_second_derivatives='analytical',
        generate_yaml=False,
        generate_html=False,
        save_iterations=False,
    )

    assert biogeme.numerically_safe is True
    assert biogeme.function_evaluator.numerically_safe is True
    simulated = biogeme.simulate(
        biogeme.model_elements.expressions_registry.free_betas_init_values
    )
    assert np.isfinite(simulated.to_numpy()).all()
    assert biogeme._simulation_evaluator.numerically_safe is True


@pytest.mark.parametrize('expression_factory', EXPRESSION_FACTORIES)
def test_safe_small_estimation_and_serialization(
    expression_factory: ExpressionFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_directory = tmp_path / expression_factory.__name__
    output_directory.mkdir()
    monkeypatch.chdir(output_directory)
    expression = expression_factory(_utilities(), _availability())
    biogeme = BIOGEME(
        _estimation_database(),
        expression,
        parameters=Parameters(),
        numerically_safe=True,
        calculating_second_derivatives='analytical',
        optimization_algorithm='scipy',
        max_iterations=80,
        bootstrap_samples=0,
        generate_yaml=True,
        generate_html=True,
        save_iterations=False,
    )
    biogeme.model_name = f'safe_{expression_factory.__name__.strip("_")}'

    results = biogeme.estimate()
    raw = results.raw_estimation_results

    assert results.algorithm_has_converged
    assert np.isfinite(results.final_log_likelihood)
    assert np.isfinite(raw.gradient).all()
    assert np.isfinite(raw.hessian).all()
    assert np.isfinite(raw.bhhh).all()
    assert Path(f'{biogeme.model_name}.yaml').is_file()
    assert Path(f'{biogeme.model_name}.html').is_file()
