"""Dense-versus-sparse equivalence tests for sparse CNL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.expressions.log_cross_nested import LogCrossNested
from biogeme.expressions.sparse_log_cross_nested import SparseLogCrossNested
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.second_derivatives import SecondDerivativesMode


def build_model(expression_class, *, explicit_mu: bool) -> ModelElements:
    database = Database(
        'sparse_cnl_equivalence',
        pd.DataFrame(
            {
                'x1': [0.2, 1.1, -0.4, 0.8, 1.7, -0.2, 0.5, 1.3],
                'x2': [1.2, -0.3, 0.7, 1.5, 0.1, 0.9, -0.6, 0.4],
                'x3': [-0.5, 0.8, 1.4, -0.1, 0.6, 1.2, 0.3, -0.7],
                'x4': [0.9, 0.4, -0.2, 1.1, 0.7, -0.4, 1.6, 0.2],
                'av1': [1, 1, 1, 1, 1, 1, 1, 1],
                'av2': [1, 1, 0, 1, 1, 1, 1, 0],
                'av3': [1, 1, 1, 1, 0, 1, 1, 1],
                'av4': [1, 1, 1, 1, 1, 1, 1, 1],
                'choice': [1, 2, 3, 4, 1, 2, 3, 4],
                'weight': [1.0, 0.7, 1.3, 0.9, 1.1, 0.8, 1.4, 0.6],
            }
        ),
    )
    beta = Beta('beta', -0.8, None, None, 0)
    asc1 = Beta('asc1', 0.0, None, None, 0)
    asc2 = Beta('asc2', 0.0, None, None, 0)
    mu1 = Beta('mu1', 1.2, 1.0, 4.0, 0)
    mu2 = Beta('mu2', 1.4, 1.0, 4.0, 0)
    mu3 = Beta('mu3', 1.6, 1.0, 4.0, 0)
    alpha = Beta('alpha', 0.35, 0.01, 0.99, 0)
    global_mu = Beta('global_mu', 1.0, 0.5, 3.0, 0)

    utilities = {
        1: asc1 + beta * Variable('x1'),
        2: asc2 + beta * Variable('x2'),
        3: beta * Variable('x3'),
        4: beta * Variable('x4'),
    }
    availability = {i: Variable(f'av{i}') for i in utilities}
    nests = NestsForCrossNestedLogit(
        choice_set=list(utilities),
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=mu1,
                dict_of_alpha={1: alpha, 2: 1.0, 3: 0.0, 4: 0.0},
                name='n1',
            ),
            OneNestForCrossNestedLogit(
                nest_param=mu2,
                dict_of_alpha={1: 1.0 - alpha, 2: 0.0, 3: 1.0, 4: 0.4},
                name='n2',
            ),
            OneNestForCrossNestedLogit(
                nest_param=mu3,
                dict_of_alpha={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.6},
                name='n3',
            ),
        ),
    )
    expression = expression_class(
        util=utilities,
        av=availability,
        nests=nests,
        choice=Variable('choice'),
        mu=global_mu if explicit_mu else None,
    )
    return ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=Variable('weight'),
        adapter=RegularAdapter(database=database),
        use_jit=True,
    )


def assert_outputs_equal(expected, actual):
    assert actual.function == pytest.approx(expected.function, rel=1e-11, abs=1e-11)
    for name in ('gradient', 'hessian', 'bhhh'):
        expected_array = getattr(expected, name)
        actual_array = getattr(actual, name)
        if expected_array is None:
            assert actual_array is None
        else:
            np.testing.assert_allclose(
                actual_array, expected_array, rtol=1e-10, atol=1e-10
            )


@pytest.mark.parametrize('explicit_mu', [False, True])
@pytest.mark.parametrize(
    'mode',
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
    ],
)
@pytest.mark.parametrize(
    'betas',
    [
        {
            'asc1': 0.0,
            'asc2': 0.0,
            'beta': -0.8,
            'mu1': 1.2,
            'mu2': 1.4,
            'mu3': 1.6,
            'alpha': 0.35,
            'global_mu': 1.0,
        },
        {
            'asc1': 0.7,
            'asc2': -0.4,
            'beta': -1.7,
            'mu1': 1.25,
            'mu2': 2.1,
            'mu3': 1.3,
            'alpha': 0.12,
            'global_mu': 1.2,
        },
        {
            'asc1': -0.9,
            'asc2': 1.1,
            'beta': -0.2,
            'mu1': 2.8,
            'mu2': 1.1,
            'mu3': 2.2,
            'alpha': 0.87,
            'global_mu': 0.9,
        },
    ],
)
def test_sparse_cnl_matches_dense(betas, mode, explicit_mu):
    dense = CompiledFormulaEvaluator(
        model_elements=build_model(LogCrossNested, explicit_mu=explicit_mu),
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    sparse = CompiledFormulaEvaluator(
        model_elements=build_model(
            SparseLogCrossNested, explicit_mu=explicit_mu
        ),
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    log_domain = CompiledFormulaEvaluator(
        model_elements=build_model(LogCrossNested, explicit_mu=explicit_mu),
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=True,
    )
    sparse_log_domain = CompiledFormulaEvaluator(
        model_elements=build_model(SparseLogCrossNested, explicit_mu=explicit_mu),
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=True,
    )
    expected = dense.evaluate(betas, *mode)
    actual = sparse.evaluate(betas, *mode)
    log_domain_actual = log_domain.evaluate(betas, *mode)
    sparse_log_domain_actual = sparse_log_domain.evaluate(betas, *mode)
    assert_outputs_equal(expected, actual)
    assert_outputs_equal(expected, log_domain_actual)
    assert_outputs_equal(expected, sparse_log_domain_actual)


def test_sparse_expression_reports_structural_sparsity():
    elements = build_model(SparseLogCrossNested, explicit_mu=False)
    expression = elements.loglikelihood
    assert isinstance(expression, SparseLogCrossNested)
    assert expression.number_of_dense_memberships == 12
    assert expression.number_of_active_memberships == 6
