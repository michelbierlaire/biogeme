"""Numerical robustness tests for the experimental log-domain CNL."""

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


def build_extreme_evaluator(expression_class, *, numerically_safe: bool):
    database = Database(
        'extreme_cnl',
        pd.DataFrame(
            {
                'x': [1000.0, -1000.0],
                'choice': [1, 2],
            }
        ),
    )
    beta = Beta('beta', 1.0, None, None, 0)
    mu1 = Beta('mu1', 1.2, 1.0, 4.0, 0)
    mu2 = Beta('mu2', 1.4, 1.0, 4.0, 0)
    utilities = {
        1: beta * Variable('x'),
        2: -beta * Variable('x'),
    }
    nests = NestsForCrossNestedLogit(
        choice_set=[1, 2],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=mu1,
                dict_of_alpha={1: 0.5, 2: 0.5},
                name='first',
            ),
            OneNestForCrossNestedLogit(
                nest_param=mu2,
                dict_of_alpha={1: 0.5, 2: 0.5},
                name='second',
            ),
        ),
    )
    expression = expression_class(
        util=utilities,
        av=None,
        nests=nests,
        choice=Variable('choice'),
    )
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


@pytest.mark.parametrize(
    'safe_expression_class', [LogCrossNested, SparseLogCrossNested]
)
def test_log_domain_remains_finite_for_extreme_utilities(
    safe_expression_class,
):
    betas = {'beta': 1.0, 'mu1': 1.2, 'mu2': 1.4}
    dense = build_extreme_evaluator(
        LogCrossNested, numerically_safe=False
    ).evaluate(
        betas, gradient=True, hessian=True, bhhh=False
    )
    log_domain = build_extreme_evaluator(
        safe_expression_class, numerically_safe=True
    ).evaluate(betas, gradient=True, hessian=True, bhhh=False)

    assert not np.isfinite(dense.function)
    assert np.isfinite(log_domain.function)
    assert np.all(np.isfinite(log_domain.gradient))
    assert np.all(np.isfinite(log_domain.hessian))


@pytest.mark.parametrize('expression_class', [LogCrossNested, SparseLogCrossNested])
def test_log_domain_handles_a_nest_with_no_available_alternative(expression_class):
    nests = NestsForCrossNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=1.2,
                dict_of_alpha={1: 1.0, 2: 0.0, 3: 0.0},
                name='unavailable',
            ),
            OneNestForCrossNestedLogit(
                nest_param=1.3,
                dict_of_alpha={1: 0.0, 2: 1.0, 3: 1.0},
                name='available',
            ),
        ),
    )
    expression = expression_class(
        util={1: 0.1, 2: 0.2, 3: 0.3},
        av={1: 0.0, 2: 1.0, 3: 1.0},
        nests=nests,
        choice=2,
    )
    assert np.isfinite(expression.get_value())
    jax_function = expression.recursive_construct_jax_function(
        numerically_safe=True
    )
    value = jax_function(
        np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    )
    assert np.isfinite(float(value))
