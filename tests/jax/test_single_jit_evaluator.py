"""Regression tests for the single-boundary JAX evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biogeme.database import Database
from biogeme.expressions import Beta, Draws, MonteCarlo, Variable, exp, log
from biogeme.jax_calculator import CompiledFormulaEvaluator
from biogeme.model_elements import ModelElements, RegularAdapter
from biogeme.models import logcnl
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
from biogeme.second_derivatives import SecondDerivativesMode


def _assert_equivalent(reference, candidate) -> None:
    assert candidate.function == pytest.approx(reference.function, rel=1e-11, abs=1e-11)
    for field in ('gradient', 'hessian', 'bhhh'):
        expected = getattr(reference, field)
        actual = getattr(candidate, field)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def _cnl_model_elements(*, use_jit: bool) -> ModelElements:
    dataframe = pd.DataFrame(
        {
            'time_1': [1.0, 1.5, 0.8, 2.0, 1.1, 1.7],
            'time_2': [1.4, 0.7, 1.8, 1.2, 1.6, 0.9],
            'time_3': [0.9, 1.3, 1.1, 1.7, 0.8, 1.5],
            'av_1': [1, 1, 1, 1, 1, 1],
            'av_2': [1, 1, 0, 1, 1, 1],
            'av_3': [1, 1, 1, 1, 0, 1],
            'choice': [1, 2, 3, 1, 2, 3],
            'weight': [1.0, 0.5, 1.5, 1.0, 0.8, 1.2],
        }
    )
    database = Database('experimental_cnl', dataframe)

    asc_1 = Beta('asc_1', 0.0, None, None, 0)
    asc_2 = Beta('asc_2', 0.0, None, None, 0)
    beta_time = Beta('beta_time', -1.0, None, None, 0)
    mu_1 = Beta('mu_1', 1.2, 1.0, 5.0, 0)
    mu_2 = Beta('mu_2', 1.3, 1.0, 5.0, 0)
    alpha = Beta('alpha', 0.4, 0.01, 0.99, 0)

    utilities = {
        1: asc_1 + beta_time * Variable('time_1'),
        2: asc_2 + beta_time * Variable('time_2'),
        3: beta_time * Variable('time_3'),
    }
    availability = {
        1: Variable('av_1'),
        2: Variable('av_2'),
        3: Variable('av_3'),
    }
    nests = NestsForCrossNestedLogit(
        choice_set=[1, 2, 3],
        tuple_of_nests=(
            OneNestForCrossNestedLogit(
                nest_param=mu_1,
                dict_of_alpha={1: alpha, 2: 0.0, 3: 1.0},
                name='first',
            ),
            OneNestForCrossNestedLogit(
                nest_param=mu_2,
                dict_of_alpha={1: 1.0 - alpha, 2: 1.0, 3: 0.0},
                name='second',
            ),
        ),
    )
    expression = logcnl(utilities, availability, nests, Variable('choice'))
    return ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=Variable('weight'),
        adapter=RegularAdapter(database=database),
        use_jit=use_jit,
    )


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
            'asc_1': 0.0,
            'asc_2': 0.0,
            'beta_time': -1.0,
            'mu_1': 1.2,
            'mu_2': 1.3,
            'alpha': 0.4,
        },
        {
            'asc_1': 0.7,
            'asc_2': -0.3,
            'beta_time': -2.1,
            'mu_1': 1.05,
            'mu_2': 2.2,
            'alpha': 0.15,
        },
        {
            'asc_1': -1.1,
            'asc_2': 0.9,
            'beta_time': -0.25,
            'mu_1': 3.0,
            'mu_2': 1.1,
            'alpha': 0.85,
        },
    ],
)
def test_jitted_cnl_matches_unjitted_cnl(betas, mode):
    reference_elements = _cnl_model_elements(use_jit=False)
    jitted_elements = _cnl_model_elements(use_jit=True)
    reference = CompiledFormulaEvaluator(
        model_elements=reference_elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    jitted = CompiledFormulaEvaluator(
        model_elements=jitted_elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    gradient, hessian, bhhh = mode
    expected = reference.evaluate(betas, gradient, hessian, bhhh)
    actual = jitted.evaluate(betas, gradient, hessian, bhhh)
    _assert_equivalent(expected, actual)


@pytest.mark.parametrize('use_jit', [False, True])
@pytest.mark.parametrize(
    ('block_size', 'observation_batch_size'),
    [(1, None), (2, 2), (4, 4), (20, 3)],
)
def test_chunked_hessian_matches_full_hessian(
    use_jit, block_size, observation_batch_size
):
    elements = _cnl_model_elements(use_jit=use_jit)
    evaluator = CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    betas = {
        'asc_1': 0.3,
        'asc_2': -0.2,
        'beta_time': -1.4,
        'mu_1': 1.25,
        'mu_2': 1.5,
        'alpha': 0.35,
    }

    full = evaluator.evaluate(betas, gradient=True, hessian=True, bhhh=False)
    chunked = evaluator.evaluate_chunked_hessian(
        betas,
        block_size=block_size,
        observation_batch_size=observation_batch_size,
    )
    _assert_equivalent(full, chunked)
    np.testing.assert_allclose(
        chunked.hessian, chunked.hessian.T, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize('block_size', [0, -1, 1.5, True])
def test_chunked_hessian_rejects_invalid_block_size(block_size):
    evaluator = CompiledFormulaEvaluator(
        model_elements=_cnl_model_elements(use_jit=True),
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    with pytest.raises(Exception, match='block size'):
        evaluator.evaluate_chunked_hessian({}, block_size=block_size)


@pytest.mark.parametrize('batch_size', [0, -1, 1.5, True])
def test_chunked_hessian_rejects_invalid_observation_batch_size(batch_size):
    evaluator = CompiledFormulaEvaluator(
        model_elements=_cnl_model_elements(use_jit=True),
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=False,
    )
    with pytest.raises(Exception, match='Observation batch size'):
        evaluator.evaluate_chunked_hessian(
            {}, block_size=1, observation_batch_size=batch_size
        )


@pytest.mark.parametrize(
    ('block_size', 'observation_batch_size'), [(1, None), (2, 2)]
)
def test_chunked_hessian_matches_full_hessian_with_monte_carlo(
    block_size, observation_batch_size
):
    database = Database(
        'monte_carlo_hessian',
        pd.DataFrame({'x': [-1.2, -0.4, 0.3, 0.9, 1.7]}),
    )
    beta = Beta('beta', -0.7, None, None, 0)
    sigma = Beta('sigma', 0.8, None, None, 0)
    latent_index = beta * Variable('x') + sigma * Draws('omega', 'NORMAL')
    expression = log(MonteCarlo(1.0 / (1.0 + exp(-latent_index))))
    elements = ModelElements.from_expression_and_weight(
        log_like=expression,
        weight=None,
        adapter=RegularAdapter(database=database),
        number_of_draws=20,
        use_jit=True,
    )
    evaluator = CompiledFormulaEvaluator(
        model_elements=elements,
        second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
        numerically_safe=True,
    )
    betas = {'beta': -0.6, 'sigma': 0.75}

    full = evaluator.evaluate(betas, gradient=True, hessian=True, bhhh=False)
    chunked = evaluator.evaluate_chunked_hessian(
        betas,
        block_size=block_size,
        observation_batch_size=observation_batch_size,
    )
    _assert_equivalent(full, chunked)
