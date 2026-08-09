from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import biogeme.biogeme as biogeme_module
from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Draws, MonteCarlo, Variable, exp, log
from biogeme.results_processing import EstimationResults


def make_biogeme(**kwargs) -> BIOGEME:
    database = Database(
        'recovery',
        pd.DataFrame({'x': [-2.0, -1.0, 0.5, 1.0, 2.0]}),
    )
    beta = Beta('beta', 0.7, None, None, 0)
    expression = -((beta - Variable('x')) ** 2)
    result = BIOGEME(
        database,
        expression,
        optimization_algorithm='simple_bounds_BFGS',
        calculating_second_derivatives='analytical',
        generate_yaml=True,
        generate_html=False,
        save_iterations=False,
        **kwargs,
    )
    result.model_name = 'recovery'
    return result


def test_hessian_failure_leaves_resumable_optimization_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint = tmp_path / 'recovery.yaml'
    biogeme = make_biogeme()
    evaluator = biogeme.function_evaluator
    original_evaluate = evaluator.evaluate

    def fail_on_hessian(the_betas, gradient, hessian, bhhh):
        if hessian:
            raise RuntimeError('simulated Hessian failure')
        return original_evaluate(the_betas, gradient, hessian, bhhh)

    monkeypatch.setattr(evaluator, 'evaluate', fail_on_hessian)
    with pytest.raises(RuntimeError, match='simulated Hessian failure'):
        biogeme.estimate(yaml_file_name=str(checkpoint))

    partial = EstimationResults.from_yaml_file(filename=str(checkpoint))
    assert partial.raw_estimation_results.optimization_complete
    assert partial.raw_estimation_results.gradient_bhhh_complete
    assert not partial.raw_estimation_results.hessian_complete
    assert partial.hessian is None
    assert partial.gradient is not None
    assert partial.bhhh is not None
    with pytest.raises(BiogemeError, match='Second derivatives matrix not available'):
        _ = partial.rao_cramer_variance_covariance_matrix

    # Exercise the explicit warning for cross-session Monte Carlo recovery.
    partial.raw_estimation_results.monte_carlo = True
    partial.dump_yaml_file(filename=str(checkpoint))

    # Simulate a restarted process configured with the safer Hessian backend.
    biogeme.biogeme_parameters.set_value(
        'analytical_hessian_mode', 'chunked', section='Estimation'
    )
    biogeme.biogeme_parameters.set_value(
        'hessian_parameter_block_size', 1, section='Estimation'
    )
    biogeme.biogeme_parameters.set_value(
        'hessian_observation_batch_size', 2, section='Estimation'
    )
    biogeme._function_evaluator = None

    def optimization_must_not_run(*args, **kwargs):
        raise AssertionError('optimization was repeated')

    monkeypatch.setattr(biogeme_module, 'model_estimation', optimization_must_not_run)
    with caplog.at_level(logging.WARNING):
        completed = biogeme.estimate_or_load(yaml_file_name=str(checkpoint))
    assert biogeme.function_evaluator.analytical_hessian_mode == 'chunked'
    assert 'Exact reproduction of the optimization draws' in caplog.text
    assert completed.raw_estimation_results.hessian_complete
    assert completed.hessian is not None
    assert completed.raw_estimation_results.analytical_hessian_mode == 'chunked'
    assert completed.raw_estimation_results.hessian_parameter_block_size == 1
    assert completed.raw_estimation_results.hessian_observation_batch_size == 2

    reloaded = EstimationResults.from_yaml_file(filename=str(checkpoint))
    assert reloaded.raw_estimation_results.hessian_complete
    assert reloaded.hessian is not None


def test_bhhh_failure_preserves_optimum_and_resume_completes_all_derivatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / 'bhhh_failure.yaml'
    biogeme = make_biogeme()
    evaluator = biogeme.function_evaluator
    original_evaluate = evaluator.evaluate

    def fail_on_bhhh(the_betas, gradient, hessian, bhhh):
        if bhhh:
            raise RuntimeError('simulated BHHH failure')
        return original_evaluate(the_betas, gradient, hessian, bhhh)

    monkeypatch.setattr(evaluator, 'evaluate', fail_on_bhhh)
    with pytest.raises(RuntimeError, match='simulated BHHH failure'):
        biogeme.estimate(yaml_file_name=str(checkpoint))

    partial = EstimationResults.from_yaml_file(filename=str(checkpoint))
    assert partial.raw_estimation_results.optimization_complete
    assert not partial.raw_estimation_results.gradient_bhhh_complete
    assert not partial.raw_estimation_results.hessian_complete
    assert partial.gradient is None
    assert partial.bhhh is None
    assert partial.hessian is None

    monkeypatch.setattr(evaluator, 'evaluate', original_evaluate)
    monkeypatch.setattr(
        biogeme_module,
        'model_estimation',
        lambda *args, **kwargs: pytest.fail('optimization was repeated'),
    )
    completed = biogeme.estimate_or_load(yaml_file_name=str(checkpoint))
    assert completed.raw_estimation_results.gradient_bhhh_complete
    assert completed.raw_estimation_results.hessian_complete
    assert completed.gradient is not None
    assert completed.bhhh is not None
    assert completed.hessian is not None


def test_complete_checkpoint_is_loaded_without_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / 'complete.yaml'
    biogeme = make_biogeme()
    expected = biogeme.estimate(yaml_file_name=str(checkpoint))

    def evaluation_must_not_run(*args, **kwargs):
        raise AssertionError('post-processing was repeated')

    monkeypatch.setattr(biogeme.function_evaluator, 'evaluate', evaluation_must_not_run)
    actual = biogeme.estimate_or_load(yaml_file_name=str(checkpoint))
    assert actual.beta_values == pytest.approx(expected.beta_values)
    np.testing.assert_allclose(actual.hessian, expected.hessian)


def test_bootstrap_failure_can_resume_without_optimization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / 'bootstrap_failure.yaml'
    biogeme = make_biogeme()

    monkeypatch.setattr(
        biogeme,
        '_bootstrap',
        lambda estimated_parameters: (_ for _ in ()).throw(
            RuntimeError('simulated bootstrap failure')
        ),
    )
    with pytest.raises(RuntimeError, match='simulated bootstrap failure'):
        biogeme.estimate(run_bootstrap=True, yaml_file_name=str(checkpoint))

    partial = EstimationResults.from_yaml_file(filename=str(checkpoint))
    assert partial.raw_estimation_results.gradient_bhhh_complete
    assert partial.raw_estimation_results.hessian_complete
    assert not partial.raw_estimation_results.bootstrap_complete

    monkeypatch.setattr(
        biogeme,
        '_bootstrap',
        lambda estimated_parameters: [np.asarray(list(estimated_parameters.values()))],
    )
    monkeypatch.setattr(
        biogeme_module,
        'model_estimation',
        lambda *args, **kwargs: pytest.fail('optimization was repeated'),
    )
    completed = biogeme.estimate_or_load(yaml_file_name=str(checkpoint))
    assert completed.raw_estimation_results.bootstrap_complete
    assert len(completed.bootstrap) == 1


def test_configured_chunked_hessian_matches_full_hessian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BIOGEME, '_available_jax_memory', staticmethod(lambda: None))
    full = make_biogeme(analytical_hessian_mode='full')
    chunked = make_biogeme(
        analytical_hessian_mode='chunked',
        hessian_parameter_block_size=1,
        hessian_observation_batch_size=2,
    )
    betas = {'beta': 0.25}
    expected = full.function_evaluator.evaluate(betas, True, True, False)
    actual = chunked.function_evaluator.evaluate(betas, True, True, False)
    assert chunked.function_evaluator.analytical_hessian_mode == 'chunked'
    assert actual.function == pytest.approx(expected.function)
    np.testing.assert_allclose(actual.gradient, expected.gradient)
    np.testing.assert_allclose(actual.hessian, expected.hessian)


def test_unsafe_full_hessian_reports_chunked_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BIOGEME, '_available_jax_memory', staticmethod(lambda: 1024))
    biogeme = make_biogeme(
        analytical_hessian_mode='full',
        hessian_memory_fraction=0.25,
    )
    with pytest.raises(BiogemeError, match='Suggested configuration') as error:
        _ = biogeme.function_evaluator
    assert "analytical_hessian_mode = 'chunked'" in str(error.value)
    assert 'hessian_parameter_block_size' in str(error.value)
    assert 'hessian_observation_batch_size' in str(error.value)


def test_automatic_hessian_uses_conservative_draws_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BIOGEME, '_available_jax_memory', staticmethod(lambda: None))
    database = Database('automatic_draws', pd.DataFrame({'x': [-1.0, 0.5, 1.0]}))
    beta = Beta('beta', -0.5, None, None, 0)
    sigma = Beta('sigma', 0.7, None, None, 0)
    index = beta * Variable('x') + sigma * Draws('omega', 'NORMAL')
    expression = log(MonteCarlo(1.0 / (1.0 + exp(-index))))
    biogeme = BIOGEME(
        database,
        expression,
        number_of_draws=20,
        analytical_hessian_mode='automatic',
        hessian_parameter_block_size=1,
        hessian_observation_batch_size=2,
        generate_yaml=False,
        generate_html=False,
    )
    assert biogeme.function_evaluator.analytical_hessian_mode == 'chunked'


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('28000', 28_000 * 1024**2),
        ('7G', 7 * 1024**3),
        ('1.5 GB', int(1.5 * 1024**3)),
        ('invalid', None),
        ('0', None),
        (None, None),
    ],
)
def test_parse_slurm_memory(value: str | None, expected: int | None) -> None:
    assert BIOGEME._parse_slurm_memory(value) == expected


def test_slurm_per_node_memory_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '28000')
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '7000')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '16')
    assert BIOGEME._slurm_memory_allocation() == 28_000 * 1024**2


def test_slurm_per_cpu_memory_uses_allocated_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '7000')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '4')
    assert BIOGEME._slurm_memory_allocation() == 28_000 * 1024**2


def test_slurm_per_cpu_memory_is_conservative_without_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '7000')
    monkeypatch.delenv('SLURM_CPUS_ON_NODE', raising=False)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    assert BIOGEME._slurm_memory_allocation() == 7_000 * 1024**2


def test_effective_memory_honors_slurm_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeviceWithoutMemoryStatistics:
        @staticmethod
        def memory_stats() -> None:
            return None

    monkeypatch.setattr(
        biogeme_module.jax,
        'devices',
        lambda: [DeviceWithoutMemoryStatistics()],
    )
    # ``os.sysconf`` is POSIX-only; on Windows add the fake attribute so the
    # test exercises the same competing-memory path on every platform.
    monkeypatch.setattr(
        biogeme_module.os,
        'sysconf',
        lambda _: 10_000_000,
        raising=False,
    )
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '28000')
    assert BIOGEME._available_jax_memory() == 28_000 * 1024**2


def test_panel_checkpoint_accepts_flattened_database_name() -> None:
    biogeme = BIOGEME(
        Database(
            'panel_data',
            pd.DataFrame(
                {
                    'panel_id': [1, 1, 2],
                    'x': [1.0, 2.0, 3.0],
                }
            ),
        ),
        -Beta('beta', 0.7, None, None, 0) * Variable('x'),
        generate_yaml=False,
        generate_html=False,
    )
    biogeme.database.panel('panel_id')
    biogeme.use_flatten_database = True
    raw_results = SimpleNamespace(
        raw_estimation_results=SimpleNamespace(
            model_name=biogeme.model_name,
            data_name='flat panel_data',
            beta_names=['beta'],
        )
    )
    biogeme._validate_loaded_estimation_results(raw_results)
