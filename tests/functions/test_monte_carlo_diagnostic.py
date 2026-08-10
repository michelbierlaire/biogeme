from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import (
    Beta,
    Draws,
    MonteCarlo,
    PanelLikelihoodTrajectory,
    Variable,
    exp,
    log,
)
from biogeme.monte_carlo_diagnostic import (
    MonteCarloDiagnosticConfiguration,
    MonteCarloDiagnosticRunner,
    atomic_write_yaml,
    build_draw_schedule,
    build_tasks,
    forecast_remaining_seconds,
    generate_diagnostic_draws,
)
from biogeme.results_processing import EstimationResults, RawEstimationResults


def configuration(
    *,
    factors: tuple[float, ...] = (1.0, 2.0),
    replications: int = 1,
    time_budget: float = 1000.0,
    objective_tolerance: float = 0.01,
    gradient_tolerance: float = 0.01,
) -> MonteCarloDiagnosticConfiguration:
    return MonteCarloDiagnosticConfiguration(
        draw_factors=factors,
        replications=replications,
        time_budget_seconds=time_budget,
        max_draws=1_000_000,
        safety_factor=1.5,
        objective_tolerance=objective_tolerance,
        gradient_tolerance=gradient_tolerance,
        minimum_level_factor=2.0,
    )


def baseline() -> dict[str, object]:
    return {
        'model_name': 'diagnostic_test',
        'original_number_of_draws': 10,
        'draw_types': {'omega': 'NORMAL'},
        'estimated_parameters': {'beta': 1.0, 'sigma': 0.5},
        'original_result': {'objective': -100.0, 'gradient': [0.0, 0.0]},
        'model_metadata': {
            'data_name': 'test',
            'sample_size': 3,
            'number_of_observations': 3,
            'parameter_names': ['beta', 'sigma'],
        },
    }


def runner(
    tmp_path: Path,
    evaluator,
    *,
    config: MonteCarloDiagnosticConfiguration | None = None,
    resume: bool = True,
) -> MonteCarloDiagnosticRunner:
    config = config or configuration()
    schedule = build_draw_schedule(10, config, {'omega': 'NORMAL'})
    tasks = build_tasks(schedule, config.replications, 10, {'omega': 'NORMAL'}, 7)
    return MonteCarloDiagnosticRunner(
        baseline=baseline(),
        configuration=config,
        planned_tasks=tasks,
        evaluate_task=evaluator,
        yaml_file=tmp_path / 'diagnostic.yaml',
        markdown_file=tmp_path / 'diagnostic.md',
        base_seed=7,
        resume=resume,
    )


def stable_evaluation(task: dict[str, object]) -> dict[str, object]:
    del task
    return {
        'objective': -100.0,
        'gradient': [0.0, 0.0],
        '_elapsed_seconds': 0.01,
    }


def raw_results(
    *,
    model_name: str,
    data_name: str,
    monte_carlo: bool,
    number_of_draws: int,
    draw_types: dict[str, str] | None,
    beta_names: list[str],
    beta_values: list[float],
    gradient: list[float],
    final_log_likelihood: float,
) -> EstimationResults:
    number_of_parameters = len(beta_names)
    raw = RawEstimationResults(
        model_name=model_name,
        user_notes='',
        beta_names=beta_names,
        beta_values=beta_values,
        lower_bounds=[None] * number_of_parameters,
        upper_bounds=[None] * number_of_parameters,
        gradient=gradient,
        hessian=None,
        bhhh=np.eye(number_of_parameters).tolist(),
        null_log_likelihood=-1.0,
        initial_log_likelihood=-1.0,
        final_log_likelihood=final_log_likelihood,
        data_name=data_name,
        sample_size=3,
        number_of_observations=3,
        monte_carlo=monte_carlo,
        number_of_draws=number_of_draws,
        types_of_draws=draw_types,
        number_of_excluded_data=0,
        draws_processing_time=timedelta(0),
        optimization_messages={},
        convergence=True,
        bootstrap=[],
        bootstrap_time=None,
    )
    return EstimationResults(raw)


def test_fast_first_schedule_and_antithetic_normalization() -> None:
    config = configuration(factors=(4.0, 0.25, 1.0, 0.5, 2.0))
    schedule = build_draw_schedule(10, config, {'omega': 'NORMAL_ANTI'})
    assert schedule == [2, 6, 10, 20, 40]
    assert all(draw_count % 2 == 0 for draw_count in schedule)


def test_runtime_forecast_uses_all_pending_replications() -> None:
    completed = [{'draw_count': 10, 'elapsed_seconds': 2.0}]
    pending = [{'draw_count': 20}, {'draw_count': 20}, {'draw_count': 40}]
    assert forecast_remaining_seconds(completed, pending, 1.5) == pytest.approx(24.0)


def test_antithetic_pairs_are_complete() -> None:
    draws, _, _ = generate_diagnostic_draws(
        draw_types={'omega': 'NORMAL_ANTI'},
        variable_names=['omega'],
        sample_size=3,
        number_of_draws=8,
        seed=11,
    )
    assert draws.shape == (3, 8, 1)
    np.testing.assert_allclose(draws[:, :4, 0], -draws[:, 4:, 0])


def test_mlhs_is_regenerated_as_a_valid_design() -> None:
    first, _, _ = generate_diagnostic_draws(
        draw_types={'omega': 'UNIFORM_MLHS'},
        variable_names=['omega'],
        sample_size=2,
        number_of_draws=6,
        seed=1,
    )
    second, _, _ = generate_diagnostic_draws(
        draw_types={'omega': 'UNIFORM_MLHS'},
        variable_names=['omega'],
        sample_size=2,
        number_of_draws=6,
        seed=2,
    )
    strata = np.floor(np.sort(first[:, :, 0], axis=None) * 12).astype(int)
    np.testing.assert_array_equal(strata, np.arange(12))
    assert not np.array_equal(first, second)


def test_randomized_halton_replications_are_distinct() -> None:
    first, methods, _ = generate_diagnostic_draws(
        draw_types={'omega': 'NORMAL_HALTON2'},
        variable_names=['omega'],
        sample_size=3,
        number_of_draws=8,
        seed=1,
    )
    second, _, _ = generate_diagnostic_draws(
        draw_types={'omega': 'NORMAL_HALTON2'},
        variable_names=['omega'],
        sample_size=3,
        number_of_draws=8,
        seed=2,
    )
    assert methods['omega'] == 'randomized_halton_modulo_one_shift'
    assert not np.array_equal(first, second)


def test_objective_and_gradient_comparison_and_separate_outputs(
    tmp_path: Path,
) -> None:
    standard_yaml = tmp_path / 'diagnostic_test.yaml'
    standard_yaml.write_text('original estimation result')

    def evaluator(task: dict[str, object]) -> dict[str, object]:
        del task
        return {
            'objective': -99.5,
            'gradient': [0.25, -0.5],
            '_elapsed_seconds': 0.0,
        }

    result = runner(tmp_path, evaluator, resume=False).run()
    first = result.data['completed_evaluations'][0]
    assert first['objective_difference'] == pytest.approx(0.5)
    assert first['relative_objective_difference'] == pytest.approx(0.005)
    assert first['gradient_difference'] == pytest.approx([0.25, -0.5])
    assert first['gradient_linf_difference'] == pytest.approx(0.5)
    assert first['gradient_l2_difference'] == pytest.approx(np.sqrt(0.3125))
    assert result.yaml_file != standard_yaml
    assert result.markdown_file.exists()
    assert standard_yaml.read_text() == 'original estimation result'


def test_time_budget_stops_before_next_task(tmp_path: Path) -> None:
    calls = 0

    def evaluator(task: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        del task
        calls += 1
        return {
            'objective': -100.0,
            'gradient': [0.0, 0.0],
            '_elapsed_seconds': 1.0,
        }

    config = configuration(factors=(0.5, 1.0, 2.0), time_budget=2.0)
    result = runner(tmp_path, evaluator, config=config, resume=False).run()
    assert calls == 1
    assert result.execution_status == 'time_budget_exceeded'
    assert len(result.data['skipped_evaluations']) == 2


def test_interrupted_but_conclusive(tmp_path: Path) -> None:
    diagnostic_runner: MonteCarloDiagnosticRunner

    def evaluator(task: dict[str, object]) -> dict[str, object]:
        if task['draw_count'] == 20:
            diagnostic_runner.request_stop()
        return stable_evaluation(task)

    diagnostic_runner = runner(tmp_path, evaluator, resume=False)
    result = diagnostic_runner.run()
    assert result.execution_status == 'interrupted'
    assert result.diagnostic_conclusion == 'stable'
    assert result.recommendation == 'no_more_draws_indicated'


def test_interrupted_and_inconclusive(tmp_path: Path) -> None:
    diagnostic_runner: MonteCarloDiagnosticRunner

    def evaluator(task: dict[str, object]) -> dict[str, object]:
        diagnostic_runner.request_stop()
        return stable_evaluation(task)

    diagnostic_runner = runner(tmp_path, evaluator, resume=False)
    result = diagnostic_runner.run()
    assert result.execution_status == 'interrupted'
    assert result.diagnostic_conclusion == 'inconclusive'
    assert result.recommendation == 'additional_diagnostics_needed'


def test_resume_does_not_repeat_completed_task(tmp_path: Path) -> None:
    first_runner: MonteCarloDiagnosticRunner

    def interrupting_evaluator(task: dict[str, object]) -> dict[str, object]:
        first_runner.request_stop()
        return stable_evaluation(task)

    first_runner = runner(tmp_path, interrupting_evaluator, resume=False)
    first_runner.run()

    resumed_calls: list[int] = []

    def resumed_evaluator(task: dict[str, object]) -> dict[str, object]:
        resumed_calls.append(int(task['draw_count']))
        return stable_evaluation(task)

    result = runner(tmp_path, resumed_evaluator, resume=True).run()
    assert resumed_calls == [20]
    assert len(result.data['completed_evaluations']) == 2
    assert result.execution_status == 'completed'


def test_atomic_checkpoint_preserves_previous_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / 'diagnostic.yaml'
    atomic_write_yaml(checkpoint, {'status': 'previous'})
    with patch(
        'biogeme.monte_carlo_diagnostic.os.replace',
        side_effect=OSError('replacement failed'),
    ):
        with pytest.raises(OSError, match='replacement failed'):
            atomic_write_yaml(checkpoint, {'status': 'new'})
    assert yaml.safe_load(checkpoint.read_text()) == {'status': 'previous'}
    assert list(tmp_path.glob('.diagnostic.yaml.*.tmp')) == []


def test_report_uses_american_english_and_documents_no_reestimation(
    tmp_path: Path,
) -> None:
    result = runner(tmp_path, stable_evaluation, resume=False).run()
    report = result.markdown_file.read_text()
    assert 'No re-estimation was performed.' in report
    assert 'ordinary estimation behavior' in report
    assert 'ordinary estimation behaviour' not in report
    assert 'No Hessian was requested.' in report
    assert 'The first Ctrl-C requests a graceful' in report
    assert 'a second Ctrl-C may' in report


def test_no_hessian_and_no_optimization_calls(tmp_path: Path) -> None:
    database = Database('diagnostic_data', pd.DataFrame({'x': [-1.0, 0.0, 1.0]}))
    beta = Beta('beta', 0.2, None, None, 0)
    omega = Draws('omega', 'NORMAL')
    expression = log(MonteCarlo(exp(-((Variable('x') - beta - 0.1 * omega) ** 2))))
    biogeme = BIOGEME(
        database,
        expression,
        number_of_draws=4,
        use_jit=False,
        generate_yaml=False,
        generate_html=False,
        monte_carlo_diagnostic_draw_factors='1,2',
        monte_carlo_diagnostic_replications=1,
        monte_carlo_diagnostic_objective_tolerance=1.0e9,
        monte_carlo_diagnostic_gradient_tolerance=1.0e9,
        monte_carlo_diagnostic_time_budget=1000,
        seed=5,
    )
    biogeme.model_name = 'diagnostic_model'
    results = raw_results(
        model_name='diagnostic_model',
        data_name='diagnostic_data',
        monte_carlo=True,
        number_of_draws=4,
        draw_types={'omega': 'NORMAL'},
        beta_names=['beta'],
        beta_values=[0.2],
        gradient=[0.0],
        final_log_likelihood=-1.0,
    )

    hessian_arguments: list[bool] = []
    original_evaluate = None

    from biogeme.monte_carlo_diagnostic import CompiledFormulaEvaluator

    original_evaluate = CompiledFormulaEvaluator.evaluate

    def recording_evaluate(self, *args, **kwargs):
        hessian_arguments.append(bool(kwargs.get('hessian')))
        return original_evaluate(self, *args, **kwargs)

    with (
        patch.object(CompiledFormulaEvaluator, 'evaluate', recording_evaluate),
        patch(
            'biogeme.biogeme.model_estimation',
            side_effect=AssertionError('optimization must not be called'),
        ),
    ):
        diagnostic = biogeme.check_monte_carlo_stability(
            estimation_results=results,
            output_directory=tmp_path,
            resume=False,
        )
    assert diagnostic.execution_status == 'completed'
    assert hessian_arguments and not any(hessian_arguments)


def test_not_applicable_without_monte_carlo_expression(tmp_path: Path) -> None:
    database = Database('plain_data', pd.DataFrame({'x': [-1.0, 0.0, 1.0]}))
    beta = Beta('beta', 0.2, None, None, 0)
    biogeme = BIOGEME(
        database,
        -((Variable('x') - beta) ** 2),
        generate_yaml=False,
        generate_html=False,
    )
    biogeme.model_name = 'plain_model_monte_carlo_diagnostic'
    results = raw_results(
        model_name='plain_model_monte_carlo_diagnostic',
        data_name='plain_data',
        monte_carlo=False,
        number_of_draws=10,
        draw_types=None,
        beta_names=['beta'],
        beta_values=[0.2],
        gradient=[0.0],
        final_log_likelihood=-1.0,
    )
    standard_yaml = tmp_path / 'plain_model_monte_carlo_diagnostic.yaml'
    standard_yaml.write_text('standard estimation')
    result = biogeme.check_monte_carlo_stability(
        estimation_results=results,
        output_directory=tmp_path,
        resume=False,
    )
    assert result.execution_status == 'completed'
    assert result.diagnostic_conclusion == 'not_applicable'
    assert result.recommendation == 'not_applicable'
    assert result.yaml_file.name == (
        'plain_model_monte_carlo_diagnostic_monte_carlo_diagnostic.yaml'
    )
    assert standard_yaml.read_text() == 'standard estimation'


def test_postprocessing_panel_model_uses_flattened_database(tmp_path: Path) -> None:
    database = Database(
        'panel_data',
        pd.DataFrame({'person': [1, 1, 2, 2], 'x': [-1.0, 0.0, 1.0, 2.0]}),
    )
    database.panel('person')
    beta = Beta('beta', 0.2, None, None, 0)
    expression = log(PanelLikelihoodTrajectory(exp(-((Variable('x') - beta) ** 2))))
    biogeme = BIOGEME(
        database,
        expression,
        generate_yaml=False,
        generate_html=False,
    )
    biogeme.model_name = 'panel_model'
    results = raw_results(
        model_name='panel_model',
        data_name='panel_data',
        monte_carlo=False,
        number_of_draws=10,
        draw_types=None,
        beta_names=['beta'],
        beta_values=[0.2],
        gradient=[0.0],
        final_log_likelihood=-1.0,
    )

    diagnostic = biogeme.check_monte_carlo_stability(
        estimation_results=results,
        output_directory=tmp_path,
        resume=False,
    )

    assert biogeme.use_flatten_database is True
    assert diagnostic.diagnostic_conclusion == 'not_applicable'


def test_automatic_diagnostic_is_disabled_by_default() -> None:
    database = Database('plain_data', pd.DataFrame({'x': [0.0]}))
    beta = Beta('beta', 0.2, None, None, 0)
    biogeme = BIOGEME(
        database,
        -((Variable('x') - beta) ** 2),
        generate_yaml=False,
        generate_html=False,
    )
    results = raw_results(
        model_name='plain_model',
        data_name='plain_data',
        monte_carlo=False,
        number_of_draws=10,
        draw_types=None,
        beta_names=['beta'],
        beta_values=[0.2],
        gradient=[0.0],
        final_log_likelihood=-1.0,
    )
    with patch.object(biogeme, 'check_monte_carlo_stability') as diagnostic:
        returned = biogeme._run_automatic_monte_carlo_diagnostic(results)

    assert returned is results
    diagnostic.assert_not_called()

    biogeme.biogeme_parameters.set_value('monte_carlo_diagnostic_auto', True)
    with patch.object(biogeme, 'check_monte_carlo_stability') as diagnostic:
        returned = biogeme._run_automatic_monte_carlo_diagnostic(results)

    assert returned is results
    diagnostic.assert_called_once_with(estimation_results=results)
