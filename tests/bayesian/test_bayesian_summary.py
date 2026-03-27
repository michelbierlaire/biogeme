from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import yaml
from biogeme.bayesian_estimation.bayesian_results_summary import (
    BayesianResultsSummary,
    EstimatedBetaSummary,
)
from biogeme.exceptions import BiogemeError


class DummyDimension(Enum):
    OBS = 'obs'
    DRAW = 'draw'


def _make_estimated_beta_summary(name: str, mean: float) -> EstimatedBetaSummary:
    return EstimatedBetaSummary(
        name=name,
        mean=mean,
        median=mean + 0.1,
        mode=mean - 0.1,
        std_err=0.25,
        z_value=mean / 0.25,
        p_value=0.05,
        hdi_low=mean - 0.5,
        hdi_high=mean + 0.5,
        rhat=1.0,
        effective_sample_size_bulk=800.0,
        effective_sample_size_tail=600.0,
    )


def _make_bayesian_results_summary() -> BayesianResultsSummary:
    parameters = {
        'beta_1': _make_estimated_beta_summary('beta_1', 1.5),
        'beta_2': _make_estimated_beta_summary('beta_2', -2.0),
        'aux_1': _make_estimated_beta_summary('aux_1', 0.25),
    }
    return BayesianResultsSummary(
        model_name='my_bayesian_model',
        data_name='my_dataset',
        chains=4,
        draws=250,
        hdi_prob=0.94,
        calculate_likelihood=True,
        calculate_waic=True,
        calculate_loo=True,
        beta_names=['beta_1', 'beta_2'],
        parameters=parameters,
        array_metadata={
            'log_like': {
                'dims': ['chain', DummyDimension.DRAW, DummyDimension.OBS],
                'shape': [4, 250, 10],
                'dtype': 'float64',
                'example_value': np.float64(0.5),
            }
        },
        posterior_predictive_loglike=-123.45,
        expected_log_likelihood=-120.0,
        best_draw_log_likelihood=-110.0,
        waic=245.2,
        waic_se=10.4,
        p_waic=12.3,
        loo=247.8,
        loo_se=11.1,
        p_loo=13.4,
        sampler='numpyro',
        target_accept=0.9,
        run_time='12.5 seconds',
        number_of_observations=10,
        user_notes=['first note', 'second note'],
        stored_variables_report=[
            {
                'group': 'posterior',
                'variable': 'beta_1',
                'dims': ['chain', 'draw'],
                'shape': [4, 250],
            },
            {
                'group': 'log_likelihood',
                'variable': 'log_like',
                'dims': ['chain', 'draw', 'obs'],
                'shape': [4, 250, 10],
            },
        ],
        identification_diagnostics_summary={
            'has_prior': True,
            'posterior': {
                'condition_number': np.float64(12.5),
                'effective_rank': np.float64(2.0),
            },
            'prior': {
                'condition_number': np.float64(8.0),
            },
            'per_parameter': [
                {
                    'parameter': 'beta_1',
                    'posterior_std': np.float64(0.25),
                    'prior_std': np.float64(1.0),
                },
                {
                    'parameter': 'beta_2',
                    'posterior_std': np.float64(0.5),
                    'prior_std': np.float64(1.5),
                },
            ],
            'flags': ['weak_identification'],
            'posterior_near_null_direction': [('beta_1', np.float64(0.7))],
            'prior_near_null_direction': [('beta_2', np.float64(-0.4))],
        },
        diagnostic_figure_references={
            'trace': 'trace.png',
            'rank': 'rank.png',
            'energy': 'energy.png',
            'autocorr': 'autocorr.png',
        },
    )


def test_estimated_beta_summary_to_dict_from_dict_round_trip() -> None:
    beta = _make_estimated_beta_summary('beta_x', 3.0)
    payload = beta.to_dict()
    rebuilt = EstimatedBetaSummary.from_dict(payload)

    assert rebuilt == beta
    assert rebuilt.name == 'beta_x'
    assert rebuilt.mean == 3.0


def test_bayesian_results_summary_basic_accessors() -> None:
    summary = _make_bayesian_results_summary()

    assert summary.posterior_draws == 1000
    assert set(summary.parameter_estimates()) == {'beta_1', 'beta_2'}
    assert set(summary.other_variables()) == {'aux_1'}
    assert summary.list_array_variables()['log_like']['shape'] == [4, 250, 10]
    assert summary.get_user_notes() == ['first note', 'second note']
    assert summary.get_identification_diagnostics_summary() is not None
    assert summary.get_diagnostic_figure_references()['trace'] == 'trace.png'


def test_bayesian_results_summary_report_stored_variables_dataframe() -> None:
    summary = _make_bayesian_results_summary()
    df = summary.report_stored_variables()

    assert list(df.columns) == ['group', 'variable', 'dims', 'shape']
    assert len(df) == 2
    assert set(df['variable']) == {'beta_1', 'log_like'}


def test_bayesian_results_summary_generate_general_information_and_short_summary() -> (
    None
):
    summary = _make_bayesian_results_summary()
    info = summary.generate_general_information()
    text = summary.short_summary()

    assert info['Sample size'] == 10
    assert info['Sampler'] == 'numpyro'
    assert info['Number of chains'] == 4
    assert 'Posterior predictive log-likelihood (sum of log mean p)' in info
    assert 'WAIC (Widely Applicable Information Criterion)' in info
    assert 'LOO (Leave-One-Out Cross-Validation)' in info
    assert 'Sample size' in text
    assert 'numpyro' in text


def test_bayesian_results_summary_get_beta_values() -> None:
    summary = _make_bayesian_results_summary()

    assert summary.get_beta_values() == {'beta_1': 1.5, 'beta_2': -2.0}
    assert summary.get_beta_values(['beta_2']) == {'beta_2': -2.0}


def test_bayesian_results_summary_get_beta_values_unknown_parameter_raises() -> None:
    summary = _make_bayesian_results_summary()

    with pytest.raises(BiogemeError):
        summary.get_beta_values(['unknown_parameter'])


def test_bayesian_results_summary_to_dict_converts_numpy_scalars_to_plain_python() -> (
    None
):
    summary = _make_bayesian_results_summary()
    payload = summary.to_dict()

    assert isinstance(payload['array_metadata']['log_like']['example_value'], float)
    assert payload['array_metadata']['log_like']['dims'] == ['chain', 'draw', 'obs']
    assert isinstance(
        payload['identification_diagnostics_summary']['posterior']['condition_number'],
        float,
    )
    assert isinstance(
        payload['identification_diagnostics_summary']['per_parameter'][0][
            'posterior_std'
        ],
        float,
    )
    assert isinstance(
        payload['identification_diagnostics_summary']['posterior_near_null_direction'][
            0
        ][1],
        float,
    )


def test_bayesian_results_summary_dump_yaml_and_load_round_trip(tmp_path: Path) -> None:
    summary = _make_bayesian_results_summary()
    yaml_file = tmp_path / 'bayesian_summary.yaml'

    summary.dump_yaml(str(yaml_file))
    rebuilt = BayesianResultsSummary.from_yaml_file(str(yaml_file))

    assert yaml_file.exists()
    assert rebuilt.model_name == summary.model_name
    assert rebuilt.data_name == summary.data_name
    assert rebuilt.chains == summary.chains
    assert rebuilt.draws == summary.draws
    assert rebuilt.hdi_prob == summary.hdi_prob
    assert rebuilt.calculate_likelihood == summary.calculate_likelihood
    assert rebuilt.calculate_waic == summary.calculate_waic
    assert rebuilt.calculate_loo == summary.calculate_loo
    assert rebuilt.beta_names == summary.beta_names
    assert rebuilt.user_notes == summary.user_notes
    assert rebuilt.sampler == summary.sampler
    assert rebuilt.target_accept == summary.target_accept
    assert rebuilt.run_time == summary.run_time
    assert rebuilt.number_of_observations == summary.number_of_observations

    assert rebuilt.parameters['beta_1'] == summary.parameters['beta_1']
    assert rebuilt.parameters['beta_2'] == summary.parameters['beta_2']
    assert rebuilt.parameters['aux_1'] == summary.parameters['aux_1']

    assert rebuilt.array_metadata == summary.to_dict()['array_metadata']
    assert rebuilt.array_metadata['log_like']['dims'] == ['chain', 'draw', 'obs']
    assert (
        rebuilt.stored_variables_report == summary.to_dict()['stored_variables_report']
    )
    assert (
        rebuilt.identification_diagnostics_summary
        == summary.to_dict()['identification_diagnostics_summary']
    )
    assert rebuilt.diagnostic_figure_references == summary.diagnostic_figure_references


def test_bayesian_results_summary_yaml_round_trip_preserves_serialized_payload(
    tmp_path: Path,
) -> None:
    """Verify that YAML round-trip preserves the exact serialized information.

    This test compares the normalized dictionary produced by ``to_dict()``
    before dumping with the normalized dictionary obtained after reloading from
    YAML. It therefore also covers values that are transformed in order to be
    serialized, such as NumPy scalars and Enum members.
    """
    summary = _make_bayesian_results_summary()
    yaml_file = tmp_path / 'bayesian_summary_exact_round_trip.yaml'

    original_payload = summary.to_dict()
    summary.dump_yaml(str(yaml_file))
    rebuilt = BayesianResultsSummary.from_yaml_file(str(yaml_file))
    rebuilt_payload = rebuilt.to_dict()

    assert rebuilt_payload == original_payload


def test_bayesian_results_summary_yaml_file_contains_serializable_plain_data(
    tmp_path: Path,
) -> None:
    summary = _make_bayesian_results_summary()
    yaml_file = tmp_path / 'bayesian_summary.yaml'

    summary.dump_yaml(str(yaml_file))

    with open(yaml_file, 'r', encoding='utf-8') as f:
        loaded = yaml.safe_load(f)

    assert loaded['model_name'] == 'my_bayesian_model'
    assert loaded['parameters']['beta_1']['mean'] == 1.5
    assert loaded['array_metadata']['log_like']['dims'] == ['chain', 'draw', 'obs']
    assert (
        loaded['identification_diagnostics_summary']['posterior']['condition_number']
        == 12.5
    )
    assert loaded['diagnostic_figure_references']['trace'] == 'trace.png'
    assert loaded['user_notes'] == ['first note', 'second note']


def test_bayesian_results_summary_empty_reports_and_accessors_are_safe() -> None:
    summary = BayesianResultsSummary(
        model_name='empty_model',
        data_name='empty_data',
        chains=1,
        draws=10,
        hdi_prob=0.9,
        calculate_likelihood=False,
        calculate_waic=False,
        calculate_loo=False,
        beta_names=[],
        parameters={},
        array_metadata={},
    )

    assert summary.get_user_notes() == []
    assert summary.get_diagnostic_figure_references() == {}
    assert summary.get_identification_diagnostics_summary() is None
    assert summary.report_stored_variables().empty
    assert summary.parameter_estimates() == {}
    assert summary.other_variables() == {}
