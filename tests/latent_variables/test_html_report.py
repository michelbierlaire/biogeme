from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from biogeme.results_processing.estimation_results import EstimateVarianceCovariance
from biogeme.results_processing.html_output import (
    _get_html_estimated_parameters,
    format_real_number,
    generate_html_file,
    get_html_condition_number,
    get_html_correlation_results,
    get_html_estimated_parameters,
    get_html_footer,
    get_html_general_statistics,
    get_html_header,
    get_html_one_pair_of_parameters,
    get_html_one_parameter,
    get_html_preamble,
)


class FakeEstimationResults:
    def __init__(self):
        self.beta_names = ['beta_time', 'beta_cost', 'asc_car']
        self.raw_estimation_results = SimpleNamespace(
            model_name='fake_model',
            data_name='fake_data',
        )
        self.algorithm_has_converged = True
        self.smallest_eigenvalue = 0.5
        self.largest_eigenvalue = 10.0
        self.smallest_eigenvector = np.array([0.0, 0.0, 0.0])
        self.condition_number = 20.0
        self.user_notes = None
        self.bootstrap_time = None
        self.optimization_messages = {
            'Algorithm': 'fake algorithm',
            'Relative gradient': 1.23456789e-5,
        }

    def get_default_variance_covariance_matrix(self):
        return EstimateVarianceCovariance.ROBUST

    def get_general_statistics(self):
        return {
            'Number of estimated parameters': 3,
            'Final log likelihood': -123.456,
            'Null log likelihood': None,
        }

    def get_parameter_value_from_index(self, parameter_index):
        return [-1.23456, -2.34567, 0.45678][parameter_index]

    def get_parameter_std_err_from_index(self, parameter_index, estimate_var_covar):
        return [0.1, 0.2, 0.3][parameter_index]

    def get_parameter_t_test_from_index(
        self, parameter_index, estimate_var_covar, target
    ):
        value = self.get_parameter_value_from_index(parameter_index)
        std_err = self.get_parameter_std_err_from_index(
            parameter_index, estimate_var_covar
        )
        return (value - target) / std_err

    def get_parameter_p_value_from_index(
        self, parameter_index, estimate_var_covar, target
    ):
        return 0.05

    def is_bound_active(self, parameter_name):
        return parameter_name == 'beta_cost'

    def is_any_bound_active(self):
        return True

    def get_parameter_index(self, parameter_name):
        return self.beta_names.index(parameter_name)

    def get_variance_covariance_matrix(self, variance_covariance_type):
        return np.array(
            [
                [1.0, 0.2, 0.3],
                [0.2, 4.0, 0.4],
                [0.3, 0.4, 9.0],
            ]
        )

    def calculate_test(self, first_parameter_index, second_parameter_index, covariance):
        return 1.5


@pytest.fixture
def results():
    return FakeEstimationResults()


def test_format_real_number():
    assert format_real_number(1234.567) == '1.23e+03'
    assert format_real_number(0.00123456) == '0.00123'
    assert format_real_number(-2.34567) == '-2.35'


def test_get_html_header(results):
    html = get_html_header(results)

    assert '<html>' in html
    assert '<head>' in html
    assert '<body bgcolor="#ffffff">' in html
    assert 'fake_model' in html
    assert 'biogeme' in html


def test_get_html_header_without_raw_results(results):
    results.raw_estimation_results = None

    html = get_html_header(results)

    assert 'No estimation result is available' in html


def test_get_html_footer():
    assert get_html_footer() == '</body>\n</html>'


def test_get_html_preamble(results):
    html = get_html_preamble(results, file_name='report.html')

    assert 'report.html' in html
    assert 'fake_data' in html


def test_get_html_preamble_without_raw_results(results):
    results.raw_estimation_results = None

    html = get_html_preamble(results, file_name='report.html')

    assert 'No estimation result is available' in html


def test_get_html_preamble_non_convergence(results):
    results.algorithm_has_converged = False

    html = get_html_preamble(results, file_name='report.html')

    assert 'Algorithm failed to converge' in html


def test_get_html_preamble_user_notes(results):
    results.user_notes = 'Important note.'

    html = get_html_preamble(results, file_name='report.html')

    assert 'Important note.' in html
    assert '<blockquote' in html


def test_get_html_general_statistics(results):
    html = get_html_general_statistics(results)

    assert 'Number of estimated parameters' in html
    assert 'Final log likelihood' in html
    assert 'Null log likelihood' not in html
    assert 'Relative gradient' in html


def test_get_html_one_parameter(results):
    html = get_html_one_parameter(
        estimation_results=results,
        parameter_index=1,
        variance_covariance_type=EstimateVarianceCovariance.ROBUST,
    )

    assert '<tr class=biostyle>' in html
    assert 'beta_cost' in html
    assert 'Active bound' in html


def test_get_html_one_parameter_with_custom_name_and_number(results):
    html = get_html_one_parameter(
        estimation_results=results,
        parameter_index=0,
        variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        parameter_number=99,
        parameter_name='renamed_time',
    )

    assert '<td>99</td>' in html
    assert 'renamed_time' in html


def test_get_html_one_parameter_invalid_index(results):
    with pytest.raises(ValueError, match='Invalid parameter index'):
        get_html_one_parameter(
            estimation_results=results,
            parameter_index=99,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )


def test_get_html_estimated_parameters_without_groups(results):
    tables = get_html_estimated_parameters(estimation_results=results)

    assert list(tables.keys()) == ['']
    assert 'beta_time' in tables['']
    assert 'beta_cost' in tables['']
    assert 'asc_car' in tables['']


def test_get_html_estimated_parameters_with_groups(results):
    tables = get_html_estimated_parameters(
        estimation_results=results,
        group_of_parameters={
            'Taste parameters': ['beta_time', 'beta_cost'],
        },
    )

    assert 'Taste parameters' in tables
    assert 'Other parameters' in tables
    assert 'beta_time' in tables['Taste parameters']
    assert 'beta_cost' in tables['Taste parameters']
    assert 'asc_car' in tables['Other parameters']


def test_get_html_estimated_parameters_parameter_in_several_groups(results):
    tables = get_html_estimated_parameters(
        estimation_results=results,
        group_of_parameters={
            'Group 1': ['beta_time', 'beta_cost'],
            'Group 2': ['beta_cost', 'asc_car'],
        },
    )

    assert 'Group 1' in tables
    assert 'Group 2' in tables
    assert 'Other parameters' not in tables
    assert 'beta_cost' in tables['Group 1']
    assert 'beta_cost' in tables['Group 2']


def test_get_html_estimated_parameters_existing_other_group(results):
    tables = get_html_estimated_parameters(
        estimation_results=results,
        group_of_parameters={
            'Other parameters': ['beta_time'],
        },
    )

    assert 'Other parameters' in tables
    assert 'Other parameters not in groups' in tables


def test_get_html_estimated_parameters_unknown_parameter(results):
    with pytest.raises(ValueError, match='Unknown parameters requested'):
        get_html_estimated_parameters(
            estimation_results=results,
            group_of_parameters={'Bad group': ['unknown_beta']},
        )


def test_get_html_estimated_parameters_with_renaming(results):
    tables = get_html_estimated_parameters(
        estimation_results=results,
        renaming_parameters={'beta_time': 'travel_time'},
    )

    assert 'travel_time' in tables['']


def test_get_html_estimated_parameters_sort_by_name(results):
    table = _get_html_estimated_parameters(
        estimation_results=results,
        selected_parameters=['beta_time', 'beta_cost', 'asc_car'],
        sort_by_name=True,
    )

    assert table.index('asc_car') < table.index('beta_cost') < table.index('beta_time')


def test_get_html_one_pair_of_parameters(results):
    html = get_html_one_pair_of_parameters(
        estimation_results=results,
        first_parameter_index=1,
        second_parameter_index=0,
        variance_covariance_type=EstimateVarianceCovariance.ROBUST,
    )

    assert 'beta_cost' in html
    assert 'beta_time' in html
    assert '<tr class=biostyle>' in html


def test_get_html_one_pair_of_parameters_invalid_first_index(results):
    with pytest.raises(ValueError, match='Invalid parameter index'):
        get_html_one_pair_of_parameters(
            estimation_results=results,
            first_parameter_index=99,
            second_parameter_index=0,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )


def test_get_html_one_pair_of_parameters_invalid_second_index(results):
    with pytest.raises(ValueError, match='Invalid parameter index'):
        get_html_one_pair_of_parameters(
            estimation_results=results,
            first_parameter_index=0,
            second_parameter_index=99,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )


def test_get_html_correlation_results(results):
    html = get_html_correlation_results(results)

    assert 'Coefficient 1' in html
    assert 'Coefficient 2' in html
    assert 'beta_cost' in html
    assert 'beta_time' in html


def test_get_html_correlation_results_with_involved_parameters(results):
    html = get_html_correlation_results(
        estimation_results=results,
        involved_parameters={
            'beta_time': 'time',
            'beta_cost': 'cost',
        },
    )

    assert 'time' in html
    assert 'cost' in html
    assert 'asc_car' not in html


def test_get_html_condition_number(results):
    html = get_html_condition_number(results)

    assert 'Smallest eigenvalue' in html
    assert 'Largest eigenvalue' in html
    assert 'Condition number' in html
    assert '20' in html


def test_generate_html_file(results, tmp_path):
    output_file = tmp_path / 'report.html'

    generate_html_file(
        estimation_results=results,
        filename=str(output_file),
        overwrite=False,
    )

    html = output_file.read_text()

    assert '<h1>Estimation report</h1>' in html
    assert '<h1>Estimated parameters</h1>' in html
    assert 'beta_time' in html
    assert 'beta_cost' in html
    assert '<h2>Correlation of coefficients</h2>' in html


def test_generate_html_file_with_parameter_groups(results, tmp_path):
    output_file = tmp_path / 'grouped_report.html'

    generate_html_file(
        estimation_results=results,
        filename=str(output_file),
        overwrite=False,
        group_of_parameters={
            'Taste parameters': ['beta_time', 'beta_cost'],
        },
    )

    html = output_file.read_text()

    assert '<h3>Taste parameters</h3>' in html
    assert '<h3>Other parameters</h3>' in html
    assert 'asc_car' in html


def test_generate_html_file_existing_file_raises(results, tmp_path):
    output_file = tmp_path / 'report.html'
    output_file.write_text('already exists')

    with pytest.raises(FileExistsError):
        generate_html_file(
            estimation_results=results,
            filename=str(output_file),
            overwrite=False,
        )


def test_generate_html_file_existing_file_overwrite(results, tmp_path):
    output_file = tmp_path / 'report.html'
    output_file.write_text('already exists')

    generate_html_file(
        estimation_results=results,
        filename=str(output_file),
        overwrite=True,
    )

    html = output_file.read_text()

    assert 'already exists' not in html
    assert '<h1>Estimation report</h1>' in html


def test_generate_html_file_bootstrap_fallback(results, tmp_path):
    output_file = tmp_path / 'bootstrap_report.html'

    generate_html_file(
        estimation_results=results,
        filename=str(output_file),
        overwrite=False,
        variance_covariance_type=EstimateVarianceCovariance.BOOTSTRAP,
    )

    html = output_file.read_text()

    assert '<h1>Estimation report</h1>' in html
