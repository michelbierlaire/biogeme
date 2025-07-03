import os
import unittest
from datetime import timedelta

from biogeme.results_processing import (
    EstimateVarianceCovariance,
    EstimationResults,
    RawEstimationResults,
    format_real_number,
    generate_html_file,
    get_html_condition_number,
    get_html_correlation_results,
    get_html_estimated_parameters,
    get_html_footer,
    get_html_general_statistics,
    get_html_header,
    get_html_preamble,
)
from biogeme.version import get_version, versionDate


class TestHTMLGeneration(unittest.TestCase):

    def setUp(self):
        # Setting up real estimation results using RawEstimationResults
        raw_estimation = RawEstimationResults(
            model_name='test_model',
            user_notes='user notes',
            beta_names=['beta1', 'beta2'],
            beta_values=[1.2, 1.3],
            lower_bounds=[-10, -11],
            upper_bounds=[None, 23],
            gradient=[0.001, -0.002],
            hessian=[[1, 2], [3, 4]],
            bhhh=[[10, 20], [30, 40]],
            null_log_likelihood=-1000,
            initial_log_likelihood=-2000,
            final_log_likelihood=-500,
            data_name='test_data',
            sample_size=1234,
            number_of_observations=987,
            monte_carlo=True,
            number_of_draws=10_000,
            types_of_draws={'var1': 'UNIFORM'},
            number_of_excluded_data=10,
            draws_processing_time=timedelta(seconds=10),
            optimization_messages={'Diagnostic': 'test'},
            convergence=False,
            bootstrap=[[1.0, 1.0], [2.0, 2.0]],
            bootstrap_time=timedelta(seconds=20),
        )

        self.estimation_results = EstimationResults(
            raw_estimation_results=raw_estimation
        )

    def test_format_real_number(self):
        self.assertEqual(format_real_number(123456.789), '1.23e+05')
        self.assertEqual(format_real_number(0.000123), '0.000123')

    def test_get_html_header(self):
        result = get_html_header(self.estimation_results)
        self.assertIn('<html>', result)
        self.assertIn('<title>test_model', result)
        self.assertIn(f'Report from biogeme {get_version()} [{versionDate}]', result)

    def test_get_html_footer(self):
        result = get_html_footer()
        self.assertEqual(result, '</body>\n</html>')

    def test_get_html_preamble(self):
        result = get_html_preamble(self.estimation_results, "testfile.html")
        self.assertIn("testfile.html", result)
        self.assertIn("test_data", result)
        self.assertIn("Algorithm failed to converge", result)

    def test_get_html_general_statistics(self):
        result = get_html_general_statistics(self.estimation_results)
        self.assertIn('<strong>Final log likelihood</strong>', result)
        self.assertIn('<strong>Number of draws</strong>', result)
        self.assertIn('<strong>Sample size</strong>', result)

    def test_get_html_estimated_parameters(self):
        result = get_html_estimated_parameters(
            self.estimation_results,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )
        self.assertIn('<th>Name</th>', result)
        self.assertIn('<td>beta1</td>', result)
        self.assertIn('<td>1.2</td>', result)

    def test_get_html_condition_number(self):
        result = get_html_condition_number(self.estimation_results)
        self.assertIn('<p>Smallest eigenvalue', result)
        self.assertIn('<p>Largest eigenvalue', result)

    def test_get_html_correlation_results(self):
        result = get_html_correlation_results(
            self.estimation_results,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )
        self.assertIn('<th>Coefficient 1</th>', result)
        self.assertIn('<th>Coefficient 2</th>', result)
        self.assertIn('<th>Robust covariance</th>', result)
        self.assertIn('<th>Robust correlation</th>', result)
        self.assertIn('<th>Robust t-test</th>', result)
        self.assertIn('<th>Robust p-value</th>', result)

    def test_generate_html_file(self):
        filename = "testfile.html"

        if os.path.exists(filename):
            os.remove(filename)

        generate_html_file(self.estimation_results, filename, overwrite=True)

        # Check that the file was created and contains HTML content
        self.assertTrue(os.path.exists(filename))

        with open(filename, 'r') as file:
            content = file.read()
            self.assertIn('<html>', content)
            self.assertIn('<h1>Estimation report</h1>', content)
            self.assertIn('<h2>Correlation of coefficients</h2>', content)
            self.assertIn('</html>', content)

        # Clean up by removing the file
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
