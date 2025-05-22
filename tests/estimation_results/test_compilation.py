import unittest
from datetime import timedelta

from biogeme.results_processing import RawEstimationResults
from biogeme.results_processing.compilation import compile_estimation_results
from biogeme.results_processing.estimation_results import EstimationResults


class TestCompileEstimationResults(unittest.TestCase):

    def setUp(self):
        raw_results = RawEstimationResults(
            model_name='test',
            user_notes='',
            beta_names=['beta1', 'beta2'],
            beta_values=[1.0, 2.0],
            lower_bounds=[-100, -100],
            upper_bounds=[100, 100],
            gradient=[0, 0],
            hessian=[[1, 1], [1, 1]],
            bhhh=[[1, 1], [1, 1]],
            null_log_likelihood=-1000,
            initial_log_likelihood=-1000,
            final_log_likelihood=-100,
            data_name='data',
            sample_size=500,
            number_of_observations=500,
            monte_carlo=False,
            number_of_draws=0,
            types_of_draws={},
            number_of_excluded_data=0,
            draws_processing_time=timedelta(1),
            optimization_messages={},
            convergence=True,
            bootstrap=[],
            bootstrap_time=None,
        )
        # Create a real instance of EstimationResults
        self.mock_results = EstimationResults(raw_results)

    def test_compile_estimation_results_basic(self):
        """Test basic functionality of compile_estimation_results."""
        dict_of_results = {
            'Model_1': self.mock_results,
            'Model_2': self.mock_results,
        }

        df, config = compile_estimation_results(
            dict_of_results=dict_of_results,
            statistics=('Number of estimated parameters', 'Sample size'),
            include_parameter_estimates=False,
            include_stderr=False,
            include_t_test=False,
            formatted=False,
            use_short_names=True,
        )

        # Verify the dataframe structure
        self.assertIn('Model_000000', df.columns)
        self.assertIn('Model_000001', df.columns)
        self.assertIn('Number of estimated parameters', df.index)
        self.assertIn('Sample size', df.index)

        # Verify the values
        self.assertEqual(df.loc['Number of estimated parameters', 'Model_000000'], '2')
        self.assertEqual(df.loc['Sample size', 'Model_000001'], '500')

    def test_compile_estimation_results_with_parameters(self):
        """Test compile_estimation_results with parameter estimates, stderr, and t-test."""
        dict_of_results = {
            'Model_1': self.mock_results,
        }

        df, config = compile_estimation_results(
            dict_of_results=dict_of_results,
            statistics=('Final log likelihood',),
            include_parameter_estimates=True,
            include_stderr=True,
            include_t_test=True,
            formatted=True,
            use_short_names=False,
        )

        # Check the inclusion of parameters, standard errors, and t-tests in formatted results
        self.assertIn('beta1 (std) (t-test)', df.index)
        self.assertIn('beta2 (std) (t-test)', df.index)
        self.assertEqual(df.loc['beta1 (std) (t-test)', 'Model_1'], '1 (0.5) (2)')
        self.assertEqual(df.loc['beta2 (std) (t-test)', 'Model_1'], '2 (0.5) (4)')


if __name__ == '__main__':
    unittest.main()
