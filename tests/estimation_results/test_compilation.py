import unittest
from unittest.mock import MagicMock

from biogeme.results_processing.compilation import compile_estimation_results
from biogeme.results_processing.estimation_results import EstimationResults


class TestCompileEstimationResults(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Create a real instance of EstimationResults
        self.mock_results = EstimationResults.__new__(EstimationResults)

        # Mock necessary methods on this instance
        self.mock_results.get_general_statistics = MagicMock(
            return_value={
                'Number of estimated parameters': ['2'],
                'Sample size': ['100'],
                'Final log likelihood': ['-1200.0'],
                'Akaike Information Criterion': ['2404.0'],
                'Bayesian Information Criterion': ['2450.0'],
            }
        )
        self.mock_results.beta_names = ['beta1', 'beta2']
        self.mock_results.get_parameter_value = MagicMock(side_effect=[1.0, 2.0])
        self.mock_results.get_parameter_std_err = MagicMock(side_effect=[0.1, 0.2])
        self.mock_results.get_parameter_t_test = MagicMock(side_effect=[10.0, 8.0])

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
        self.assertEqual(df.loc['Sample size', 'Model_000001'], '100')

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
        self.assertEqual(df.loc['beta1 (std) (t-test)', 'Model_1'], '1 (0.1) (10)')
        self.assertEqual(df.loc['beta2 (std) (t-test)', 'Model_1'], '2 (0.2) (8)')


if __name__ == '__main__':
    unittest.main()
