import unittest
from datetime import timedelta

import pandas as pd

from biogeme.results_processing import (
    EstimateVarianceCovariance,
    EstimationResults,
    RawEstimationResults,
    get_pandas_correlation_results,
    get_pandas_estimated_parameters,
    get_pandas_one_pair_of_parameters,
    get_pandas_one_parameter,
)


class TestPandasGeneration(unittest.TestCase):
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
            convergence=True,
            bootstrap=[[1.0, 1.0], [2.0, 2.0]],
            bootstrap_time=timedelta(seconds=20),
        )

        self.estimation_results = EstimationResults(
            raw_estimation_results=raw_estimation
        )

    def test_get_pandas_one_parameter(self):
        # Test for getting one parameter in pandas format
        parameter_index = 0
        result = get_pandas_one_parameter(
            estimation_results=self.estimation_results,
            parameter_index=parameter_index,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )

        # Check that the result is a dictionary and contains expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('Name', result)
        self.assertIn('Value', result)
        self.assertIn('Robust std err.', result)
        self.assertIn('Robust t-stat.', result)
        self.assertIn('Robust p-value', result)

        # Validate parameter values
        self.assertEqual(result['Name'], 'beta1')
        self.assertEqual(result['Value'], 1.2)

    def test_get_pandas_estimated_parameters(self):
        # Test for getting all estimated parameters in pandas DataFrame format.
        result = get_pandas_estimated_parameters(
            estimation_results=self.estimation_results,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )

        # Check that the result is a dictionary containing the default table.
        self.assertIsInstance(result, dict)
        self.assertIn('Estimated parameters', result)

        df = result['Estimated parameters']
        self.assertIsInstance(df, pd.DataFrame)

        # Ensure the DataFrame has expected columns and data.
        self.assertIn('Name', df.columns)
        self.assertIn('Value', df.columns)
        self.assertIn('Robust std err.', df.columns)
        self.assertIn('Robust t-stat.', df.columns)
        self.assertIn('Robust p-value', df.columns)

        # Verify the content for the first parameter.
        self.assertEqual(df.iloc[0]['Name'], 'beta1')
        self.assertEqual(df.iloc[0]['Value'], 1.2)
        self.assertIn('beta2', set(df['Name']))

    def test_get_pandas_estimated_parameters_by_group(self):
        result = get_pandas_estimated_parameters(
            estimation_results=self.estimation_results,
            group_of_parameters={'First group': ['beta1']},
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )

        self.assertIsInstance(result, dict)
        self.assertIn('First group', result)
        self.assertIn('Other parameters', result)

        first_group_df = result['First group']
        self.assertIsInstance(first_group_df, pd.DataFrame)
        self.assertEqual(list(first_group_df['Name']), ['beta1'])

        other_parameters_df = result['Other parameters']
        self.assertIsInstance(other_parameters_df, pd.DataFrame)
        self.assertEqual(list(other_parameters_df['Name']), ['beta2'])

    def test_get_pandas_estimated_parameters_parameter_in_several_groups(self):
        result = get_pandas_estimated_parameters(
            estimation_results=self.estimation_results,
            group_of_parameters={
                'First group': ['beta1'],
                'Second group': ['beta1', 'beta2'],
            },
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )

        self.assertIsInstance(result, dict)
        self.assertIn('First group', result)
        self.assertIn('Second group', result)
        self.assertNotIn('Other parameters', result)

        self.assertEqual(list(result['First group']['Name']), ['beta1'])
        self.assertEqual(list(result['Second group']['Name']), ['beta1', 'beta2'])

    def test_get_pandas_one_pair_of_parameters(self):
        # Test for getting one pair of parameter correlations in pandas format
        first_index = 0
        second_index = 1
        result = get_pandas_one_pair_of_parameters(
            estimation_results=self.estimation_results,
            first_parameter_index=first_index,
            second_parameter_index=second_index,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )

        # Check that the result is a dictionary and contains expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('First parameter', result)
        self.assertIn('Second parameter', result)
        self.assertIn('Robust covariance', result)
        self.assertIn('Robust correlation', result)
        self.assertIn('Robust t-stat.', result)
        self.assertIn('Robust p-value', result)

        # Validate parameter names and covariance/correlation
        self.assertEqual(result['First parameter'], 'beta1')
        self.assertEqual(result['Second parameter'], 'beta2')

    def test_get_pandas_correlation_results(self):
        # Test for getting the correlation results in pandas DataFrame format
        df = get_pandas_correlation_results(
            estimation_results=self.estimation_results,
            variance_covariance_type=EstimateVarianceCovariance.ROBUST,
        )

        # Check if the result is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        # Ensure the DataFrame has expected columns and data
        self.assertIn('First parameter', df.columns)
        self.assertIn('Second parameter', df.columns)
        self.assertIn('Robust covariance', df.columns)
        self.assertIn('Robust correlation', df.columns)
        self.assertIn('Robust t-stat.', df.columns)
        self.assertIn('Robust p-value', df.columns)

        # Validate the content for the first pair of parameters
        self.assertEqual(df.iloc[0]['First parameter'], 'beta2')
        self.assertEqual(df.iloc[0]['Second parameter'], 'beta1')


if __name__ == '__main__':
    unittest.main()
