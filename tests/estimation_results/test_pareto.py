import unittest
from datetime import timedelta

from biogeme_optimization.pareto import Pareto

from biogeme.results_processing import (
    EstimationResults,
    RawEstimationResults,
    pareto_optimal,
)


class TestParetoOptimal(unittest.TestCase):

    def setUp(self):
        # Set up multiple real EstimationResults for the Pareto optimal test
        raw_estimation_1 = RawEstimationResults(
            model_name='model_1',
            user_notes='user notes 1',
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
            data_name='data_1',
            sample_size=1000,
            number_of_observations=900,
            monte_carlo=True,
            number_of_draws=5000,
            types_of_draws={'var1': 'UNIFORM'},
            number_of_excluded_data=10,
            draws_processing_time=timedelta(seconds=15),
            optimization_messages={'Diagnostic': 'optimal'},
            convergence=True,
            bootstrap=[[1.0, 1.0], [2.0, 2.0]],
            bootstrap_time=timedelta(seconds=10),
        )

        raw_estimation_2 = RawEstimationResults(
            model_name='model_2',
            user_notes='user notes 2',
            beta_names=['beta3', 'beta4'],
            beta_values=[2.1, 2.4],
            lower_bounds=[-5, -6],
            upper_bounds=[None, 20],
            gradient=[0.002, -0.003],
            hessian=[[5, 6], [7, 8]],
            bhhh=[[15, 25], [35, 45]],
            null_log_likelihood=-900,
            initial_log_likelihood=-1800,
            final_log_likelihood=-700,
            data_name='data_2',
            sample_size=1100,
            number_of_observations=950,
            monte_carlo=True,
            number_of_draws=6000,
            types_of_draws={'var2': 'NORMAL'},
            number_of_excluded_data=15,
            draws_processing_time=timedelta(seconds=20),
            optimization_messages={'Diagnostic': 'converged'},
            convergence=True,
            bootstrap=[[2.0, 2.0], [3.0, 3.0]],
            bootstrap_time=timedelta(seconds=12),
        )

        # Create estimation results objects
        self.estimation_results_1 = EstimationResults(
            raw_estimation_results=raw_estimation_1
        )
        self.estimation_results_2 = EstimationResults(
            raw_estimation_results=raw_estimation_2
        )

        # Dictionary of results to test Pareto optimality
        self.dict_of_results = {
            'config_1': self.estimation_results_1,
            'config_2': self.estimation_results_2,
        }

    def test_pareto_optimal(self):
        # Call the function to identify Pareto optimal models
        optimal_results = pareto_optimal(self.dict_of_results)

        # Check if the result is a dictionary
        self.assertIsInstance(optimal_results, dict)

        # Ensure the results contain only Pareto optimal configurations
        # Based on the final log-likelihood and number of parameters, only one should be Pareto optimal
        # Model 1 has a better log-likelihood (-500), while Model 2 has more parameters
        self.assertIn('config_1', optimal_results)  # config_1 should be included
        self.assertNotIn('config_2', optimal_results)  # config_2 is not Pareto optimal

        # Validate the content of the Pareto optimal result
        self.assertEqual(optimal_results['config_1'], self.estimation_results_1)

    def test_pareto_with_existing_pareto(self):
        # Test with an existing Pareto set passed in
        existing_pareto = Pareto()
        optimal_results = pareto_optimal(self.dict_of_results, a_pareto=existing_pareto)

        # Ensure the results still reflect the correct Pareto optimal configuration
        self.assertIn('config_1', optimal_results)
        self.assertNotIn('config_2', optimal_results)

        # Ensure that the Pareto set was modified
        self.assertTrue(len(existing_pareto.pareto) > 0)

    def test_pareto_empty(self):
        # Test with an empty dictionary
        empty_dict = {}
        optimal_results = pareto_optimal(empty_dict)

        # Ensure that no results are returned for an empty input
        self.assertEqual(len(optimal_results), 0)


if __name__ == '__main__':
    unittest.main()
