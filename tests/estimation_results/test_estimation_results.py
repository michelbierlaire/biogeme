"""
Test the estimation results

Michel Bierlaire
Fri Oct 4 09:52:23 2024
"""

import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

import numpy as np
from biogeme.exceptions import BiogemeError
from biogeme.results_processing import (
    EstimateVarianceCovariance,
    EstimationResults,
    RawEstimationResults,
    calc_p_value,
    get_pandas_estimated_parameters,
)


class TestEstimationResults(unittest.TestCase):
    def setUp(self):
        """Set up the test fixtures."""
        self.raw_results = RawEstimationResults(
            model_name='Test Model',
            user_notes='Test notes',
            beta_names=['beta1', 'beta2'],
            beta_values=[1.0, 1.5],
            lower_bounds=[0.0, 1.5],
            upper_bounds=[1.5, 2.5],
            gradient=[0.1, 0.2],
            hessian=[[-0.5, -0.2], [-0.2, -0.4]],
            bhhh=[[0.01, 0.02], [0.02, 0.03]],
            null_log_likelihood=-300.0,
            initial_log_likelihood=-200.0,
            final_log_likelihood=-150.0,
            data_name='Sample Data',
            sample_size=1000,
            number_of_observations=100,
            monte_carlo=False,
            number_of_draws=500,
            types_of_draws={'type1': 'UNIFORM'},
            number_of_excluded_data=10,
            draws_processing_time=timedelta(seconds=300),
            optimization_messages={'message': 'Optimization successful'},
            convergence=True,
            bootstrap=[[1.0, 2.0], [1.1, 2.1]],
            bootstrap_time=timedelta(seconds=100),
        )
        self.estimation_results = EstimationResults(self.raw_results)

    def test_empty(self):
        with self.assertRaises(BiogemeError):
            the_empty = EstimationResults(raw_estimation_results=None)

    def test_number_of_parameters(self):
        """Test the number_of_parameters property."""
        self.assertEqual(self.estimation_results.number_of_parameters, 2)

    def test_number_of_free_parameters(self):
        """Test the number_of_free_parameters property."""
        self.assertEqual(self.estimation_results.number_of_free_parameters, 1)

    def test_is_any_bound_active(self):
        """Test if any bound is active."""
        self.assertTrue(self.estimation_results.is_any_bound_active())

    def test_is_bound_active(self):
        """Test if a specific bound is active."""
        self.assertFalse(self.estimation_results.is_bound_active('beta1'))
        self.assertTrue(self.estimation_results.is_bound_active('beta2'))

    def test_akaike_information_criterion(self):
        """Test AIC calculation."""
        expected_aic = 2.0 * 2 - 2 * -150.0
        self.assertEqual(
            self.estimation_results.akaike_information_criterion, expected_aic
        )

    def test_bayesian_information_criterion(self):
        """Test BIC calculation."""
        expected_bic = -2.0 * -150.0 + 2 * np.log(1000)
        self.assertAlmostEqual(
            self.estimation_results.bayesian_information_criterion, expected_bic
        )

    def test_algorithm_has_converged(self):
        """Test if the algorithm has converged."""
        self.assertTrue(self.estimation_results.algorithm_has_converged)

    def test_variance_covariance_missing(self):
        """Test if the variance covariance matrix is missing."""
        self.assertFalse(self.estimation_results.variance_covariance_missing)

    def test_eigen_structure(self):
        """Test eigen structure calculation."""
        eigenvalues, eigenvectors = self.estimation_results.eigen_structure()
        self.assertEqual(len(eigenvalues), 2)
        self.assertEqual(eigenvectors.shape, (2, 2))

    def test_smallest_eigenvalue(self):
        """Test smallest eigenvalue calculation."""
        self.assertAlmostEqual(
            self.estimation_results.smallest_eigenvalue, 0.24384471871911698, places=7
        )

    def test_largest_eigenvalue(self):
        """Test largest eigenvalue calculation."""
        self.assertAlmostEqual(
            self.estimation_results.largest_eigenvalue, 0.6561552812808831, places=7
        )

    def test_condition_number(self):
        """Test condition number calculation."""
        expected_condition_number = (
            self.estimation_results.largest_eigenvalue
            / self.estimation_results.smallest_eigenvalue
        )
        self.assertAlmostEqual(
            self.estimation_results.condition_number, expected_condition_number
        )

    def test_rao_cramer_variance_covariance_matrix(self):
        """Test Rao-Cramer variance-covariance matrix calculation."""
        rao_cramer_matrix = (
            self.estimation_results.rao_cramer_variance_covariance_matrix
        )
        self.assertEqual(rao_cramer_matrix.shape, (2, 2))

    def test_likelihood_ratio_null(self):
        """Test likelihood ratio test against the null model."""
        expected_lr_null = -2 * (-300.0 - (-150.0))
        self.assertEqual(
            self.estimation_results.likelihood_ratio_null, expected_lr_null
        )

    def test_get_parameter_value(self):
        """Test parameter value retrieval."""
        self.assertEqual(self.estimation_results.get_parameter_value_from_index(1), 1.5)

    def test_get_parameter_t_test(self):
        """Test t-test calculation for a parameter."""
        t_value = self.estimation_results.get_parameter_t_test_from_index(
            parameter_index=0, estimate_var_covar=EstimateVarianceCovariance.RAO_CRAMER
        )
        self.assertAlmostEqual(np.abs(t_value), 0.632455532033676)
        t_value = self.estimation_results.get_parameter_t_test_from_index(
            parameter_index=0,
            estimate_var_covar=EstimateVarianceCovariance.RAO_CRAMER,
            target=2.0,
        )
        self.assertAlmostEqual(np.abs(t_value), 0.632455532033676)

    def test_get_parameter_p_value(self):
        """Test p-value calculation for a parameter."""
        p_value = self.estimation_results.get_parameter_p_value_from_index(
            0, EstimateVarianceCovariance.RAO_CRAMER
        )
        self.assertAlmostEqual(p_value, 0.5270892568655381)

    def test_calc_p_value(self):
        """Test the p-value calculation using calc_p_value function."""
        t_stat = 1.96
        p_value = calc_p_value(t_stat)
        self.assertAlmostEqual(p_value, 0.05, places=2)

    def test_likelihood_ratio_test(self):
        """Test likelihood ratio test between two models."""
        other_model = MagicMock(spec=EstimationResults)
        other_model.final_log_likelihood = -140.0
        other_model.number_of_parameters = 3

        # Mock the likelihood ratio function
        with patch(
            'biogeme.tools.likelihood_ratio.likelihood_ratio_test',
            return_value=('Pass', 2.0, 3.84),
        ):
            result = self.estimation_results.likelihood_ratio_test(other_model)
            self.assertEqual(result, ('Pass', 2.0, 3.84))

    def test_get_beta_values(self):
        """Test retrieval of beta values."""
        betas = self.estimation_results.get_beta_values()
        self.assertEqual(betas['beta1'], 1.0)

    def test_bootstrap_variance_covariance_matrix(self):
        """Test bootstrap variance-covariance matrix."""
        bootstrap_var_covar = (
            self.estimation_results.bootstrap_variance_covariance_matrix
        )
        self.assertEqual(bootstrap_var_covar.shape, (2, 2))

    def test_short_summary(self):
        """Test the short summary method."""
        summary = self.estimation_results.short_summary()
        self.assertIn('Test Model', summary)
        self.assertIn('Nbr of parameters:', summary)
        self.assertIn('Sample size:', summary)

    def test_get_general_statistics(self):
        """Test the general statistics dictionary generation."""
        stats = self.estimation_results.get_general_statistics()
        self.assertIn('Number of estimated parameters', stats)
        self.assertIn('Sample size', stats)

    def test_dump_yaml_file(self):
        """Test the YAML dumping of raw estimation results."""
        with patch(
            'biogeme.results_processing.estimation_results.serialize_to_yaml'
        ) as mock_serialize:
            self.estimation_results.dump_yaml_file('test.yaml')
            mock_serialize.assert_called_once_with(
                data=self.raw_results, filename='test.yaml'
            )

    def test_pandas_results(self):
        pandas_results = get_pandas_estimated_parameters(
            estimation_results=self.estimation_results
        )
        nrows, ncolumns = pandas_results.shape
        self.assertEqual(nrows, 2)
        self.assertEqual(ncolumns, 6)

    def test_gradient_None(self):
        raw_results = RawEstimationResults(
            model_name='Test Model',
            user_notes='Test notes',
            beta_names=['beta1', 'beta2'],
            beta_values=[1.0, 1.5],
            lower_bounds=[0.0, 1.5],
            upper_bounds=[1.5, 2.5],
            gradient=None,
            hessian=None,
            bhhh=None,
            null_log_likelihood=-300.0,
            initial_log_likelihood=-200.0,
            final_log_likelihood=-150.0,
            data_name='Sample Data',
            sample_size=1000,
            number_of_observations=100,
            monte_carlo=False,
            number_of_draws=500,
            types_of_draws={'type1': 'UNIFORM'},
            number_of_excluded_data=10,
            draws_processing_time=timedelta(seconds=300),
            optimization_messages={'message': 'Optimization successful'},
            convergence=True,
            bootstrap=[[1.0, 2.0], [1.1, 2.1]],
            bootstrap_time=timedelta(seconds=100),
        )
        with self.assertRaises(BiogemeError):
            _ = EstimationResults(raw_results)


if __name__ == '__main__':
    unittest.main()
