import os
import unittest
from datetime import timedelta

from biogeme.results_processing import (
    EstimationResults,
    RawEstimationResults,
    get_f12,
    generate_f12_file,
)


class TestF12Generation(unittest.TestCase):

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
            optimization_messages={'Diagnostic': 'test', 'Number of iterations': 999},
            convergence=True,
            bootstrap=[[1.0, 1.0], [2.0, 2.0]],
            bootstrap_time=timedelta(seconds=20),
        )

        self.estimation_results = EstimationResults(
            raw_estimation_results=raw_estimation
        )

    def test_get_f12(self):
        # Generate the F12 output as a string
        f12_output = get_f12(self.estimation_results)
        # Check that the output contains relevant elements
        self.assertIn("test_model", f12_output)
        self.assertIn("From biogeme", f12_output)
        self.assertIn("END", f12_output)
        self.assertIn("beta1", f12_output)
        self.assertIn("beta2", f12_output)

        # Verify that parameter values are included
        self.assertIn("+1.200000000000e+00", f12_output)
        self.assertIn("+1.300000000000e+00", f12_output)

    def test_generate_f12_file(self):
        # Name of the file to generate
        filename = "test_f12_output.f12"

        # Remove file if it exists to ensure a clean test
        if os.path.exists(filename):
            os.remove(filename)

        # Generate the F12 file
        generate_f12_file(self.estimation_results, filename, overwrite=True)

        # Check that the file was created
        self.assertTrue(os.path.exists(filename))

        # Check the contents of the file
        with open(filename, 'r') as file:
            content = file.read()
            self.assertIn("test_model", content)
            self.assertIn("beta1", content)
            self.assertIn("+1.200000000000e+00", content)
            self.assertIn("999", content)

        # Clean up by removing the generated file
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
