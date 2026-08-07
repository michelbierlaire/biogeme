"""
Test the raw estimation results

Michel Bierlaire
Tue Oct 1 19:18:45 2024
"""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from yaml import SafeLoader, dump, load

from biogeme.results_processing import (
    RawEstimationResults,
    deserialize_from_yaml,
    serialize_to_yaml,
)


class TestRawEstimationResults(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test"""
        self.test_data = RawEstimationResults(
            model_name='Test Model',
            user_notes='These are user notes.',
            beta_names=['beta1', 'beta2'],
            beta_values=[0.5, 1.5],
            lower_bounds=[0.0, 1.0],
            upper_bounds=[1.0, 2.0],
            gradient=[0.01, 0.02],
            hessian=[[0.1, 0.2], [0.2, 0.3]],
            bhhh=[[0.01, 0.02], [0.02, 0.03]],
            null_log_likelihood=-150.0,
            initial_log_likelihood=-140.0,
            final_log_likelihood=-120.0,
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
            bootstrap=[[0.5, 1.5], [1.5, 2.5]],
            bootstrap_time=timedelta(seconds=100),
        )

    def test_serialization(self):
        """Test serialization of RawEstimationResults to YAML"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name

        # Call the serialize_to_yaml function
        serialize_to_yaml(self.test_data, temp_filename)

        # Verify the content of the file
        with open(temp_filename, 'r') as file:
            content = file.read()
            self.assertIn('Test Model', content)
            self.assertIn('These are user notes.', content)
            self.assertIn('beta1', content)
            self.assertIn('500', content)  # Check number of draws
            self.assertIn('300.0', content)  # Check draws_processing_time as seconds

        # Clean up the temp file
        os.remove(temp_filename)

    def test_deserialization(self):
        """Test deserialization of RawEstimationResults from YAML"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name

        # Serialize the data first
        serialize_to_yaml(self.test_data, temp_filename)

        # Deserialize the data
        loaded_data = deserialize_from_yaml(temp_filename)

        # Verify that loaded data matches original data
        self.assertEqual(loaded_data.model_name, self.test_data.model_name)
        self.assertEqual(loaded_data.beta_values, self.test_data.beta_values)
        self.assertEqual(loaded_data.sample_size, self.test_data.sample_size)
        self.assertEqual(loaded_data.number_of_draws, self.test_data.number_of_draws)
        self.assertEqual(
            loaded_data.draws_processing_time, self.test_data.draws_processing_time
        )

        # Clean up the temp file
        os.remove(temp_filename)

    def test_old_yaml_without_lifecycle_fields_is_complete(self):
        """Files created before recoverable post-processing remain compatible."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
        serialize_to_yaml(self.test_data, temp_filename)
        with open(temp_filename) as file:
            content = file.read()
        lifecycle_names = (
            'optimization_complete',
            'gradient_bhhh_complete',
            'hessian_complete',
            'bootstrap_complete',
        )
        filtered = '\n'.join(
            line
            for line in content.splitlines()
            if not line.startswith(lifecycle_names)
        )
        with open(temp_filename, 'w') as file:
            file.write(filtered)

        loaded = deserialize_from_yaml(temp_filename)
        self.assertTrue(loaded.optimization_complete)
        self.assertTrue(loaded.gradient_bhhh_complete)
        self.assertTrue(loaded.hessian_complete)
        self.assertTrue(loaded.bootstrap_complete)
        os.remove(temp_filename)

    def test_atomic_serialization_preserves_previous_checkpoint(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
        serialize_to_yaml(self.test_data, temp_filename)
        with open(temp_filename) as file:
            original_content = file.read()

        self.test_data.model_name = 'Replacement that must not be installed'
        with patch(
            'biogeme.results_processing.raw_estimation_results.os.replace',
            side_effect=OSError('simulated atomic replacement failure'),
        ):
            with self.assertRaisesRegex(OSError, 'simulated atomic replacement'):
                serialize_to_yaml(self.test_data, temp_filename)

        with open(temp_filename) as file:
            self.assertEqual(file.read(), original_content)
        temporary_files = list(
            Path(temp_filename).parent.glob(f'.{Path(temp_filename).name}.*.tmp')
        )
        self.assertEqual(temporary_files, [])
        os.remove(temp_filename)

    def test_timedelta_representer(self):
        """Test timedelta serialization"""
        result_yaml = dump(
            {
                'draws_processing_time': self.test_data.draws_processing_time.total_seconds()
            }
        )
        self.assertIn('300.0', result_yaml)

    def test_timedelta_constructor(self):
        """Test timedelta deserialization"""
        yaml_data = 'draws_processing_time: "300.0"'
        result = load(yaml_data, Loader=SafeLoader)
        self.assertEqual(result['draws_processing_time'], '300.0')

    def test_empty_serialization(self):
        """Test that serialization works with minimal data"""
        empty_data = RawEstimationResults(
            model_name='',
            user_notes='',
            beta_names=[],
            beta_values=[],
            lower_bounds=[],
            upper_bounds=[],
            gradient=[],
            hessian=[],
            bhhh=[],
            null_log_likelihood=0.0,
            initial_log_likelihood=0.0,
            final_log_likelihood=0.0,
            data_name='',
            sample_size=0,
            number_of_observations=0,
            monte_carlo=False,
            number_of_draws=0,
            types_of_draws={},
            number_of_excluded_data=0,
            draws_processing_time=timedelta(seconds=0),
            optimization_messages={},
            convergence=False,
            bootstrap=[],
            bootstrap_time=None,
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name

        serialize_to_yaml(empty_data, temp_filename)

        # Verify the file has been written without any errors
        with open(temp_filename, 'r') as file:
            content = file.read()
            self.assertIn('0.0', content)

        # Clean up the temp file
        os.remove(temp_filename)


if __name__ == '__main__':
    unittest.main()
