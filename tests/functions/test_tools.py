"""
Test the tools module

:author: Michel Bierlaire
:date: Sun Aug  8 17:42:48 2021
"""

import os
import tempfile
import unittest
from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

import biogeme.tools.database
import biogeme.tools.derivatives
import biogeme.tools.files
import biogeme.tools.likelihood_ratio
import biogeme.tools.primes
import biogeme.tools.unique_ids
from biogeme.exceptions import BiogemeError
from biogeme.function_output import FunctionOutput
from biogeme.tools import (
    create_backup,
    format_timedelta,
    is_valid_filename,
    safe_deserialize_array,
    safe_serialize_array,
)
from biogeme.tools.derivatives import CheckDerivativesResults
from test_data import (
    input_flatten,
    output_flatten_1,
    output_flatten_2,
    output_flatten_3,
)


# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring


def my_function(
    x: np.ndarray, gradient: bool, hessian: bool, bhhh: bool
) -> FunctionOutput:
    f = np.log(x[0]) + np.exp(x[1])
    g = np.empty(2)
    g[0] = 1.0 / x[0]
    g[1] = np.exp(x[1])
    H = np.empty((2, 2))
    H[0, 0] = -1.0 / x[0] ** 2
    H[0, 1] = 0.0
    H[1, 0] = 0.0
    H[1, 1] = np.exp(x[1])
    return FunctionOutput(function=f, gradient=g, hessian=H)


class TestTools(unittest.TestCase):
    def test_findiff_g(self):
        x = np.array([1.1, 1.1])
        g_fd = biogeme.tools.derivatives.findiff_g(my_function, x)
        np.testing.assert_almost_equal(g_fd, [0.90909087, 3.00416619], decimal=5)

    def test_findiff_H(self):
        x = np.array([1.1, 1.1])
        H_fd = biogeme.tools.derivatives.findiff_h(my_function, x)
        np.testing.assert_almost_equal(
            H_fd,
            [
                [
                    -0.8264462,
                    0,
                ],
                [
                    0,
                    3.00416619,
                ],
            ],
            decimal=5,
        )

    def test_checkDerivatives(self):
        x = np.array([1.1, -1.5])
        results: CheckDerivativesResults = biogeme.tools.derivatives.check_derivatives(
            my_function, x, names=['First', 'Second']
        )

        np.testing.assert_almost_equal(results.errors_gradient, [0, 0])
        np.testing.assert_almost_equal(results.errors_hessian, [[0, 0], [0, 0]])

        x = np.array([3.2, 1.32])
        results: CheckDerivativesResults = biogeme.tools.derivatives.check_derivatives(
            my_function, x, names=['First', 'Second']
        )
        np.testing.assert_almost_equal(results.errors_gradient, [0, 0], decimal=5)
        np.testing.assert_almost_equal(
            results.errors_hessian, [[0, 0], [0, 0]], decimal=5
        )

    def test_getPrimeNumbers(self):
        result = biogeme.tools.primes.get_prime_numbers(7)
        self.assertListEqual(result, [2, 3, 5, 7, 11, 13, 17])

        with self.assertRaises(BiogemeError):
            result = biogeme.tools.primes.get_prime_numbers(0)

        with self.assertRaises(BiogemeError):
            result = biogeme.tools.primes.get_prime_numbers(-1)

        with self.assertRaises(BiogemeError):
            result = biogeme.tools.primes.get_prime_numbers(0.3)

    def test_calculatePrimeNumbers(self):
        result = biogeme.tools.primes.calculate_prime_numbers(10)
        self.assertListEqual(result, [2, 3, 5, 7])
        result = biogeme.tools.primes.calculate_prime_numbers(0)
        self.assertListEqual(result, [])

        with self.assertRaises(BiogemeError):
            result = biogeme.tools.primes.calculate_prime_numbers(-1)

        with self.assertRaises(BiogemeError):
            result = biogeme.tools.primes.calculate_prime_numbers(0.3)

    def test_countNumberOfGroups(self):
        df = pd.DataFrame(
            {
                'ID': [1, 1, 2, 3, 3, 1, 2, 3],
                'value': [1000, 2000, 3000, 4000, 5000, 5000, 10000, 20000],
            }
        )
        nbr = biogeme.tools.count_number_of_groups(df, 'ID')
        self.assertEqual(nbr, 6)
        nbr = biogeme.tools.count_number_of_groups(df, 'value')
        self.assertEqual(nbr, 7)

    def test_likelihood_ratio_test(self):
        model1 = (-1340.8, 5)
        model2 = (-1338.49, 7)
        r = biogeme.tools.likelihood_ratio.likelihood_ratio_test(model1, model2)
        expected = (
            'H0 cannot be rejected at level 5.0%',
            4.619999999999891,
            5.991464547107979,
        )
        self.assertEqual(len(r), len(expected))
        self.assertEqual(r[0], expected[0])
        self.assertAlmostEqual(r[1], expected[1], places=3)
        self.assertAlmostEqual(r[2], expected[2], places=3)

        r = biogeme.tools.likelihood_ratio.likelihood_ratio_test(model2, model1)
        self.assertEqual(len(r), len(expected))
        self.assertEqual(r[0], expected[0])
        self.assertAlmostEqual(r[1], expected[1], places=3)
        self.assertAlmostEqual(r[2], expected[2], places=3)
        r = biogeme.tools.likelihood_ratio.likelihood_ratio_test(
            model1, model2, significance_level=0.1
        )
        expected = (
            'H0 can be rejected at level 10.0%',
            4.619999999999891,
            4.605170185988092,
        )

        self.assertEqual(len(r), len(expected))
        self.assertEqual(r[0], expected[0])
        self.assertAlmostEqual(r[1], expected[1], places=3)
        self.assertAlmostEqual(r[2], expected[2], places=3)
        with self.assertRaises(BiogemeError):
            model1 = (-1340.8, 7)
            model2 = (-1338.49, 5)
            biogeme.tools.likelihood_ratio.likelihood_ratio_test(model1, model2)

    def test_flatten_database(self):
        df = deepcopy(input_flatten)
        result_1 = biogeme.tools.database.flatten_database(df, 'ID', row_name='Name')
        output_flatten_1.index.name = 'ID'
        pd.testing.assert_frame_equal(result_1, output_flatten_1)
        result_2 = biogeme.tools.database.flatten_database(input_flatten, 'ID')
        output_flatten_2.index.name = 'ID'
        pd.testing.assert_frame_equal(result_2, output_flatten_2)
        result_3 = biogeme.tools.database.flatten_database(
            df, 'ID', identical_columns=[]
        )
        output_flatten_3.index.name = 'ID'
        pd.testing.assert_frame_equal(result_3, output_flatten_3)

    def test_temporary_file(self):
        content = 'a random sentence'
        with biogeme.tools.files.TemporaryFile() as temporary_file:
            with open(temporary_file.name, 'w', encoding='utf-8') as f:
                f.write(content)
            with open(temporary_file.name, encoding='utf-8') as f:
                check = f.read()
            self.assertEqual(content, check)

    def test_model_names(self):
        model_names = biogeme.tools.unique_ids.ModelNames(prefix='Test')

        name_a = model_names('model_a')
        correct_name_a = 'Test_000000'
        self.assertEqual(name_a, correct_name_a)
        name_b = model_names('model_b')
        correct_name_b = 'Test_000001'
        self.assertEqual(name_a, correct_name_a)
        name_a = model_names('model_a')
        self.assertEqual(name_a, correct_name_a)

    def test_generate_unique_ids(self):
        list_1 = ['A', 'B', 'C']
        correct_dict_1 = {'A': 'A', 'B': 'B', 'C': 'C'}
        dict_1 = biogeme.tools.unique_ids.generate_unique_ids(list_1)
        self.assertDictEqual(correct_dict_1, dict_1)

        list_2 = ['A', 'B', 'B']
        correct_dict_2 = {'A': 'A', 'B_0': 'B', 'B_1': 'B'}
        dict_2 = biogeme.tools.unique_ids.generate_unique_ids(list_2)
        self.assertDictEqual(correct_dict_2, dict_2)

        list_3 = ['B', 'B', 'B']
        correct_dict_3 = {'B_2': 'B', 'B_0': 'B', 'B_1': 'B'}
        dict_3 = biogeme.tools.unique_ids.generate_unique_ids(list_3)
        self.assertDictEqual(correct_dict_3, dict_3)


class UniqueProductTestCase(unittest.TestCase):
    def test_unique_product(self):
        # Test case with small iterables
        iterables = [range(2), range(3), range(2)]
        expected_output = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (0, 2, 0),
            (0, 2, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (1, 2, 0),
            (1, 2, 1),
        ]
        result = list(biogeme.tools.unique_ids.unique_product(*iterables))
        self.assertEqual(result, expected_output)

        # Test case with larger iterables
        iterables = [range(3), range(4), range(2)]
        expected_output = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (0, 2, 0),
            (0, 2, 1),
            (0, 3, 0),
            (0, 3, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (1, 2, 0),
            (1, 2, 1),
            (1, 3, 0),
            (1, 3, 1),
            (2, 0, 0),
            (2, 0, 1),
            (2, 1, 0),
            (2, 1, 1),
            (2, 2, 0),
            (2, 2, 1),
            (2, 3, 0),
            (2, 3, 1),
        ]
        result = list(biogeme.tools.unique_ids.unique_product(*iterables))
        self.assertEqual(result, expected_output)

        # Test case with empty iterable
        iterables = [range(3), [], range(2)]
        expected_output = []
        result = list(biogeme.tools.unique_ids.unique_product(*iterables))
        self.assertEqual(result, expected_output)

        # Test case with single iterable
        iterables = [range(3)]
        expected_output = [(0,), (1,), (2,)]
        result = list(biogeme.tools.unique_ids.unique_product(*iterables))
        self.assertEqual(result, expected_output)


class TestFormatTimedelta(unittest.TestCase):

    def test_hours_minutes_seconds(self):
        td = timedelta(hours=2, minutes=30, seconds=15)
        self.assertEqual(format_timedelta(td), '2h 30m 15s')

    def test_minutes_seconds(self):
        td = timedelta(minutes=45, seconds=30)
        self.assertEqual(format_timedelta(td), '45m 30s')

    def test_seconds_microseconds(self):
        td = timedelta(seconds=5, microseconds=500000)
        self.assertEqual(format_timedelta(td), '5.5s')

    def test_only_microseconds_more_than_a_millisecond(self):
        td = timedelta(microseconds=1250)
        self.assertEqual(format_timedelta(td), '1ms')

    def test_only_microseconds_less_than_a_millisecond(self):
        td = timedelta(microseconds=999)
        self.assertEqual(format_timedelta(td), '999μs')

    def test_zero_units(self):
        td = timedelta(hours=0, minutes=0, seconds=5)
        self.assertEqual(format_timedelta(td), '5.0s')

    def test_large_timedelta(self):
        td = timedelta(days=2, hours=3, minutes=4, seconds=5)
        self.assertEqual(format_timedelta(td), '51h 4m 5s')

    def test_microseconds_rounding(self):
        # This test ensures that microseconds are correctly floored when converted to milliseconds
        td = timedelta(microseconds=1999)  # Should floor to 1ms, not round to 2ms
        self.assertEqual(format_timedelta(td), '1ms')


class TestIsValidFilename(unittest.TestCase):

    def test_empty_filename(self):
        self.assertEqual(is_valid_filename(''), (False, 'Name is empty'))

    def test_filename_with_invalid_chars(self):
        invalid_filenames = ['<invalid>', 'in"valid', 'in|valid', 'invalid?']
        for filename in invalid_filenames:
            result, message = is_valid_filename(filename)
            self.assertFalse(result)
            self.assertIn('Name contains one invalid char:', message)

    def test_windows_reserved_names(self):
        if os.name == 'nt':
            reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
            for name in reserved_names:
                result, message = is_valid_filename(name)
                self.assertFalse(result)
                self.assertIn('Name is a reserved name:', message)

    def test_exceeds_length_limit(self):
        long_filename = 'a' * 256
        self.assertEqual(
            is_valid_filename(long_filename),
            (False, f'The length of the filename exceeds 255: {len(long_filename)}'),
        )

    def test_valid_filename(self):
        self.assertEqual(is_valid_filename('valid_filename.txt'), (True, ''))

    def test_filename_with_space(self):
        self.assertEqual(is_valid_filename('valid filename.txt'), (True, ''))


class TestCreateBackup(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)  # Ensure cleanup after tests

    def create_test_file(self, filename, content='test content'):
        """Helper function to create a file with specified content."""
        path = os.path.join(self.test_dir.name, filename)
        with open(path, 'w') as f:
            f.write(content)
        return path

    @patch('biogeme.tools.files.logger')
    def test_file_does_not_exist(self, mock_logger):
        non_existent_file = os.path.join(self.test_dir.name, 'non_existent.txt')
        self.assertFalse(os.path.exists(non_existent_file))
        result = create_backup(non_existent_file)
        # Get the arguments of the last call to mock_logger.info
        args, kwargs = mock_logger.info.call_args

        # Check if the first argument (the log message) contains the substring
        self.assertIn(
            'No backup has been generated',
            args[0],
            "Log message does not contain expected substring",
        )

    def test_file_exists_rename_true(self):
        filename = self.create_test_file('exists.txt')
        self.assertTrue(os.path.exists(filename))
        backup_name = create_backup(filename)
        self.assertFalse(os.path.exists(filename))
        self.assertTrue(os.path.exists(backup_name))

    def test_file_exists_rename_false(self):
        filename = self.create_test_file('exists_copy.txt')
        self.assertTrue(os.path.exists(filename))
        backup_name = create_backup(filename, rename=False)
        self.assertTrue(os.path.exists(filename))
        self.assertTrue(os.path.exists(backup_name))

    def test_multiple_backups(self):
        filename = self.create_test_file('multiple_backups.txt')
        backup_name1 = create_backup(filename, rename=False)
        backup_name2 = create_backup(filename, rename=False)
        self.assertNotEqual(backup_name1, backup_name2)
        self.assertTrue(os.path.exists(filename))
        self.assertTrue(os.path.exists(backup_name1))
        self.assertTrue(os.path.exists(backup_name2))


class TestSafeSerialization(unittest.TestCase):
    def test_serialize_deserialize_1d(self):
        arr = np.array([1.0, np.nan, 3.0])
        serialized = safe_serialize_array(arr)
        self.assertEqual(serialized, [1.0, None, 3.0])
        deserialized = safe_deserialize_array(serialized)
        np.testing.assert_array_equal(np.isnan(arr), np.isnan(deserialized))
        np.testing.assert_allclose(
            np.nan_to_num(arr, nan=0.0),
            np.nan_to_num(deserialized, nan=0.0),
            rtol=1e-10,
        )

    def test_serialize_deserialize_2d(self):
        arr = np.array([[1.0, np.nan], [2.0, 3.0]])
        serialized = safe_serialize_array(arr)
        self.assertEqual(serialized, [[1.0, None], [2.0, 3.0]])
        deserialized = safe_deserialize_array(serialized)
        np.testing.assert_array_equal(np.isnan(arr), np.isnan(deserialized))
        np.testing.assert_allclose(
            np.nan_to_num(arr, nan=0.0),
            np.nan_to_num(deserialized, nan=0.0),
            rtol=1e-10,
        )

    def test_invalid_input_serialize(self):
        with self.assertRaises(TypeError):
            safe_serialize_array("not an array")

    def test_invalid_input_deserialize(self):
        with self.assertRaises(TypeError):
            safe_deserialize_array("not a list")


if __name__ == '__main__':
    unittest.main()
