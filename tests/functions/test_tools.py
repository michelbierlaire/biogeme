"""
Test the tools module

:author: Michel Bierlaire
:date: Sun Aug  8 17:42:48 2021
"""
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from biogeme import tools
import biogeme.exceptions as excep
from test_data import (
    input_flatten,
    output_flatten_1,
    output_flatten_2,
    output_flatten_3,
)


def myFunction(x):
    f = np.log(x[0]) + np.exp(x[1])
    g = np.empty(2)
    g[0] = 1.0 / x[0]
    g[1] = np.exp(x[1])
    H = np.empty((2, 2))
    H[0, 0] = -1.0 / x[0] ** 2
    H[0, 1] = 0.0
    H[1, 0] = 0.0
    H[1, 1] = np.exp(x[1])
    return f, g, H


class TestTools(unittest.TestCase):
    def test_findiff_g(self):
        x = np.array([1.1, 1.1])
        g_fd = tools.findiff_g(myFunction, x)
        np.testing.assert_almost_equal(g_fd, [0.90909087, 3.00416619])

    def test_findiff_H(self):
        x = np.array([1.1, 1.1])
        H_fd = tools.findiff_H(myFunction, x)
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
        )

    def test_checkDerivatives(self):
        x = np.array([1.1, -1.5])
        _, _, _, gdiff, hdiff = tools.checkDerivatives(
            myFunction, x, names=['First', 'Second']
        )
        np.testing.assert_almost_equal(gdiff, [0, 0])
        np.testing.assert_almost_equal(hdiff, [[0, 0], [0, 0]])

        x = np.array([3.2, 1.32])
        _, _, _, gdiff, hdiff = tools.checkDerivatives(
            myFunction, x, names=['First', 'Second']
        )
        np.testing.assert_almost_equal(gdiff, [0, 0], decimal=5)
        np.testing.assert_almost_equal(hdiff, [[0, 0], [0, 0]], decimal=5)

    def test_getPrimeNumbers(self):
        result = tools.get_prime_numbers(7)
        self.assertListEqual(result, [2, 3, 5, 7, 11, 13, 17])

        with self.assertRaises(excep.BiogemeError):
            result = tools.get_prime_numbers(0)

        with self.assertRaises(excep.BiogemeError):
            result = tools.get_prime_numbers(-1)

        with self.assertRaises(excep.BiogemeError):
            result = tools.get_prime_numbers(0.3)

    def test_calculatePrimeNumbers(self):
        result = tools.calculate_prime_numbers(10)
        self.assertListEqual(result, [2, 3, 5, 7])
        result = tools.calculate_prime_numbers(0)
        self.assertListEqual(result, [])

        with self.assertRaises(excep.BiogemeError):
            result = tools.calculate_prime_numbers(-1)

        with self.assertRaises(excep.BiogemeError):
            result = tools.calculate_prime_numbers(0.3)

    def test_countNumberOfGroups(self):
        df = pd.DataFrame(
            {
                'ID': [1, 1, 2, 3, 3, 1, 2, 3],
                'value': [1000, 2000, 3000, 4000, 5000, 5000, 10000, 20000],
            }
        )
        nbr = tools.countNumberOfGroups(df, 'ID')
        self.assertEqual(nbr, 6)
        nbr = tools.countNumberOfGroups(df, 'value')
        self.assertEqual(nbr, 7)

    def test_likelihood_ratio_test(self):
        model1 = (-1340.8, 5)
        model2 = (-1338.49, 7)
        r = tools.likelihood_ratio_test(model1, model2)
        self.assertSequenceEqual(
            r,
            (
                'H0 cannot be rejected at level 5.0%',
                4.619999999999891,
                5.991464547107979,
            ),
        )
        r = tools.likelihood_ratio_test(model2, model1)
        self.assertSequenceEqual(
            r,
            (
                'H0 cannot be rejected at level 5.0%',
                4.619999999999891,
                5.991464547107979,
            ),
        )
        r = tools.likelihood_ratio_test(model1, model2, significance_level=0.1)
        self.assertSequenceEqual(
            r,
            (
                'H0 can be rejected at level 10.0%',
                4.619999999999891,
                4.605170185988092,
            ),
        )
        with self.assertRaises(excep.BiogemeError):
            model1 = (-1340.8, 7)
            model2 = (-1338.49, 5)
            tools.likelihood_ratio_test(model1, model2)

    def test_flatten_database(self):
        df = deepcopy(input_flatten)
        result_1 = tools.flatten_database(df, 'ID', row_name='Name')
        output_flatten_1.index.name = 'ID'
        pd.testing.assert_frame_equal(result_1, output_flatten_1)
        result_2 = tools.flatten_database(input_flatten, 'ID')
        output_flatten_2.index.name = 'ID'
        pd.testing.assert_frame_equal(result_2, output_flatten_2)
        result_3 = tools.flatten_database(df, 'ID', identical_columns=[])
        output_flatten_3.index.name = 'ID'
        pd.testing.assert_frame_equal(result_3, output_flatten_3)

    def test_temporary_file(self):
        content = 'a random sentence'
        with tools.TemporaryFile() as filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            with open(filename, 'r', encoding='utf-8') as f:
                check = f.read()
            self.assertEqual(content, check)

    def test_model_names(self):
        model_names = tools.ModelNames(prefix='Test')

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
        dict_1 = tools.generate_unique_ids(list_1)
        self.assertDictEqual(correct_dict_1, dict_1)

        list_2 = ['A', 'B', 'B']
        correct_dict_2 = {'A': 'A', 'B_0': 'B', 'B_1': 'B'}
        dict_2 = tools.generate_unique_ids(list_2)
        self.assertDictEqual(correct_dict_2, dict_2)

        list_3 = ['B', 'B', 'B']
        correct_dict_3 = {'B_2': 'B', 'B_0': 'B', 'B_1': 'B'}
        dict_3 = tools.generate_unique_ids(list_3)
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
        result = list(tools.unique_product(*iterables))
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
        result = list(tools.unique_product(*iterables))
        self.assertEqual(result, expected_output)

        # Test case with empty iterable
        iterables = [range(3), [], range(2)]
        expected_output = []
        result = list(tools.unique_product(*iterables))
        self.assertEqual(result, expected_output)

        # Test case with single iterable
        iterables = [range(3)]
        expected_output = [(0,), (1,), (2,)]
        result = list(tools.unique_product(*iterables))
        self.assertEqual(result, expected_output)


if __name__ == '__main__':
    unittest.main()
