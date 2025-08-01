"""Tests for sampling utilities

Michel Bierlaire
Wed Mar 26 21:02:23 2025
"""

import unittest

import pandas as pd

from biogeme.database.sampling import (
    sample_with_replacement,
    sample_panel_with_replacement,
    split_validation_sets,
)
from biogeme.exceptions import BiogemeError


class TestSamplingFunctions(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {'id': [1, 1, 2, 2, 3, 3], 'value': [10, 20, 30, 40, 50, 60]}
        )

        self.individual_map = pd.DataFrame.from_dict(
            {1: [0, 1], 2: [2, 3], 3: [4, 5]}, orient='index'
        )

    def test_sample_with_replacement_default(self):
        result = sample_with_replacement(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(set(result.columns) == set(self.df.columns))

    def test_sample_with_replacement_custom_size(self):
        size = 10
        result = sample_with_replacement(self.df, size=size)
        self.assertEqual(len(result), size)

    def test_sample_panel_with_replacement_default(self):
        result = sample_panel_with_replacement(self.df, self.individual_map)
        self.assertTrue(len(result) % 2 == 0)  # Each individual contributes 2 rows

    def test_sample_panel_with_replacement_custom_size(self):
        result = sample_panel_with_replacement(self.df, self.individual_map, size=5)
        self.assertEqual(len(result) % 2, 0)
        self.assertEqual(len(result) // 2, 5)

    def test_sample_panel_with_replacement_invalid_map(self):
        with self.assertRaises(BiogemeError):
            sample_panel_with_replacement(self.df, None)

        with self.assertRaises(BiogemeError):
            sample_panel_with_replacement(self.df, pd.DataFrame())

    def test_split_validation_sets_no_grouping(self):
        result = split_validation_sets(self.df, slices=3)
        self.assertEqual(len(result), 3)
        for est, val in result:
            self.assertEqual(len(est) + len(val), len(self.df))

    def test_split_validation_sets_with_grouping(self):
        result = split_validation_sets(self.df, slices=2, group_column='id')
        self.assertEqual(len(result), 2)
        for est, val in result:
            self.assertEqual(len(est) + len(val), len(self.df))

    def test_split_validation_sets_invalid_slices(self):
        with self.assertRaises(BiogemeError):
            split_validation_sets(self.df, slices=1)

    def test_split_validation_sets_invalid_group_column(self):
        with self.assertRaises(BiogemeError):
            split_validation_sets(self.df, slices=2, group_column='not_a_column')


if __name__ == '__main__':
    unittest.main()
