""" Tests for checks.py

Michel Bierlaire
Fri Apr 5 11:21:26 2024
"""

import unittest

from biogeme.tools.checks import check_consistency_of_named_dicts


class TestCheckConsistencyOfNamedDicts(unittest.TestCase):
    def test_all_dictionaries_consistent(self):
        """Test case where all dictionaries have the same set of keys."""
        dicts = {
            "first": {1: 'a', 2: 'b'},
            "second": {1: 'c', 2: 'd'},
            "third": {1: 'e', 2: 'f'},
        }
        consistent, report = check_consistency_of_named_dicts(dicts)
        self.assertTrue(consistent)
        self.assertIsNone(report)

    def test_dictionaries_with_inconsistencies(self):
        """Test case where dictionaries have different sets of keys."""
        dicts = {
            "first": {1: 'a', 2: 'b'},
            "second": {1: 'c', 3: 'd'},
            "third": {2: 'e', 3: 'f'},
        }
        consistent, report = check_consistency_of_named_dicts(dicts)
        self.assertFalse(consistent)
        self.assertIsNotNone(report)
        self.assertIn("second missing keys compared to first:", report)
        self.assertIn("second has extra keys compared to first:", report)

    def test_empty_dictionaries_consistent(self):
        """Test case where all dictionaries are empty."""
        dicts = {"first": {}, "second": {}, "third": {}}
        consistent, report = check_consistency_of_named_dicts(dicts)
        self.assertTrue(consistent)
        self.assertIsNone(report)

    def test_one_empty_dictionary_inconsistent(self):
        """Test case where one dictionary is empty and others are not."""
        dicts = {"first": {1: 'a'}, "second": {}, "third": {1: 'b'}}
        consistent, report = check_consistency_of_named_dicts(dicts)
        self.assertFalse(consistent)
        self.assertIsNotNone(report)
        self.assertIn("second missing keys compared to first:", report)

    def test_single_dictionary_consistent(self):
        """Test case with only one dictionary."""
        dicts = {"first": {1: 'a', 2: 'b'}}
        consistent, report = check_consistency_of_named_dicts(dicts)
        self.assertTrue(consistent)
        self.assertIsNone(report)


if __name__ == '__main__':
    unittest.main()
