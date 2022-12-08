"""
Test the check_parameters module

:author: Michel Bierlaire
:date: Thu Dec  1 16:35:18 2022

"""
import unittest
import biogeme.check_parameters as cp


class TestCheckParameters(unittest.TestCase):
    """Tests for the biogeme.check_parameters module"""

    def test_zero_one(self):
        diag, _ = cp.zero_one(0.5)
        self.assertEqual(diag, True)
        diag, _ = cp.zero_one(-0.5)
        self.assertEqual(diag, False)
        diag, _ = cp.zero_one(1.5)
        self.assertEqual(diag, False)

    def test_is_number(self):
        diag, _ = cp.is_number(1.2)
        self.assertEqual(diag, True)
        diag, _ = cp.is_number(1)
        self.assertEqual(diag, True)
        diag, _ = cp.is_number(-1)
        self.assertEqual(diag, True)
        diag, _ = cp.is_number(-1.2e-4)
        self.assertEqual(diag, True)
        diag, _ = cp.is_number('a_string')
        self.assertEqual(diag, False)

    def test_is_positive(self):
        diag, _ = cp.is_positive(1.2)
        self.assertEqual(diag, True)
        diag, _ = cp.is_positive(0)
        self.assertEqual(diag, False)
        diag, _ = cp.is_positive(-1.2)
        self.assertEqual(diag, False)

    def test_is_integer(self):
        diag, _ = cp.is_integer(1)
        self.assertEqual(diag, True)
        diag, _ = cp.is_integer(1.2)
        self.assertEqual(diag, False)

    def test_check_algo_name(self):
        diag, _ = cp.check_algo_name('scipy')
        self.assertEqual(diag, True)
        diag, _ = cp.check_algo_name('any_string')
        self.assertEqual(diag, False)

    def test_boolean(self):
        diag, _ = cp.is_boolean(True)
        self.assertEqual(diag, True)
        diag, _ = cp.is_boolean(1)
        self.assertEqual(diag, False)
        diag, _ = cp.is_boolean('True')
        self.assertEqual(diag, False)

    def test_is_non_negative(self):
        diag, _ = cp.is_non_negative(1.2)
        self.assertEqual(diag, True)
        diag, _ = cp.is_non_negative(0)
        self.assertEqual(diag, True)
        diag, _ = cp.is_non_negative(-1.2)
        self.assertEqual(diag, False)
