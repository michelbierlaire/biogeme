"""
Test the function creating LaTeX code

Michel Bierlaire
Tue Oct 1 19:18:45 2024
"""

import unittest

from biogeme.results_processing import latex_output


class TestAddTrailingZero(unittest.TestCase):

    def test_string_without_period(self):
        """Test case for numbers without a period (should append .0)."""
        self.assertEqual(latex_output.add_trailing_zero("123"), "123.0")
        self.assertEqual(latex_output.add_trailing_zero("456"), "456.0")
        self.assertEqual(latex_output.add_trailing_zero("0"), "0.0")

    def test_string_with_period(self):
        """Test case for numbers already containing a period (should remain unchanged)."""
        self.assertEqual(latex_output.add_trailing_zero("123.45"), "123.45")
        self.assertEqual(latex_output.add_trailing_zero("0.123"), "0.123")
        self.assertEqual(latex_output.add_trailing_zero("100."), "100.0")

    def test_empty_string(self):
        """Test case for empty string input (should append .0)."""
        self.assertEqual(latex_output.add_trailing_zero(""), "0.0")


class TestFormatRealNumber(unittest.TestCase):

    def test_non_scientific_format(self):
        # Test values that don't need scientific notation
        self.assertEqual(latex_output.format_real_number(123.456), '123.0')
        self.assertEqual(latex_output.format_real_number(0.0123456), '0.0123')
        self.assertEqual(latex_output.format_real_number(999.999), '1.0e+03')

    def test_scientific_lowercase_e(self):
        # Test values that result in scientific notation with 'e'
        self.assertEqual(latex_output.format_real_number(1.2345e-5), '1.23e-05')
        self.assertEqual(latex_output.format_real_number(1e-10), '1.0e-10')
        self.assertEqual(latex_output.format_real_number(3.1415e20), '3.14e+20')

    def test_scientific_uppercase_E(self):
        # Test values that result in scientific notation with 'E'
        # If necessary, you can modify format_real_number to support 'E' if required
        self.assertEqual(latex_output.format_real_number(2.5e-4), '0.00025')
        self.assertEqual(latex_output.format_real_number(9.8765e10), '9.88e+10')

    def test_edge_cases(self):
        # Test very small and very large numbers
        self.assertEqual(latex_output.format_real_number(1e-100), '1.0e-100')
        self.assertEqual(latex_output.format_real_number(1e100), '1.0e+100')

    def test_zero_and_near_zero(self):
        # Test zero and very small values close to zero
        self.assertEqual(latex_output.format_real_number(0), '0.0')
        self.assertEqual(latex_output.format_real_number(0.00000001), '1.0e-08')

    def test_negative_numbers(self):
        # Test negative values
        self.assertEqual(latex_output.format_real_number(-123.456), '-123.0')
        self.assertEqual(latex_output.format_real_number(-0.000123), '-0.000123')
        self.assertEqual(latex_output.format_real_number(-1.2345e-5), '-1.23e-05')

    def test_trailing_zeros(self):
        # Test proper formatting of numbers that require trailing zeros
        self.assertEqual(latex_output.format_real_number(1), '1.0')
        self.assertEqual(latex_output.format_real_number(10), '10.0')
        self.assertEqual(latex_output.format_real_number(1.234), '1.23')


if __name__ == "__main__":
    unittest.main()
