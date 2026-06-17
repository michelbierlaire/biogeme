"""
Test the function creating LaTeX code

Michel Bierlaire
Tue Oct 1 19:18:45 2024
"""

import unittest

from biogeme.results_processing import latex_output
from biogeme.results_processing.latex_output import (
    get_latex_estimated_parameters,
    get_sign_for_p_value,
)


class TestGroupedLatexOutput(unittest.TestCase):
    def setUp(self):
        from test_html_output import TestHTMLGeneration

        html_test_case = TestHTMLGeneration()
        html_test_case.setUp()
        self.estimation_results = html_test_case.estimation_results

    def test_get_latex_estimated_parameters(self):
        result = get_latex_estimated_parameters(self.estimation_results)

        self.assertIsInstance(result, dict)
        self.assertIn('', result)
        self.assertNotIn('Estimated parameters', result)

        table = result['']
        self.assertIn('beta1', table)
        self.assertIn('beta2', table)

    def test_get_latex_estimated_parameters_by_group(self):
        result = get_latex_estimated_parameters(
            self.estimation_results,
            group_of_parameters={'First group': ['beta1']},
        )

        self.assertIn('First group', result)
        self.assertIn('Other parameters', result)

        self.assertIn('beta1', result['First group'])
        self.assertNotIn('beta2', result['First group'])

        self.assertIn('beta2', result['Other parameters'])
        self.assertNotIn('beta1', result['Other parameters'])

    def test_get_latex_estimated_parameters_parameter_in_several_groups(self):
        result = get_latex_estimated_parameters(
            self.estimation_results,
            group_of_parameters={
                'First group': ['beta1'],
                'Second group': ['beta1', 'beta2'],
            },
        )

        self.assertIn('First group', result)
        self.assertIn('Second group', result)
        self.assertNotIn('Other parameters', result)

        self.assertIn('beta1', result['First group'])
        self.assertNotIn('beta2', result['First group'])

        self.assertIn('beta1', result['Second group'])
        self.assertIn('beta2', result['Second group'])

    def test_generate_latex_file_content_without_parameter_groups(self):
        content = latex_output.generate_latex_file_content(
            estimation_results=self.estimation_results,
            filename='testfile.tex',
            variance_covariance_type=self.estimation_results.get_default_variance_covariance_matrix(),
        )

        self.assertIn('\\section{Estimated parameters}', content)
        self.assertNotIn('\\subsection{Estimated parameters}', content)
        self.assertIn('beta1', content)
        self.assertIn('beta2', content)

    def test_generate_latex_file_content_with_parameter_groups(self):
        content = latex_output.generate_latex_file_content(
            estimation_results=self.estimation_results,
            filename='testfile_groups.tex',
            variance_covariance_type=self.estimation_results.get_default_variance_covariance_matrix(),
            group_of_parameters={'First group': ['beta1']},
        )

        self.assertIn('\\section{Estimated parameters}', content)
        self.assertIn('\\subsection{First group}', content)
        self.assertIn('\\subsection{Other parameters}', content)

        first_group_position = content.index('\\subsection{First group}')
        other_parameters_position = content.index('\\subsection{Other parameters}')
        first_group_section = content[first_group_position:other_parameters_position]
        other_parameters_section = content[other_parameters_position:]

        self.assertIn('beta1', first_group_section)
        self.assertNotIn('beta2', first_group_section)
        self.assertNotIn('beta1', other_parameters_section)
        self.assertIn('beta2', other_parameters_section)


class TestAddTrailingZero(unittest.TestCase):
    def test_string_without_period(self):
        """Test case for numbers without a period (should append .0)."""
        self.assertEqual(latex_output.add_trailing_zero('123'), '123.0')
        self.assertEqual(latex_output.add_trailing_zero('456'), '456.0')
        self.assertEqual(latex_output.add_trailing_zero('0'), '0.0')

    def test_string_with_period(self):
        """Test case for numbers already containing a period (should remain unchanged)."""
        self.assertEqual(latex_output.add_trailing_zero('123.45'), '123.45')
        self.assertEqual(latex_output.add_trailing_zero('0.123'), '0.123')
        self.assertEqual(latex_output.add_trailing_zero('100.'), '100.0')

    def test_empty_string(self):
        """Test case for empty string input (should append .0)."""
        self.assertEqual(latex_output.add_trailing_zero(''), '0.0')


class TestFormatRealNumber(unittest.TestCase):
    def test_non_scientific_format(self):
        # Test values that don't need scientific notation
        self.assertEqual(latex_output.format_real_number(123.456), '123.0')
        self.assertEqual(latex_output.format_real_number(0.0123456), '0.0123')
        self.assertEqual(latex_output.format_real_number(999.999), '1.00e+03')

    def test_scientific_lowercase_e(self):
        # Test values that result in scientific notation with 'e'
        self.assertEqual(latex_output.format_real_number(1.2345e-5), '1.23e-05')
        self.assertEqual(latex_output.format_real_number(1e-10), '1.00e-10')
        self.assertEqual(latex_output.format_real_number(3.1415e20), '3.14e+20')

    def test_scientific_uppercase_E(self):
        # Test values that result in scientific notation with 'E'
        # If necessary, you can modify format_real_number to support 'E' if required
        self.assertEqual(latex_output.format_real_number(2.5e-4), '0.000250')
        self.assertEqual(latex_output.format_real_number(9.8765e10), '9.88e+10')

    def test_edge_cases(self):
        # Test very small and very large numbers
        self.assertEqual(latex_output.format_real_number(1e-100), '1.00e-100')
        self.assertEqual(latex_output.format_real_number(1e100), '1.00e+100')

    def test_zero_and_near_zero(self):
        # Test zero and very small values close to zero
        self.assertEqual(latex_output.format_real_number(0), '0.00')
        self.assertEqual(latex_output.format_real_number(0.00000001), '1.00e-08')

    def test_negative_numbers(self):
        # Test negative values
        self.assertEqual(latex_output.format_real_number(-123.456), '-123.0')
        self.assertEqual(latex_output.format_real_number(-0.000123), '-0.000123')
        self.assertEqual(latex_output.format_real_number(-1.2345e-5), '-1.23e-05')

    def test_trailing_zeros(self):
        # Test proper formatting of numbers that require trailing zeros
        self.assertEqual(latex_output.format_real_number(1), '1.00')
        self.assertEqual(latex_output.format_real_number(10), '10.0')
        self.assertEqual(latex_output.format_real_number(1.234), '1.23')


class TestGetSignForPValue(unittest.TestCase):
    def test_valid_thresholds(self):
        p_thresholds = [(0.01, '***'), (0.05, '**'), (0.1, '*'), (0.2, '+')]
        self.assertEqual(
            get_sign_for_p_value(0.005, p_thresholds), '***'
        )  # Smallest p-value
        self.assertEqual(
            get_sign_for_p_value(0.03, p_thresholds), '**'
        )  # Between 0.01 and 0.05
        self.assertEqual(
            get_sign_for_p_value(0.07, p_thresholds), '*'
        )  # Between 0.05 and 0.1
        self.assertEqual(
            get_sign_for_p_value(0.15, p_thresholds), '+'
        )  # Between 0.1 and 0.2
        self.assertEqual(get_sign_for_p_value(0.25, p_thresholds), '')  # No match

    def test_string(self):
        p_thresholds = [(0.01, '***'), (0.05, '**'), (0.1, '*'), (0.2, '+')]
        self.assertEqual(
            get_sign_for_p_value('0.005', p_thresholds), '***'
        )  # Smallest p-value
        p_thresholds = [(0.01, 12), (0.05, '**'), (0.1, '*'), (0.2, '+')]
        self.assertEqual(
            get_sign_for_p_value(0.005, p_thresholds), '12'
        )  # Smallest p-value

    def test_exact_threshold_matches(self):
        p_thresholds = [(0.01, '***'), (0.05, '**'), (0.1, '*'), (0.2, '+')]
        self.assertEqual(get_sign_for_p_value(0.01, p_thresholds), '***')
        self.assertEqual(get_sign_for_p_value(0.05, p_thresholds), '**')
        self.assertEqual(get_sign_for_p_value(0.1, p_thresholds), '*')
        self.assertEqual(get_sign_for_p_value(0.2, p_thresholds), '+')

    def test_empty_threshold_list(self):
        self.assertEqual(get_sign_for_p_value(0.05, []), '')  # No thresholds given

    def test_invalid_p_value(self):
        p_thresholds = [(0.01, '***'), (0.05, '**'), (0.1, '*')]

        with self.assertRaises(TypeError):
            get_sign_for_p_value(None, p_thresholds)  # None instead of float

        with self.assertRaises(TypeError):
            get_sign_for_p_value('a_string', p_thresholds)  # None instead of float

    def test_invalid_threshold_structure(self):
        with self.assertRaises(TypeError):
            get_sign_for_p_value(0.05, [0.01, 0.05, 0.1])  # Not a list of tuples
        with self.assertRaises(TypeError):
            get_sign_for_p_value(
                0.05, [(0.01, '***'), ('invalid', '**')]
            )  # Non-float threshold

    def test_unsorted_thresholds(self):
        p_thresholds = [(0.1, '*'), (0.01, '***'), (0.05, '**')]  # Unsorted list
        self.assertEqual(
            get_sign_for_p_value(0.03, p_thresholds), '**'
        )  # Finds correct minimum threshold


if __name__ == '__main__':
    unittest.main()
