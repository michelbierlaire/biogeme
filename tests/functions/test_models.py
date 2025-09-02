"""
Test the models module

Michel Bierlaire
Mon Aug 18 2025, 10:00:14
"""

import unittest

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Beta,
    BinaryMax,
    BinaryMin,
    Expression,
    Numeric,
    Variable,
)
from biogeme.models import (
    ordered_logit,
    ordered_logit_from_thresholds,
    ordered_probit,
    ordered_probit_from_thresholds,
    piecewise_as_variable,
    piecewise_formula,
    piecewise_function,
    piecewise_variables,
)


class TestOrderedLogit(unittest.TestCase):
    def test_two_discrete_values(self):
        result = ordered_logit(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2],
            reference_threshold_parameter=Beta('tau', 0, None, None, 0),
            scale_parameter=1.0,
        )
        self.assertIsInstance(result, dict)

    def test_multiple_discrete_values(self):
        result = ordered_logit(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2, 3, 4],
            reference_threshold_parameter=Beta('tau', 0, None, None, 0),
            scale_parameter=1.0,
        )
        self.assertIsInstance(result, dict)

    def test_not_a_parameter(self):
        with self.assertRaises(BiogemeError):
            ordered_logit(
                continuous_value=Numeric(10),
                list_of_discrete_values=[1, 2, 3],
                reference_threshold_parameter=1,
                scale_parameter=1.0,
            )


class TestOrderedProbit(unittest.TestCase):
    def test_two_discrete_values(self):
        result = ordered_probit(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2],
            reference_threshold_parameter=Beta('tau', 0, None, None, 0),
            scale_parameter=1.0,
        )
        self.assertIsInstance(result, dict)

    def test_multiple_discrete_values(self):
        result = ordered_probit(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2, 3, 4],
            reference_threshold_parameter=Beta('tau', 0, None, None, 0),
            scale_parameter=1.0,
        )
        self.assertIsInstance(result, dict)

    def test_not_a_parameter(self):
        with self.assertRaises(BiogemeError):
            ordered_probit(
                continuous_value=Numeric(10),
                list_of_discrete_values=[1, 2, 3],
                reference_threshold_parameter=1,
                scale_parameter=1.0,
            )


class TestOrderedFromThresholds(unittest.TestCase):
    def test_logit_two_discrete_values_from_thresholds(self):
        thresholds = [Beta('tau1', 0, None, None, 0)]
        result = ordered_logit_from_thresholds(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2],
            threshold_parameters=thresholds,
            scale_parameter=Numeric(1.0),
        )
        self.assertIsInstance(result, dict)

    def test_logit_multiple_discrete_values_from_thresholds(self):
        thresholds = [
            Beta('tau1', 0, None, None, 0),
            Beta('tau2', 0, None, None, 0),
            Beta('tau3', 0, None, None, 0),
        ]
        result = ordered_logit_from_thresholds(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2, 3, 4],
            threshold_parameters=thresholds,
            scale_parameter=Numeric(1.0),
        )
        self.assertIsInstance(result, dict)

    def test_logit_threshold_length_mismatch_raises(self):
        thresholds = [
            Beta('tau1', 0, None, None, 0),
            Beta('tau2', 0, None, None, 0),
        ]
        with self.assertRaises(BiogemeError):
            ordered_logit_from_thresholds(
                continuous_value=Numeric(10),
                list_of_discrete_values=[1, 2, 3, 4],
                threshold_parameters=thresholds,
                scale_parameter=Numeric(1.0),
            )

    def test_probit_two_discrete_values_from_thresholds(self):
        thresholds = [Beta('tau1', 0, None, None, 0)]
        result = ordered_probit_from_thresholds(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2],
            threshold_parameters=thresholds,
            scale_parameter=Numeric(1.0),
        )
        self.assertIsInstance(result, dict)

    def test_probit_multiple_discrete_values_from_thresholds(self):
        thresholds = [
            Beta('tau1', 0, None, None, 0),
            Beta('tau2', 0, None, None, 0),
            Beta('tau3', 0, None, None, 0),
        ]
        result = ordered_probit_from_thresholds(
            continuous_value=Numeric(10),
            list_of_discrete_values=[1, 2, 3, 4],
            threshold_parameters=thresholds,
            scale_parameter=Numeric(1.0),
        )
        self.assertIsInstance(result, dict)

    def test_probit_threshold_length_mismatch_raises(self):
        thresholds = [
            Beta('tau1', 0, None, None, 0),
            Beta('tau2', 0, None, None, 0),
        ]
        with self.assertRaises(BiogemeError):
            ordered_probit_from_thresholds(
                continuous_value=Numeric(10),
                list_of_discrete_values=[1, 2, 3, 4],
                threshold_parameters=thresholds,
                scale_parameter=Numeric(1.0),
            )


class TestPiecewiseVariables(unittest.TestCase):
    def test_correct_input(self):
        variable = 'x'
        thresholds = [None, 10, 20, None]
        expected_number_of_variables = len(thresholds) - 1
        result = piecewise_variables(variable, thresholds)
        expected_result = [
            BinaryMin(Variable('x'), 10.0),
            BinaryMax(0.0, BinaryMin((Variable('x') - 10.0), 10.0)),
            BinaryMax(0.0, (Variable('x') - 20.0)),
        ]
        self.assertEqual(len(result), expected_number_of_variables)
        for expected, obtained in zip(expected_result, result):
            self.assertEqual(
                repr(expected),
                repr(obtained),
            )

    def test_all_none_thresholds(self):
        variable = 'x'
        thresholds = [None, None, None]
        with self.assertRaises(BiogemeError):
            piecewise_variables(variable, thresholds)

    def test_invalid_none_thresholds(self):
        variable = 'x'
        thresholds = [None, None, 20, 30]
        with self.assertRaises(BiogemeError):
            piecewise_variables(variable, thresholds)

    def test_wrong_variable_type(self):
        variable = 123  # Not a string or Expression
        thresholds = [10, 20]
        expected_error_msg = "Expression of type Variable expected, not <class 'int'>"
        with self.assertRaisesRegex(BiogemeError, expected_error_msg):
            piecewise_variables(variable, thresholds)

    def test_edge_case_empty_thresholds(self):
        variable = 'x'
        thresholds = []
        expected_error_msg = 'No threshold has been provided.'
        with self.assertRaisesRegex(BiogemeError, expected_error_msg):
            piecewise_variables(variable, thresholds)


class TestPiecewiseFormula(unittest.TestCase):
    def test_valid_input_without_betas(self):
        # Test to ensure function returns correct Expression with valid inputs and no betas
        variable = 'x'
        thresholds = [None, 10, 20, None]
        result = piecewise_formula(variable, thresholds)
        expected_result = 'MultipleSum'
        self.assertTrue(str(result).startswith(expected_result))

    def test_valid_input_with_betas(self):
        # Test to ensure function returns correct Expression with valid inputs and betas
        variable = 'x'
        thresholds = [None, 10, 20, None]
        beta_1 = Beta('beta_1', 0, None, None, 0)
        beta_2 = Beta('beta_2', 0, None, None, 0)
        beta_3 = Beta('beta_3', 0, None, None, 0)
        betas = [beta_1, beta_2, beta_3]
        result = piecewise_formula(
            variable=variable, thresholds=thresholds, betas=betas
        )
        expected_result = 'MultipleSum'
        self.assertTrue(str(result).startswith(expected_result))

    def test_variable_not_variable_or_str(self):
        # Test to check function raises an error if `variable` is neither a Variable instance nor a string
        variable = 123  # Incorrect type
        thresholds = [10, 20]
        with self.assertRaisesRegex(
            BiogemeError,
            'The first argument of piecewiseFormula must be the name of a variable, or the variable itself.',
        ):
            piecewise_formula(variable, thresholds)

    def test_all_none_thresholds(self):
        # Test to check function raises an error if all thresholds are None
        variable = 'x'
        thresholds = [None, None, None]
        with self.assertRaisesRegex(
            BiogemeError,
            'All thresholds for the piecewise linear specification are set to None.',
        ):
            piecewise_formula(variable, thresholds)

    def test_invalid_none_in_thresholds(self):
        # Test to check function raises an error if None is used incorrectly in thresholds
        variable = 'x'
        thresholds = [None, None, 20, 30]
        with self.assertRaisesRegex(
            BiogemeError, 'only the first and the last thresholds can be None'
        ):
            piecewise_formula(variable, thresholds)

    def test_incorrect_betas_length(self):
        # Test to check function raises an error if the length of betas does not match the expected
        variable = 'x'
        thresholds = [None, 10, 20, None]
        beta_1 = Beta('beta_1', 0, None, None, 0)
        beta_2 = Beta('beta_2', 0, None, None, 0)
        betas = [beta_1, beta_2]
        expected_error_msg = 'As there are 4 thresholds, a total of 3 Beta parameters are needed, and not 2.'
        with self.assertRaisesRegex(BiogemeError, expected_error_msg):
            piecewise_formula(variable, thresholds, betas)


class TestPiecewiseAsVariable(unittest.TestCase):
    def test_valid_input_without_betas(self):
        variable = Variable('x')
        thresholds = [None, 10, 20, None]
        # Call the function under test
        result = piecewise_as_variable(variable.name, thresholds)
        # Assert the result is an Expression
        self.assertIsInstance(result, Expression)
        expected_result = '(BinaryMin(x, `10.0`)'
        self.assertTrue(str(result).startswith(expected_result))

    def test_valid_input_with_betas(self):
        variable = 'x'
        thresholds = [None, 10, 20, None]
        beta_1 = Beta('beta_1', 0, None, None, 0)
        beta_2 = Beta('beta_2', 0, None, None, 0)
        betas = [beta_1, beta_2]
        result = piecewise_as_variable(variable, thresholds, betas)
        self.assertIsInstance(result, Expression)
        expected_result = '(BinaryMin(x, `10.0`)'
        self.assertTrue(str(result).startswith(expected_result))

    def test_invalid_variable_type(self):
        variable = 123  # Incorrect type
        thresholds = [10, 20]
        with self.assertRaises(BiogemeError):
            piecewise_as_variable(variable, thresholds)

    def test_all_none_thresholds_error_message(self):
        variable = 'x'
        thresholds = [None, None, None]
        with self.assertRaisesRegex(
            BiogemeError,
            'All thresholds for the piecewise linear specification are set to None.',
        ):
            piecewise_as_variable(variable, thresholds)

    def test_incorrect_betas_length_error_message(self):
        variable = 'x'
        thresholds = [None, 10, 20, None]
        betas = [
            Beta(f'beta_{variable}_minus_inf_10', 0, None, None, 0)
        ]  # Incorrect number
        with self.assertRaisesRegex(
            BiogemeError, 'a total of 2 Beta parameters are needed, and not 1.'
        ):
            piecewise_as_variable(variable, thresholds, betas)


class TestPiecewiseFunction(unittest.TestCase):
    def test_single_interval(self):
        x = 5
        thresholds = [None, 10]  # -inf to 10
        betas = [1]  # Only one beta, as there's only one interval
        result = piecewise_function(x, thresholds, betas)
        self.assertEqual(
            result,
            5,
            "Should match the input x because it's within the interval and beta=1.",
        )

    def test_multiple_intervals(self):
        x = 15
        thresholds = [None, 10, 20, None]  # -inf to 10, 10 to 20, 20 to inf
        betas = [1, 2, 0.5]  # Different betas for each interval
        result = piecewise_function(x, thresholds, betas)
        self.assertEqual(
            10 + 5 * 2, result, 'Should accumulate values across intervals correctly.'
        )

    def test_below_first_threshold(self):
        x = -5
        thresholds = [0, 10]  # 0 to 10
        betas = [1]  # Only one beta, as there's only one interval
        result = piecewise_function(x, thresholds, betas)
        self.assertEqual(result, 0, 'Should be 0 as x is below the first threshold.')

    def test_above_last_threshold(self):
        x = 25
        thresholds = [0, 10, 20, None]  # 0 to 10, 10 to 20, 20 to inf
        betas = [1, 2, 3]  # Different betas for each interval
        result = piecewise_function(x, thresholds, betas)
        self.assertEqual(
            result,
            10 + 20 + 15,
            'Should handle values above the last threshold correctly.',
        )

    def test_all_none_thresholds(self):
        x = 5
        thresholds = [None, None]  # Invalid configuration
        betas = [1]
        with self.assertRaisesRegex(
            BiogemeError,
            'All thresholds for the piecewise linear specification are set to None.',
        ):
            piecewise_function(x, thresholds, betas)

    def test_incorrect_betas_length(self):
        x = 5
        thresholds = [None, 10, 20, None]
        betas = [1, 2]  # Incorrect number of betas
        with self.assertRaisesRegex(
            BiogemeError,
            'a total of 3 values are needed to initialize the parameters. But 2 are provided',
        ):
            piecewise_function(x, thresholds, betas)

    def test_invalid_none_in_thresholds(self):
        x = 5
        thresholds = [0, None, 20]  # Invalid configuration
        betas = [1, 2]
        with self.assertRaisesRegex(
            BiogemeError,
            'For piecewise linear specification, only the first and the last thresholds can be None',
        ):
            piecewise_function(x, thresholds, betas)


if __name__ == '__main__':
    unittest.main()
