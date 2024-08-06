import unittest

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Numeric, Expression
from biogeme.mdcev import Generalized


class TestGeneralizedMdcev(unittest.TestCase):

    def setUp(self):
        self.model_name = 'TestModel'
        self.baseline_utilities = {1: Numeric(1), 2: Numeric(2)}
        self.gamma_parameters = {1: Numeric(0.5), 2: None}
        self.alpha_parameters = {1: Numeric(0.2), 2: Numeric(0.5)}
        self.scale_parameter = Numeric(0.1)
        self.weights = Numeric(1.0)
        self.instance = Generalized(
            model_name=self.model_name,
            baseline_utilities=self.baseline_utilities,
            gamma_parameters=self.gamma_parameters,
            alpha_parameters=self.alpha_parameters,
            scale_parameter=self.scale_parameter,
            weights=self.weights,
        )

    def test_utility_one_alternative_with_gamma(self):
        the_id = 1
        the_consumption = Numeric(3)
        expected_value = -0.5567281192442506
        utility = self.instance.transformed_utility(the_id, the_consumption)
        self.assertIsInstance(utility, Expression)
        self.assertAlmostEqual(utility.get_value(), expected_value, places=3)
        log_determinant = self.instance.calculate_log_determinant_one_alternative(
            the_id, the_consumption
        )
        self.assertAlmostEqual(
            log_determinant.get_value(), -1.4759065198095778, places=3
        )
        inverse_determinant = (
            self.instance.calculate_inverse_of_determinant_one_alternative(
                the_id, the_consumption
            )
        )
        self.assertAlmostEqual(inverse_determinant.get_value(), 4.375, places=3)

    def test_utility_one_alternative_without_gamma(self):
        the_id = 2
        the_consumption = Numeric(3)
        utility = self.instance.transformed_utility(the_id, the_consumption)
        self.assertAlmostEqual(utility.get_value(), 1.450693855665945, places=3)
        log_determinant = self.instance.calculate_log_determinant_one_alternative(
            the_id, the_consumption
        )
        self.assertAlmostEqual(
            log_determinant.get_value(), -1.791759469228055, places=3
        )
        inverse_determinant = (
            self.instance.calculate_inverse_of_determinant_one_alternative(
                the_id, the_consumption
            )
        )
        self.assertAlmostEqual(inverse_determinant.get_value(), 6.0, places=3)

    def test_calculate_log_determinant_one_alternative_error(self):
        self.assertRaises(
            BiogemeError,
            self.instance.calculate_log_determinant_one_alternative,
            the_id=10,
            consumption=Numeric(2),
        )
