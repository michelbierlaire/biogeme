import unittest

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Numeric, Expression
from biogeme.mdcev import NonMonotonic


class TestNonMonotonicMdcev(unittest.TestCase):

    def setUp(self):
        self.model_name = 'TestModel'
        self.baseline_utilities = {1: Numeric(1), 2: Numeric(2)}
        self.gamma_parameters = {1: Numeric(0.5), 2: None}
        self.alpha_parameters = {1: Numeric(0.1), 2: Numeric(0.5)}
        self.scale_parameter = Numeric(0.1)
        self.weights = Numeric(1.0)
        self.instance = NonMonotonic(
            model_name=self.model_name,
            baseline_utilities=self.baseline_utilities,
            mu_utilities=self.baseline_utilities,
            gamma_parameters=self.gamma_parameters,
            alpha_parameters=self.alpha_parameters,
            scale_parameter=self.scale_parameter,
            weights=self.weights,
        )

    def test_one_alternative_with_gamma(self):
        the_id = 1
        the_consumption = Numeric(3)
        expected_value = 1.471743848695463
        utility = self.instance.transformed_utility(the_id, the_consumption)
        self.assertIsInstance(utility, Expression)
        self.assertAlmostEqual(utility.get_value(), expected_value, places=3)
        log_determinant = self.instance.calculate_log_determinant_one_alternative(
            the_id, the_consumption
        )
        self.assertAlmostEqual(
            log_determinant.get_value(), -2.109442618302976, places=3
        )
        inverse_determinant = (
            self.instance.calculate_inverse_of_determinant_one_alternative(
                the_id, the_consumption
            )
        )
        self.assertAlmostEqual(
            inverse_determinant.get_value(), 8.243645146922482, places=3
        )

    def test_utility_one_alternative_without_gamma(self):
        the_id = 2
        the_consumption = Numeric(3)
        utility = self.instance.transformed_utility(the_id, the_consumption)
        self.assertAlmostEqual(utility.get_value(), 6.266073527774857, places=3)
        log_determinant = self.instance.calculate_log_determinant_one_alternative(
            the_id, the_consumption
        )
        self.assertAlmostEqual(
            log_determinant.get_value(), -0.34106561356210996, places=3
        )
        inverse_determinant = (
            self.instance.calculate_inverse_of_determinant_one_alternative(
                the_id, the_consumption
            )
        )
        self.assertAlmostEqual(
            inverse_determinant.get_value(), 1.4064455197352266, places=3
        )

    def test_calculate_log_determinant_one_alternative_error(self):
        self.assertRaises(
            BiogemeError,
            self.instance.calculate_log_determinant_one_alternative,
            the_id=10,
            consumption=Numeric(2),
        )
