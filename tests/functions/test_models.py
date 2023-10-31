"""
Test the models module

:author: Michel Bierlaire
:date: Fri Jul 21 17:17:37 2023
"""
import unittest
from biogeme.models import ordered_logit, ordered_probit
from biogeme.expressions import Beta, Elem
from biogeme.exceptions import BiogemeError


class TestOrderedLogit(unittest.TestCase):
    def test_two_discrete_values(self):
        result = ordered_logit(
            continuous_value=10,
            list_of_discrete_values=[1, 2],
            tau_parameter=Beta('tau', 0, None, None, 0),
        )
        self.assertIsInstance(result, dict)

    def test_multiple_discrete_values(self):
        result = ordered_logit(
            continuous_value=10,
            list_of_discrete_values=[1, 2, 3, 4],
            tau_parameter=Beta('tau', 0, None, None, 0),
        )
        self.assertIsInstance(result, dict)

    def test_not_a_parameter(self):
        with self.assertRaises(BiogemeError):
            ordered_logit(10, [1, 2, 3], 1)


class TestOrderedProbit(unittest.TestCase):
    def test_two_discrete_values(self):
        result = ordered_probit(
            continuous_value=10,
            list_of_discrete_values=[1, 2],
            tau_parameter=Beta('tau', 0, None, None, 0),
        )
        self.assertIsInstance(result, dict)

    def test_multiple_discrete_values(self):
        result = ordered_probit(
            continuous_value=10,
            list_of_discrete_values=[1, 2, 3, 4],
            tau_parameter=Beta('tau', 0, None, None, 0),
        )
        self.assertIsInstance(result, dict)

    def test_not_a_parameter(self):
        with self.assertRaises(BiogemeError):
            ordered_probit(10, [1, 2, 3], 1)


if __name__ == '__main__':
    unittest.main()
