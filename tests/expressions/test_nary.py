"""Tests for nary expressions

Michel Bierlaire
Wed Mar 26 11:46:06 2025
"""

import unittest
import warnings

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Beta,
    ConditionalSum,
    ConditionalTermTuple,
    Elem,
    LinearTermTuple,
    LinearUtility,
    MultipleSum,
    Numeric,
    Variable,
    bioLinearUtility,
    bioMultSum,
)


class TestNaryExpressions(unittest.TestCase):
    def test_conditional_sum_value(self):
        terms = [
            ConditionalTermTuple(Numeric(1), Numeric(10)),
            ConditionalTermTuple(Numeric(0), Numeric(100)),
            ConditionalTermTuple(Numeric(2), Numeric(5)),
        ]
        expr = ConditionalSum(terms)
        self.assertEqual(expr.get_value(), 15.0)

    def test_multiple_sum_value(self):
        expr = MultipleSum([Numeric(1), Numeric(2), Numeric(3)])
        self.assertEqual(expr.get_value(), 6.0)

    def test_elem_valid_key(self):
        mapping = {0: Numeric(1), 1: Numeric(2)}
        expr = Elem(mapping, Numeric(1))
        self.assertEqual(expr.get_value(), 2.0)

    def test_elem_invalid_key(self):
        mapping = {0: Numeric(1)}
        expr = Elem(mapping, Numeric(3))
        with self.assertRaises(BiogemeError):
            expr.get_value()

    def test_linear_utility_value(self):
        b = Beta("beta", 2.0, None, None, 0)
        x = Variable("x")
        x.variable_id = 0
        expr = LinearUtility([LinearTermTuple(b, x)])
        self.assertIn("LinearUtility", repr(expr))

    def test_deprecated_bioMultSum(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = bioMultSum([Numeric(1)])
            self.assertTrue(
                any(issubclass(wi.category, DeprecationWarning) for wi in w)
            )

    def test_deprecated_bioLinearUtility(self):
        b = Beta("b", 1.0, None, None, 0)
        x = Variable("x")
        x.variable_id = 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = bioLinearUtility([LinearTermTuple(b, x)])
            self.assertTrue(
                any(issubclass(wi.category, DeprecationWarning) for wi in w)
            )


if __name__ == '__main__':
    unittest.main()
