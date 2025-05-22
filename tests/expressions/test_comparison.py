"""Tests for comparison expressions

Michel Bierlaire
Wed Mar 26 13:25:27 2025
"""

import unittest
from biogeme.expressions.comparison_expressions import (
    Equal,
    NotEqual,
    LessOrEqual,
    GreaterOrEqual,
    Less,
    Greater,
)
from biogeme.expressions import Numeric


class TestComparisonExpressions(unittest.TestCase):
    def setUp(self):
        self.a = Numeric(3)
        self.b = Numeric(5)
        self.c = Numeric(3)

    def test_equal(self):
        expr = Equal(self.a, self.c)
        self.assertEqual(expr.get_value(), 1.0)
        self.assertEqual(str(expr), '(`3.0` == `3.0`)')
        self.assertEqual(repr(expr), 'Equal(<Numeric value=3.0>, <Numeric value=3.0>)')

        expr_false = Equal(self.a, self.b)
        self.assertEqual(expr_false.get_value(), 0.0)

    def test_not_equal(self):
        expr = NotEqual(self.a, self.b)
        self.assertEqual(expr.get_value(), 1.0)
        self.assertEqual(str(expr), '(`3.0` != `5.0`)')
        self.assertEqual(
            repr(expr), "NotEqual(<Numeric value=3.0>, <Numeric value=5.0>)"
        )

        expr_false = NotEqual(self.a, self.c)
        self.assertEqual(expr_false.get_value(), 0.0)

    def test_less_or_equal(self):
        expr = LessOrEqual(self.a, self.b)
        self.assertEqual(expr.get_value(), 1.0)
        self.assertEqual(str(expr), '(`3.0` <= `5.0`)')
        self.assertEqual(
            repr(expr), "LessOrEqual(<Numeric value=3.0>, <Numeric value=5.0>)"
        )

        expr_equal = LessOrEqual(self.a, self.c)
        self.assertEqual(expr_equal.get_value(), 1.0)

        expr_false = LessOrEqual(self.b, self.a)
        self.assertEqual(expr_false.get_value(), 0.0)

    def test_greater_or_equal(self):
        expr = GreaterOrEqual(self.b, self.a)
        self.assertEqual(expr.get_value(), 1.0)
        self.assertEqual(str(expr), '(`5.0` >= `3.0`)')
        self.assertEqual(
            repr(expr), "GreaterOrEqual(<Numeric value=5.0>, <Numeric value=3.0>)"
        )

        expr_equal = GreaterOrEqual(self.a, self.c)
        self.assertEqual(expr_equal.get_value(), 1.0)

        expr_false = GreaterOrEqual(self.a, self.b)
        self.assertEqual(expr_false.get_value(), 0.0)

    def test_less(self):
        expr = Less(self.a, self.b)
        self.assertEqual(expr.get_value(), 1.0)
        self.assertEqual(str(expr), '(`3.0` < `5.0`)')
        self.assertEqual(repr(expr), "Less(<Numeric value=3.0>, <Numeric value=5.0>)")

        expr_false = Less(self.b, self.a)
        self.assertEqual(expr_false.get_value(), 0.0)

    def test_greater(self):
        expr = Greater(self.b, self.a)
        self.assertEqual(expr.get_value(), 1.0)
        self.assertEqual(str(expr), '(`5.0` > `3.0`)')
        self.assertEqual(
            repr(expr), "Greater(<Numeric value=5.0>, <Numeric value=3.0>)"
        )

        expr_false = Greater(self.a, self.b)
        self.assertEqual(expr_false.get_value(), 0.0)


if __name__ == "__main__":
    unittest.main()
