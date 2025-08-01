"""Tests for logical expressions

Michel Bierlaire
Sat Jun 21 2025, 10:57:30
"""

import unittest

from biogeme.expressions import Numeric
from biogeme.expressions.logical_and import And


class TestComparisonExpressions(unittest.TestCase):
    def setUp(self):
        self.a = Numeric(3)
        self.b = Numeric(0)
        self.c = Numeric(3)

    def test_and(self):
        expr = self.a & self.c
        self.assertIsInstance(expr, And)
        self.assertEqual(expr.get_value(), 1.0)
        expr_2 = self.a & self.b
        self.assertEqual(expr_2.get_value(), 0.0)
        expr_3 = self.b & self.a
        self.assertEqual(expr_3.get_value(), 0.0)
