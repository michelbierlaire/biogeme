"""Tests for binary expressions

Michel Bierlaire
Wed Mar 26 11:26:56 2025
"""

import unittest
import warnings

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Numeric, bioMax, bioMin
from biogeme.expressions.binary_max import BinaryMax
from biogeme.expressions.binary_min import BinaryMin
from biogeme.expressions.divide import Divide
from biogeme.expressions.logical_and import And
from biogeme.expressions.logical_or import Or
from biogeme.expressions.minus import Minus
from biogeme.expressions.plus import Plus
from biogeme.expressions.power import Power
from biogeme.expressions.times import Times


class TestComprehensiveBinaryExpressions(unittest.TestCase):
    def setUp(self):
        self.a = Numeric(3)
        self.b = Numeric(2)
        self.zero = Numeric(0)

    def test_expressions(self):
        operators = [
            ('Plus', Plus, '+', lambda x, y: x + y),
            ('Minus', Minus, '-', lambda x, y: x - y),
            ('Times', Times, '*', lambda x, y: x * y),
            ('Divide', Divide, '/', lambda x, y: x / y),
            ('Power', Power, '**', lambda x, y: x**y),
            ('BinaryMin', BinaryMin, 'BinaryMin', min),
            ('BinaryMax', BinaryMax, 'BinaryMax', max),
            ('And', And, 'and', lambda x, y: 1 if x and y else 0),
            ('Or', Or, 'or', lambda x, y: 1 if x or y else 0),
        ]
        for name, op_class, op_symbol, func in operators:
            with self.subTest(operator=name):
                expr = op_class(self.a, self.b)
                expected = func(3, 2)
                # Test get_value
                self.assertEqual(expr.get_value(), expected)
                # Test __str__ contains operator symbol or name
                self.assertIn(str(op_symbol), str(expr))
                # Test __repr__ contains operator name for BinaryMin/Max, else operator symbol
                if name in ['BinaryMin', 'BinaryMax']:
                    self.assertIn(name, repr(expr))
                else:
                    self.assertIn(op_symbol, repr(expr))
                # Test JAX function correctness
                jax_fn = expr.recursive_construct_jax_function(numerically_safe=False)
                self.assertAlmostEqual(
                    float(jax_fn(None, None, None, None)), float(expected), places=5
                )

    def test_divide_by_zero(self):
        expr = Divide(self.a, self.zero)
        with self.assertRaises(BiogemeError):
            expr.get_value()

    def test_biomin_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = bioMin(self.a, self.b)
            self.assertTrue(
                any(issubclass(item.category, DeprecationWarning) for item in w)
            )

    def test_biomax_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = bioMax(self.a, self.b)
            self.assertTrue(
                any(issubclass(item.category, DeprecationWarning) for item in w)
            )


if __name__ == '__main__':
    import unittest

    unittest.main()
