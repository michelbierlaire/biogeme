import unittest

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, Elem, Numeric
from biogeme.jax_calculator import get_value_and_derivatives


class TestElemExpression(unittest.TestCase):

    def test_valid_key(self):
        expr = Elem({1: Numeric(10), 2: Numeric(20)}, Numeric(1))
        self.assertEqual(expr.get_value(), 10)

    def test_valid_negative_key(self):
        expr = Elem({-1: Numeric(5), 0: Numeric(15)}, Numeric(-1))
        self.assertEqual(expr.get_value(), 5)

    def test_invalid_key_raises(self):
        expr = Elem({1: Numeric(10)}, Numeric(3))
        with self.assertRaises(BiogemeError):
            expr.get_value()

    def test_derivative_with_respect_to_parameter(self):
        b1 = Beta('b1', 1.0, None, None, 0)
        b2 = Beta('b2', 2.0, None, None, 0)
        expr = Elem({0: b1 * b1, 1: b2 * b2}, Numeric(1))
        result = get_value_and_derivatives(
            expression=expr, numerically_safe=False, use_jit=True
        )
        value = result.function
        gradient = list(result.gradient)
        expected_gradient = [0, 4]
        self.assertEqual(value, 4.0)
        self.assertListEqual(gradient, expected_gradient)

        expr = Elem({0: b1 * b1, 1: b2 * b2}, Numeric(2))
        result = get_value_and_derivatives(
            expression=expr, numerically_safe=False, use_jit=True
        )
        value = result.function
        gradient = list(result.gradient)
        expected_gradient = [2, 0]
        self.assertEqual(value, 1.0)
        self.assertListEqual(gradient, expected_gradient)


if __name__ == '__main__':
    unittest.main()
