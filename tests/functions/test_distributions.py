"""
Test the distributions module

:author: Michel Bierlaire
:date: Sun Nov 19 19:52:22 2023

"""

import unittest
from biogeme.expressions import Numeric
from biogeme.distributions import (
    normalpdf,
    lognormalpdf,
    uniformpdf,
    triangularpdf,
    logisticcdf,
)


class TestNormalPdf(unittest.TestCase):
    def test_normal_values(self):
        # Test with normal values
        self.assertAlmostEqual(normalpdf(0).get_value(), 0.3989422804)
        self.assertAlmostEqual(normalpdf(1, 1, 1).get_value(), 0.3989422804)

    def test_expression_inputs(self):
        expression = normalpdf(1, 1, 1)
        value = expression.get_value()
        self.assertAlmostEqual(value, 0.3989422804)

    def test_negative_sigma(self):
        # Test with negative sigma (should handle or raise an error)
        with self.assertRaises(ValueError):
            normalpdf(0, 0, -1)

    def test_zero_sigma(self):
        # Test with sigma equal to zero (should handle or raise an error)
        with self.assertRaises(ValueError):
            normalpdf(0, 0, 0)

    def test_non_numeric_inputs(self):
        # Test with non-numeric inputs
        with self.assertRaises(TypeError):
            normalpdf("string", "string", "string")

    def test_mixed_inputs(self):
        # Test with mixed type inputs
        self.assertAlmostEqual(normalpdf(0.0, Numeric(0), 1).get_value(), 0.3989422804)
        # Add more mixed type tests here

    def test_large_values(self):
        # Test with very large values
        self.assertAlmostEqual(normalpdf(1000, 0, 1).get_value(), 0)

    def test_small_values(self):
        # Test with very small values
        self.assertAlmostEqual(normalpdf(0, 0, 0.0001).get_value(), 3989.4228034270454)


class TestLognormalPdf(unittest.TestCase):
    def test_normal_values(self):
        # Test with normal values
        self.assertAlmostEqual(lognormalpdf(1, 0, 1).get_value(), 0.3989422804)
        expected_value = 0.04222768901274423
        self.assertAlmostEqual(lognormalpdf(2, 3, 4).get_value(), expected_value)

    def test_expression_inputs(self):
        expression = lognormalpdf(1, 0, 1)
        value = expression.get_value()
        self.assertAlmostEqual(value, 0.3989422804)
        expression = lognormalpdf(2, 3, 4)
        value = expression.get_value()
        self.assertAlmostEqual(value, 0.04222768901274423)

    def test_invalid_x_values(self):
        # Test with invalid x values (x <= 0)
        with self.assertRaises(ValueError):
            lognormalpdf(0, 0, 1)
        with self.assertRaises(ValueError):
            lognormalpdf(-1, 0, 1)

    def test_negative_sigma(self):
        # Test with negative sigma (should handle or raise an error)
        with self.assertRaises(ValueError):
            lognormalpdf(1, 0, -1)

    def test_zero_sigma(self):
        # Test with sigma equal to zero (should handle or raise an error)
        with self.assertRaises(ValueError):
            lognormalpdf(1, 0, 0)

    def test_non_numeric_inputs(self):
        # Test with non-numeric inputs
        with self.assertRaises(TypeError):
            lognormalpdf("string", "string", "string")

    def test_mixed_inputs(self):
        # Test with mixed type inputs
        self.assertAlmostEqual(
            lognormalpdf(1.0, Numeric(0), 1).get_value(), 0.3989422804
        )
        # Add more mixed type tests here

    def test_large_values(self):
        # Test with very large values
        self.assertAlmostEqual(
            lognormalpdf(1000, 0, 1).get_value(), 0
        )  # Replace with actual expected value

    def test_small_values(self):
        # Test with very small but positive values
        expected_value = 0
        self.assertAlmostEqual(
            lognormalpdf(0.0001, 0, 1).get_value(), expected_value
        )  # Replace with actual expected value


class TestUniformPdf(unittest.TestCase):
    def test_within_bounds(self):
        # Test with x within the bounds [a, b]
        self.assertAlmostEqual(uniformpdf(0, -1, 1).get_value(), 0.5)
        self.assertAlmostEqual(uniformpdf(0.5, -1, 1).get_value(), 0.5)

    def test_outside_bounds(self):
        # Test with x outside the bounds [a, b]
        self.assertAlmostEqual(uniformpdf(-2, -1, 1).get_value(), 0.0)
        self.assertAlmostEqual(uniformpdf(2, -1, 1).get_value(), 0.0)

    def test_expression_inputs(self):
        self.assertAlmostEqual(
            uniformpdf(Numeric(0), Numeric(-1), Numeric(1)).get_value(), 0.5
        )
        self.assertAlmostEqual(
            uniformpdf(Numeric(0.5), Numeric(-1), Numeric(1)).get_value(), 0.5
        )

    def test_invalid_bounds(self):
        # Test with invalid bounds (a >= b)
        with self.assertRaises(ValueError):
            uniformpdf(0, 1, -1)

    def test_non_numeric_inputs(self):
        # Test with non-numeric inputs
        with self.assertRaises(TypeError):
            uniformpdf("string", "string", "string")

    def test_mixed_inputs(self):
        # Test with mixed type inputs
        self.assertAlmostEqual(uniformpdf(0.5, Numeric(-1), 1).get_value(), 0.5)

    def test_edge_cases(self):
        # Test edge cases (x at the boundaries a and b)
        self.assertAlmostEqual(uniformpdf(-1, -1, 1).get_value(), 0.5)
        self.assertAlmostEqual(uniformpdf(1, -1, 1).get_value(), 0.5)


class TestTriangularPdf(unittest.TestCase):
    def test_valid_values(self):
        # Test with x within the bounds [a, b] and c within (a, b)
        self.assertAlmostEqual(triangularpdf(0, -1, 1, 0).get_value(), 1.0)
        self.assertAlmostEqual(triangularpdf(-0.5, -1, 1, 0).get_value(), 0.5)
        self.assertAlmostEqual(triangularpdf(0.5, -1, 1, 0).get_value(), 0.5)

    def test_outside_bounds(self):
        # Test with x outside the bounds [a, b]
        self.assertAlmostEqual(triangularpdf(-2, -1, 1, 0).get_value(), 0.0)
        self.assertAlmostEqual(triangularpdf(2, -1, 1, 0).get_value(), 0.0)

    def test_invalid_c_values(self):
        # Test with invalid c values (c <= a or c >= b)
        with self.assertRaises(ValueError):
            triangularpdf(0, -1, 1, -1)
        with self.assertRaises(ValueError):
            triangularpdf(0, -1, 1, 2)

    def test_expression_inputs(self):
        # Test with Expression type inputs
        self.assertAlmostEqual(
            triangularpdf(Numeric(0), Numeric(-1), Numeric(1), Numeric(0)).get_value(),
            1.0,
        )
        self.assertAlmostEqual(
            triangularpdf(
                Numeric(-0.5), Numeric(-1), Numeric(1), Numeric(0)
            ).get_value(),
            0.5,
        )
        self.assertAlmostEqual(
            triangularpdf(
                Numeric(0.5), Numeric(-1), Numeric(1), Numeric(0)
            ).get_value(),
            0.5,
        )

    def test_non_numeric_inputs(self):
        # Test with non-numeric inputs
        with self.assertRaises(TypeError):
            triangularpdf("string", "string", "string", "string")

    def test_mixed_inputs(self):
        # Test with mixed type inputs
        self.assertAlmostEqual(triangularpdf(0.5, Numeric(-1), 1, 0).get_value(), 0.5)
        # Add more mixed type tests here

    def test_edge_cases(self):
        # Test edge cases (x at the boundaries a, b, and c)
        self.assertAlmostEqual(triangularpdf(-1, -1, 1, 0).get_value(), 0.0)
        self.assertAlmostEqual(triangularpdf(1, -1, 1, 0).get_value(), 0.0)
        self.assertAlmostEqual(triangularpdf(0, -1, 1, 0).get_value(), 1.0)


class TestLogisticCdf(unittest.TestCase):
    def test_normal_values(self):
        # Test with normal values
        self.assertAlmostEqual(logisticcdf(0).get_value(), 0.5)
        self.assertAlmostEqual(logisticcdf(1, 0, 1).get_value(), 0.7310585786300049)

    def test_expression_inputs(self):
        # Test with Expression type inputs if applicable
        self.assertAlmostEqual(logisticcdf(Numeric(0)).get_value(), 0.5)
        self.assertAlmostEqual(
            logisticcdf(Numeric(1), Numeric(0), Numeric(1)).get_value(),
            0.7310585786300049,
        )

    def test_negative_scale(self):
        # Test with negative scale (should handle or raise an error)
        with self.assertRaises(ValueError):
            logisticcdf(0, 0, -1)

    def test_zero_scale(self):
        # Test with scale equal to zero (should handle or raise an error)
        with self.assertRaises(ValueError):
            logisticcdf(0, 0, 0)

    def test_non_numeric_inputs(self):
        # Test with non-numeric inputs
        with self.assertRaises(TypeError):
            logisticcdf("string", "string", "string")

    def test_mixed_inputs(self):
        # Test with mixed type inputs
        self.assertAlmostEqual(logisticcdf(0.0, Numeric(0), 1).get_value(), 0.5)
        # Add more mixed type tests here

    def test_large_values(self):
        # Test with very large values
        self.assertAlmostEqual(logisticcdf(1000, 0, 1).get_value(), 1.0)

    def test_small_values(self):
        # Test with very small values
        self.assertAlmostEqual(logisticcdf(-1000, 0, 1).get_value(), 0.0)


if __name__ == '__main__':
    unittest.main()
