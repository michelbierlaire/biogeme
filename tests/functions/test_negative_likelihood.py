"""Unit tests for negative loglikelihood function

Michel Bierlaire
Sat Mar 29 16:31:28 2025
"""

import unittest

import numpy as np
from biogeme_optimization.function import FunctionData

from biogeme.exceptions import BiogemeError
from biogeme.function_output import FunctionOutput
from biogeme.likelihood.negative_likelihood import NegativeLikelihood


class TestNegativeLikelihood(unittest.TestCase):
    def setUp(self):
        self.dimension = 2
        self.mock_output = FunctionOutput(
            function=10.0,
            gradient=np.array([1.0, 2.0]),
            hessian=np.array([[1.0, 0.0], [0.0, 3.0]]),
            bhhh=None,
        )

        def mock_loglikelihood(x, gradient, hessian, bhhh):
            return self.mock_output

        self.neg_likelihood = NegativeLikelihood(self.dimension, mock_loglikelihood)
        self.neg_likelihood.x = np.array([1.0, 2.0])

    def test_dimension(self):
        self.assertEqual(self.neg_likelihood.dimension(), self.dimension)

    def test_f_returns_negative_loglikelihood(self):
        result = self.neg_likelihood._f()
        self.assertAlmostEqual(result, -10.0)

    def test_f_g_returns_negative_function_and_gradient(self):
        result: FunctionData = self.neg_likelihood._f_g()
        self.assertAlmostEqual(result.function, -10.0)
        np.testing.assert_array_almost_equal(
            result.gradient, -self.mock_output.gradient
        )
        self.assertIsNone(result.hessian)

    def test_f_g_h_returns_negative_function_gradient_and_hessian(self):
        result: FunctionData = self.neg_likelihood._f_g_h()
        self.assertAlmostEqual(result.function, -10.0)
        np.testing.assert_array_almost_equal(
            result.gradient, -self.mock_output.gradient
        )
        np.testing.assert_array_almost_equal(result.hessian, -self.mock_output.hessian)

    def test_f_raises_error_if_x_not_set(self):
        self.neg_likelihood.x = None
        with self.assertRaises(BiogemeError):
            self.neg_likelihood._f()

    def test_f_g_raises_error_if_x_not_set(self):
        self.neg_likelihood.x = None
        with self.assertRaises(BiogemeError):
            self.neg_likelihood._f_g()

    def test_f_g_h_raises_error_if_x_not_set(self):
        self.neg_likelihood.x = None
        with self.assertRaises(BiogemeError):
            self.neg_likelihood._f_g_h()


if __name__ == '__main__':
    unittest.main()
