"""Unit tests for the function call version of the formula

Michel Bierlaire
Sat Mar 29 15:31:30 2025
"""

import unittest

import numpy as np
import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.jax_calculator import (
    CallableExpression,
    CompiledFormulaEvaluator,
    function_from_compiled_formula,
)
from biogeme.model_elements import FlatPanelAdapter, ModelElements, RegularAdapter
from biogeme.second_derivatives import SecondDerivativesMode


class TestFunctionFromExpression(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame.from_dict({'x': [1.0, 2.0, 3.0]})
        self.database = Database('test', self.data)
        self.x = Variable('x')
        self.beta = Beta('beta', 1.0, None, None, 0)
        self.expression = self.beta * self.x
        self.initial_betas = {'beta': 1.0}
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        compiled_function = CompiledFormulaEvaluator(
            model_elements=model_elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        self.func: CallableExpression = function_from_compiled_formula(
            the_compiled_function=compiled_function, the_betas=self.initial_betas
        )

    def test_function_output_only(self):
        result = self.func(np.array([2.0]), gradient=False, hessian=False, bhhh=False)
        expected_value = 2.0 * np.sum(self.data['x'])
        self.assertAlmostEqual(result.function, expected_value, places=6)
        self.assertIsNone(result.gradient)
        self.assertIsNone(result.hessian)
        self.assertIsNone(result.bhhh)

    def test_function_with_gradient(self):
        result = self.func(np.array([3.0]), gradient=True, hessian=False, bhhh=False)
        expected_gradient = np.sum(self.data['x'])
        self.assertAlmostEqual(result.gradient[0], expected_gradient, places=6)

    def test_function_with_hessian(self):
        result = self.func(np.array([4.0]), gradient=True, hessian=True, bhhh=False)
        self.assertTrue(np.allclose(result.hessian, np.zeros((1, 1))))

    def test_function_with_bhhh(self):
        result = self.func(np.array([5.0]), gradient=True, hessian=False, bhhh=True)
        self.assertEqual(result.bhhh.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
