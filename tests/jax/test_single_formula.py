"""Unit tests for the formula implementation

Michel Bierlaire
Sat Mar 29 10:56:48 2025
"""

import unittest

import numpy as np
import pandas as pd

from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.jax_calculator import CompiledFormulaEvaluator, calculate_single_formula
from biogeme.jax_calculator.single_formula import evaluate_model_per_row
from biogeme.model_elements import FlatPanelAdapter, ModelElements, RegularAdapter
from biogeme.second_derivatives import SecondDerivativesMode


class TestCompiledFormulaEvaluator(unittest.TestCase):
    def setUp(self):
        # Simple synthetic dataset
        self.data = pd.DataFrame.from_dict({'x': [1.0, 2.0, 3.0]})
        self.database = Database('test', self.data)
        self.x = Variable('x')
        self.beta = Beta('beta', 1.0, None, None, 0)
        self.expression = self.beta * self.x
        self.betas = {'beta': 2.0}
        self.draws = None

    def test_function_only(self):
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        evaluator = CompiledFormulaEvaluator(
            model_elements=model_elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        output = evaluator.evaluate(
            self.betas, gradient=False, hessian=False, bhhh=False
        )
        expected_value = float(np.sum(np.array(self.data['x']) * self.betas['beta']))
        self.assertAlmostEqual(output.function, expected_value, places=6)
        self.assertIsNone(output.gradient)
        self.assertIsNone(output.hessian)
        self.assertIsNone(output.bhhh)

    def test_function_and_gradient(self):
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        evaluator = CompiledFormulaEvaluator(
            model_elements=model_elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        output = evaluator.evaluate(
            self.betas, gradient=True, hessian=False, bhhh=False
        )
        self.assertIsNotNone(output.gradient)
        expected_gradient = np.sum(self.data['x'])
        self.assertAlmostEqual(output.gradient[0], expected_gradient, places=6)

    def test_function_gradient_hessian(self):
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        evaluator = CompiledFormulaEvaluator(
            model_elements=model_elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        output = evaluator.evaluate(self.betas, gradient=True, hessian=True, bhhh=False)
        self.assertIsNotNone(output.hessian)
        expected_hessian = np.zeros((1, 1))
        self.assertTrue(np.allclose(output.hessian, expected_hessian))

    def test_function_bhhh(self):
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        evaluator = CompiledFormulaEvaluator(
            model_elements=model_elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        output = evaluator.evaluate(self.betas, gradient=True, hessian=False, bhhh=True)
        self.assertIsNotNone(output.bhhh)
        self.assertEqual(output.bhhh.shape, (1, 1))

    def test_missing_beta_uses_default(self):
        betas_missing = {}
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        evaluator = CompiledFormulaEvaluator(
            model_elements=model_elements,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        output = evaluator.evaluate(
            betas_missing, gradient=False, hessian=False, bhhh=False
        )
        expected_value = float(
            np.sum(np.array(self.data['x']) * 1.0)
        )  # default beta = 1.0
        self.assertAlmostEqual(output.function, expected_value, places=6)

    def test_legacy_calculate_single_formula(self):
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        output = calculate_single_formula(
            model_elements=model_elements,
            the_betas=self.betas,
            gradient=True,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        self.assertIsNotNone(output.function)
        self.assertIsNotNone(output.gradient)


class TestEvaluateExpressionPerRow(unittest.TestCase):
    def setUp(self):
        # Sample data: x = [1.0, 2.0, 3.0]
        self.data = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        self.database = Database('test', self.data)
        self.draws = None

        # Expression: beta * x
        self.x = Variable('x')
        self.beta = Beta('beta', 1.0, None, None, 0)
        self.expression = self.beta * self.x

    def test_evaluate_expression_per_row_with_specified_beta(self):
        betas = {'beta': 2.0}
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )

        results = evaluate_model_per_row(
            model_elements=model_elements,
            the_betas=betas,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(results, expected)

    def test_evaluate_expression_per_row_with_default_beta(self):
        # No beta passed, should use default (1.0)
        betas = {}
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=self.expression, weight=None, adapter=adapter, use_jit=True
        )
        results = evaluate_model_per_row(
            model_elements=model_elements,
            the_betas=betas,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(results, expected)


if __name__ == '__main__':
    unittest.main()
