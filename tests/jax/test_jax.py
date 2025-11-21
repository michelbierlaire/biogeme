"""Unit tests for the Jax implementation of the expressions.

Michel Bierlaire
Sat Mar 29 10:56:13 2025

"""

import logging
import unittest

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

from biogeme.database import Database
from biogeme.draws import DrawsManagement
from biogeme.expressions import (
    Beta,
    BinaryMax,
    BinaryMin,
    Draws,
    IntegrateNormal,
    LogLogit,
    MonteCarlo,
    NormalCdf,
    RandomVariable,
    Variable,
    cos,
    exp,
    log,
    logzero,
    sin,
)
from biogeme.expressions.power_constant import PowerConstant
from biogeme.jax_calculator import calculate_single_formula
from biogeme.model_elements import FlatPanelAdapter, ModelElements, RegularAdapter
from biogeme.second_derivatives import SecondDerivativesMode

logging.basicConfig(level=logging.WARNING)


class TestBetaConstructJaxFunction(unittest.TestCase):
    def setUp(self):
        """Set up a Beta instance for testing"""
        self.beta_1 = Beta(
            name="beta_1", value=1.0, lowerbound=None, upperbound=None, status=0
        )
        self.beta_2 = Beta(
            name="beta_2", value=2.0, lowerbound=None, upperbound=None, status=0
        )

        data = pd.DataFrame(
            {"income": [1, 2, 3], "age": [10, 20, 30], "choice": [0, 1, 0]}
        )
        self.database = Database('test', dataframe=data)
        self.draws_manager = DrawsManagement(sample_size=3, number_of_draws=4)

    def test_beta(self):
        """Test if the generated JAX function correctly retrieves the parameter value."""

        parameters = {'beta_1': 0.5}  # Example parameters

        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )

        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=self.beta_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum(
            [
                0.5,
                0.5,
                0.5,
            ]
        )
        self.assertEqual(jax_1.function, expected_result)
        expected_gradient = [3]
        self.assertEqual(jax_1.gradient, expected_gradient)
        expected_hessian = [[0]]
        self.assertEqual(jax_1.hessian, expected_hessian)
        expected_bhhh = [[3]]
        self.assertEqual(jax_1.bhhh, expected_bhhh)

    def test_fixed_beta(self):

        beta_value = 0.5
        parameters = {'beta_1': beta_value}  # Example parameters
        value = 3
        fixed_beta = Beta(
            name="fixed_beta", value=value, lowerbound=None, upperbound=None, status=1
        )

        the_expression = fixed_beta + self.beta_1

        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=the_expression, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum(
            [
                beta_value + value,
                beta_value + value,
                beta_value + value,
            ]
        )
        self.assertEqual(jax_1.function, expected_result)
        expected_gradient = [3]
        self.assertEqual(jax_1.gradient, expected_gradient)
        expected_hessian = [[0]]
        self.assertEqual(jax_1.hessian, expected_hessian)
        expected_bhhh = [[3]]
        self.assertEqual(jax_1.bhhh, expected_bhhh)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=fixed_beta, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum(
            [
                value,
                value,
                value,
            ]
        )

        self.assertEqual(jax_2.function, expected_result)

    def test_variable(self):
        """Test if the generated JAX function correctly retrieves the parameter value."""

        parameters = {}
        expression_income = Variable('income')
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_income = ModelElements.from_expression_and_weight(
            log_like=expression_income,
            weight=None,
            adapter=adapter,
            use_jit=True,
        )
        jax_income = calculate_single_formula(
            model_elements=model_elements_income,
            the_betas=parameters,
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1, 2, 3])
        self.assertEqual(jax_income.function, expected_result)
        expected_gradient = None
        self.assertEqual(jax_income.gradient, expected_gradient)
        expected_hessian = None
        self.assertEqual(jax_income.hessian, expected_hessian)

        expression_age = Variable('age')

        model_elements_age = ModelElements.from_expression_and_weight(
            log_like=expression_age, weight=None, adapter=adapter, use_jit=True
        )

        jax_age = calculate_single_formula(
            model_elements=model_elements_age,
            the_betas=parameters,
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([10, 20, 30])
        self.assertEqual(jax_age.function, expected_result)

        expression_choice = Variable('choice')

        model_elements_choice = ModelElements.from_expression_and_weight(
            log_like=expression_choice,
            weight=None,
            adapter=adapter,
            use_jit=True,
        )

        jax_choice = calculate_single_formula(
            model_elements=model_elements_choice,
            the_betas=parameters,
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([0, 1, 0])
        self.assertEqual(jax_choice.function, expected_result)

    def test_plus(self):
        parameters = {'beta_1': 0.9, 'beta_2': 1.5}  # Example parameters
        expression_1 = self.beta_1 + self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([0.9 + 1.5, 0.9 + 1.5, 0.9 + 1.5])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([3, 3])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))
        expected_hessian = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))
        expected_bhhh = jnp.array([[3.0, 3.0], [3.0, 3.0]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_2': 1.5}
        expression_2 = self.beta_2 + Variable('age')
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1.5 + 10, 1.5 + 20, 1.5 + 30])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))

        expected_gradient = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))
        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))
        expected_bhhh = jnp.array([[3]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

        expression_3 = Variable('income') + self.beta_2 + Variable('age')
        model_elements_3 = ModelElements.from_expression_and_weight(
            log_like=expression_3, weight=None, adapter=adapter, use_jit=True
        )
        jax_3 = calculate_single_formula(
            model_elements=model_elements_3,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1 + 1.5 + 10, 2 + 1.5 + 20, 3 + 1.5 + 30])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))
        expected_gradient = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_3.gradient, expected_gradient))
        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_3.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_3.hessian, expected_hessian))
        expected_bhhh = jnp.array([[3]])
        self.assertEqual(jax_3.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_3.bhhh, expected_bhhh))

    def test_minus(self):
        parameters = {'beta_1': 0.9, 'beta_2': 1.5}  # Example parameters
        expression_1 = self.beta_1 - self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([0.9 - 1.5, 0.9 - 1.5, 0.9 - 1.5])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([3, -3])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))
        expected_bhhh = np.array([[3.0, -3.0], [-3.0, 3.0]])
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))
        expected_hessian = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))

        parameters = {'beta_2': 1.5}
        expression_2 = self.beta_2 - Variable('age')
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1.5 - 10, 1.5 - 20, 1.5 - 30])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))
        expected_bhhh = jnp.array([[3]])
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))
        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))

        expression_3 = Variable('income') - self.beta_2 - Variable('age')
        model_elements_3 = ModelElements.from_expression_and_weight(
            log_like=expression_3, weight=None, adapter=adapter, use_jit=True
        )
        jax_3 = calculate_single_formula(
            model_elements=model_elements_3,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1 - 1.5 - 10, 2 - 1.5 - 20, 3 - 1.5 - 30])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))
        expected_gradient = jnp.array([-3])
        self.assertTrue(jnp.allclose(jax_3.gradient, expected_gradient))
        expected_bhhh = jnp.array([[3]])
        self.assertTrue(jnp.allclose(jax_3.bhhh, expected_bhhh))
        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_3.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_3.hessian, expected_hessian))

    def test_times(self):
        parameters = {'beta_1': 0.9, 'beta_2': 1.5}  # Example parameters
        expression_1 = self.beta_1 * self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([0.9 * 1.5, 0.9 * 1.5, 0.9 * 1.5])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([1.5 * 3, 0.9 * 3])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))
        expected_hessian = jnp.array([[0, 3], [3, 0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))
        expected_bhhh = jnp.array([[6.75, 4.05], [4.05, 2.43]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_2': 1.5}
        expression_2 = self.beta_2 * Variable('age')
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )

        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([1.5 * 10, 1.5 * 20, 1.5 * 30])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient = jnp.array([10 + 20 + 30])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))
        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))
        expected_bhhh = jnp.array([[1400]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

        expression_3 = Variable('income') * self.beta_2 * Variable('age')
        model_elements_3 = ModelElements.from_expression_and_weight(
            log_like=expression_3, weight=None, adapter=adapter, use_jit=True
        )

        jax_3 = calculate_single_formula(
            model_elements=model_elements_3,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1 * 1.5 * 10, 2 * 1.5 * 20, 3 * 1.5 * 30])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))
        expected_gradient = jnp.array([1 * 10 + 2 * 20 + 3 * 30])
        self.assertTrue(jnp.allclose(jax_3.gradient, expected_gradient))
        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_3.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_3.hessian, expected_hessian))
        expected_bhhh = jnp.array([[9800]])
        self.assertEqual(jax_3.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_3.bhhh, expected_bhhh))

    def test_divide(self):
        parameters = {'beta_1': 0.9, 'beta_2': 1.5}  # Example parameters
        expression_1 = self.beta_1 / self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([0.9 / 1.5, 0.9 / 1.5, 0.9 / 1.5])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([3 / 1.5, -3 * 0.9 / (1.5 * 1.5)])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))
        expected_hessian = jnp.array(
            [[0, -3 / 1.5**2], [-3 / 1.5**2, 6 * 0.9 / 1.5**3]]
        )
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))
        expected_bhhh = jnp.array([[4 / 3, -0.8], [-0.8, 0.48]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_2': 1.5}
        expression_2 = self.beta_2 / Variable('age')
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1.5 / 10, 1.5 / 20, 1.5 / 30])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient = jnp.array([1.0 / 10 + 1.0 / 20 + 1.0 / 30])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))
        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))
        expected_bhhh = jnp.array([[0.01361111]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

        expression_3 = Variable('income') / self.beta_2 / Variable('age')
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_3 = ModelElements.from_expression_and_weight(
            log_like=expression_3, weight=None, adapter=adapter, use_jit=True
        )
        jax_3 = calculate_single_formula(
            model_elements=model_elements_3,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1 / 1.5 / 10, 2 / 1.5 / 20, 3 / 1.5 / 30])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))
        expected_gradient = jnp.array(
            [-1.0 / (10 * 1.5**2) - 2.0 / (20 * 1.5**2) - 3.0 / (30 * 1.5**2)]
        )
        self.assertTrue(jnp.allclose(jax_3.gradient, expected_gradient))
        expected_hessian = jnp.array(
            [[2 * 1 / (10 * 1.5**3) + 2 * 2 / (20 * 1.5**3) + 2 * 3 / (30 * 1.5**3)]]
        )
        self.assertEqual(jax_3.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_3.hessian, expected_hessian))
        expected_bhhh = jnp.array([[0.00592593]])
        self.assertEqual(jax_3.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_3.bhhh, expected_bhhh))

    def test_power(self):
        parameters = {'beta_1': 0.9, 'beta_2': 1.5}  # Example parameters
        expression_1 = self.beta_1**self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([0.9**1.5, 0.9**1.5, 0.9**1.5])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array(
            [
                3
                * parameters['beta_2']
                * parameters['beta_1'] ** (parameters['beta_2'] - 1),  # ∂f/∂β₁
                3
                * np.log(parameters['beta_1'])
                * parameters['beta_1'] ** parameters['beta_2'],  # ∂f/∂β₂
            ]
        )
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))
        expected_hessian = jnp.array(
            [
                [
                    3
                    * parameters['beta_2']
                    * (parameters['beta_2'] - 1)
                    * parameters['beta_1'] ** (parameters['beta_2'] - 2),
                    3
                    * parameters['beta_1'] ** (parameters['beta_2'] - 1)
                    * (1 + parameters['beta_2'] * np.log(parameters['beta_1'])),
                ],
                [
                    3
                    * parameters['beta_1'] ** (parameters['beta_2'] - 1)
                    * (1 + parameters['beta_2'] * np.log(parameters['beta_1'])),
                    3
                    * (np.log(parameters['beta_1'])) ** 2
                    * parameters['beta_1'] ** parameters['beta_2'],
                ],
            ]
        )
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))
        expected_bhhh = jnp.array([[6.075, -0.38403916], [-0.38403916, 0.02427755]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_2': 1.5}
        expression_2 = self.beta_2 ** Variable('age')
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([1.5**10, 1.5**20, 1.5**30])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient = jnp.array(
            [
                10 * parameters['beta_2'] ** 9
                + 20 * parameters['beta_2'] ** 19
                + 30 * parameters['beta_2'] ** 29
            ]
        )
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))

        expected_hessian = jnp.array(
            [
                [
                    10 * 9 * parameters['beta_2'] ** 8
                    + 20 * 19 * parameters['beta_2'] ** 18
                    + 30 * 29 * parameters['beta_2'] ** 28
                ]
            ]
        )
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))
        expected_bhhh = jnp.array([[1.4709354e13]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

    def test_min(self):
        parameters = {'beta_1': 0.9, 'beta_2': 1.5}  # Example parameters
        expression_1 = BinaryMin(self.beta_1, self.beta_2)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([0.9, 0.9, 0.9])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))

        expected_gradient = jnp.array([3, 0])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))

        expected_hessian = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))

        expected_bhhh = jnp.array([[3, 0], [0, 0]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_2': 1.5}
        expression_2 = BinaryMin(self.beta_2, Variable('age'))
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1.5, 1.5, 1.5])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))

        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))

        expected_bhhh = jnp.array([[3]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

    def test_max(self):
        parameters = {'beta_1': 0.9, 'beta_2': 1.5}  # Example parameters
        expression_1 = BinaryMax(self.beta_1, self.beta_2)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([1.5, 1.5, 1.5])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([0, 3])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))

        expected_hessian = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))

        expected_bhhh = jnp.array([[0, 0], [0, 3]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_2': 1.5}
        expression_2 = BinaryMax(self.beta_2, Variable('age'))
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([10, 20, 30])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))

        expected_gradient = jnp.array([0])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))

        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))

        expected_bhhh = jnp.array([[0]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

    def test_and(self):
        parameters = {'beta_1': 0, 'beta_2': 1.5}  # Example parameters
        expression_1 = self.beta_1 & self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([0, 0, 0])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([0, 0])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))

        expected_hessian = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))

        expected_bhhh = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_1': 1.5}  # Example parameters
        expression_2 = self.beta_1 * (Variable('age') & Variable('income'))
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1.5, 1.5, 1.5])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))

        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))
        expected_bhhh = jnp.array([[3]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

        parameters = {'beta_1': 1, 'beta_2': 1.5}  # Example parameters
        expression_3 = self.beta_1 & self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_3 = ModelElements.from_expression_and_weight(
            log_like=expression_3, weight=None, adapter=adapter, use_jit=True
        )
        jax_3 = calculate_single_formula(
            model_elements=model_elements_3,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1, 1, 1])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))
        expected_gradient = jnp.array([0, 0])
        self.assertTrue(jnp.allclose(jax_3.gradient, expected_gradient))

        expected_hessian = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_3.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_3.hessian, expected_hessian))

        expected_bhhh = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_3.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_3.bhhh, expected_bhhh))

    def test_or(self):
        parameters = {'beta_1': 0, 'beta_2': 1.5}  # Example parameters
        expression_1 = self.beta_1 | self.beta_2
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1, 1, 1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([0, 0])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))

        expected_hessian = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))

        expected_bhhh = jnp.array([[0, 0], [0, 0]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        parameters = {'beta_1': 1.5}  # Example parameters
        expression_2 = self.beta_1 * (Variable('age') | Variable('income'))
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1.5, 1.5, 1.5])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient))

        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian))

        expected_bhhh = jnp.array([[3]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh))

    def test_unary_minus(self):
        parameters = {'beta_1': -1}  # Example parameters
        expression_1 = -self.beta_1
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1, 1, 1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient = jnp.array([-3])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient))

        expected_hessian = jnp.array([[0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian))

        expected_bhhh = jnp.array([[3]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh))

        expression_2 = -Variable('age')
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )

        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas={},
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([-10, -20, -30])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))

    def test_normal_cdf(self):
        value_1 = 0
        cdf_1 = norm.cdf(value_1)
        pdf_1 = norm.pdf(value_1)
        value_2 = -1
        cdf_2 = norm.cdf(value_2)
        pdf_2 = norm.pdf(value_2)
        parameters = {'beta_1': value_1}
        expression_1 = NormalCdf(self.beta_1)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([cdf_1, cdf_1, cdf_1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient_1 = jnp.array([3 * pdf_1])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient_1))
        expected_hessian_1 = jnp.array([[-3 * value_1 * pdf_1]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian_1.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian_1))

        expected_bhhh_1 = jnp.array([[0.47746485]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh_1.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh_1))

        parameters = {'beta_2': value_2}
        expression_2 = NormalCdf(self.beta_2)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([cdf_2, cdf_2, cdf_2])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient_2 = jnp.array([3 * pdf_2])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient_2))
        expected_hessian_2 = jnp.array([[-3 * value_2 * pdf_2]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian_2.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian_2))

        expected_bhhh_2 = jnp.array([[0.17564951]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh_2.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh_2))

    def test_exp(self):
        value_1 = 0
        exp_1 = np.exp(value_1)
        value_2 = -1
        exp_2 = np.exp(value_2)
        parameters = {'beta_1': value_1}  # Example parameters
        expression_1 = exp(self.beta_1)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([exp_1, exp_1, exp_1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient_1 = jnp.array([3 * exp_1])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient_1))
        expected_hessian_1 = jnp.array([[3]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian_1.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian_1))
        expected_bhhh_1 = jnp.array([[3]])
        self.assertEqual(jax_1.bhhh.shape, expected_bhhh_1.shape)
        self.assertTrue(jnp.allclose(jax_1.bhhh, expected_bhhh_1))

        parameters = {'beta_2': value_2}  # Example parameters

        expression_2 = exp(self.beta_2)
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([exp_2, exp_2, exp_2])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))

    def test_sin(self):
        value_1 = 0
        sin_1 = np.sin(value_1)
        value_2 = -1
        sin_2 = np.sin(value_2)
        parameters = {'beta_1': value_1}  # Example parameters

        # Test sin(beta_1)
        expression_1 = sin(self.beta_1)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([sin_1, sin_1, sin_1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))

        expected_gradient_1 = jnp.array([3 * np.cos(value_1)])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient_1))

        expected_hessian_1 = jnp.array([[-3 * np.sin(value_1)]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian_1.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian_1))

        # Test sin(beta_2)
        parameters = {'beta_2': value_2}
        expression_2 = sin(self.beta_2)
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )

        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([sin_2, sin_2, sin_2])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))

        expected_gradient_2 = jnp.array([3 * np.cos(value_2)])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient_2))

        expected_hessian_2 = jnp.array([[-3 * np.sin(value_2)]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian_2.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian_2))

    def test_cos(self):
        value_1 = 0
        cos_1 = np.cos(value_1)
        value_2 = -1
        cos_2 = np.cos(value_2)
        parameters = {'beta_1': value_1}  # Example parameters
        expression_1 = cos(self.beta_1)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([cos_1, cos_1, cos_1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        parameters = {'beta_2': value_2}
        expression_2 = cos(self.beta_2)
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )

        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([cos_2, cos_2, cos_2])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))

    def test_log(self):
        value_1 = 1
        log_1 = np.log(value_1)
        value_2 = 2
        log_2 = np.log(value_2)
        parameters = {'beta_1': value_1}  # Example parameters
        expression_1 = log(self.beta_1)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([log_1, log_1, log_1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient_1 = jnp.array([3 * (1 / value_1)])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient_1))
        expected_hessian_1 = jnp.array([[-3 * (1 / value_1**2)]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian_1.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian_1))

        parameters = {'beta_2': value_2}
        expression_2 = log(self.beta_2)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([log_2, log_2, log_2])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient_2 = jnp.array([3 * (1 / value_2)])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient_2))
        expected_hessian_2 = jnp.array([[-3 * (1 / value_2**2)]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian_2.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian_2))

    def test_log_zero(self):
        value_1 = 0
        log_1 = 0
        value_2 = 2
        log_2 = np.log(value_2)
        parameters = {'beta_1': value_1}  # Example parameters
        expression_1 = logzero(self.beta_1)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([log_1, log_1, log_1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        expected_gradient_1 = jnp.array([0.0])
        self.assertTrue(jnp.allclose(jax_1.gradient, expected_gradient_1))
        expected_hessian_1 = jnp.array([[0.0]])
        self.assertEqual(jax_1.hessian.shape, expected_hessian_1.shape)
        self.assertTrue(jnp.allclose(jax_1.hessian, expected_hessian_1))

        parameters = {'beta_2': value_2}
        expression_2 = logzero(self.beta_2)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([log_2, log_2, log_2])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient_2 = jnp.array([3 * (1 / value_2)])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient_2))
        expected_hessian_2 = jnp.array([[-3 * (1 / value_2**2)]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian_2.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian_2))

    def test_power_constant(self):
        parameters = {'beta_1': -2}  # Example parameters

        expression_1 = PowerConstant(self.beta_1, 2)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([4, 4, 4])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))

        expression_2 = PowerConstant(self.beta_1, 3)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2, weight=None, adapter=adapter, use_jit=True
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([-8, -8, -8])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        parameters = {'beta_2': 0}  # Example parameters

        expression_3 = PowerConstant(self.beta_2, -3.1)
        model_elements_3 = ModelElements.from_expression_and_weight(
            log_like=expression_3, weight=None, adapter=adapter, use_jit=True
        )
        jax_3 = calculate_single_formula(
            model_elements=model_elements_3,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=True,
        )
        expected_result = sum([0, 0, 0])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))

    def test_integrate(self):
        beta = Beta('beta', 1, -10, 10, 0)
        omega = RandomVariable('omega')
        integral = IntegrateNormal(omega - omega + beta, name='omega')
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=integral, weight=None, adapter=adapter, use_jit=True
        )
        jax_1 = calculate_single_formula(
            model_elements=model_elements,
            the_betas={'beta': 1.0},
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1, 1, 1])
        self.assertTrue(jnp.allclose(jax_1.function, expected_result))
        jax_2 = calculate_single_formula(
            model_elements=model_elements,
            the_betas={'beta': 1.0},
            gradient=True,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([1, 1, 1])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))
        expected_gradient_2 = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient_2))
        jax_3 = calculate_single_formula(
            model_elements=model_elements,
            the_betas={'beta': 1.0},
            gradient=True,
            hessian=True,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([1, 1, 1])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))
        expected_gradient_3 = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_3.gradient, expected_gradient_3))
        expected_hessian_3 = jnp.array([0])
        self.assertTrue(jnp.allclose(jax_3.hessian, expected_hessian_3))

        jax_4 = calculate_single_formula(
            model_elements=model_elements,
            the_betas={'beta': 1.0},
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([1, 1, 1])
        self.assertTrue(jnp.allclose(jax_4.function, expected_result))
        expected_gradient_4 = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_4.gradient, expected_gradient_4))
        expected_hessian_4 = jnp.array([0])
        self.assertTrue(jnp.allclose(jax_4.hessian, expected_hessian_4))
        expected_bhhh_4 = jnp.array([3])
        self.assertTrue(jnp.allclose(jax_4.bhhh, expected_bhhh_4))

    def test_integrate_2(self):

        value_beta_1 = 1
        value_beta_2 = 1
        beta_1 = Beta('beta_1', value_beta_1, -10, 10, 0)
        beta_2 = Beta('beta_2', value_beta_2, -10, 10, 0)
        beta_parameters = {'beta_1': value_beta_1, 'beta_2': value_beta_2}

        # Integral using Monte Carlo
        draws = Draws('draws', 'NORMAL_HALTON3')
        formula_mc = beta_1 * beta_2 * exp(draws)
        integral_mc = MonteCarlo(formula_mc)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_mc = ModelElements.from_expression_and_weight(
            log_like=integral_mc,
            weight=None,
            adapter=adapter,
            number_of_draws=1000000,
            use_jit=True,
        )
        jax_mc = calculate_single_formula(
            model_elements=model_elements_mc,
            the_betas=beta_parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        # Integral using scipy
        def f(x):
            x = np.clip(x, a_min=None, a_max=100)  # Prevent exp(>100)
            return value_beta_1 * value_beta_2 * np.exp(x)

        def integrand(x):
            return f(x) * norm.pdf(x)

        result, error = quad(integrand, -np.inf, np.inf)

        omega = RandomVariable('omega')
        formula_rv = beta_1 * beta_2 * exp(omega)
        integral_rv = IntegrateNormal(formula_rv, name='omega')
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=integral_rv, weight=None, adapter=adapter, use_jit=True
        )

        jax_rv = calculate_single_formula(
            model_elements=model_elements,
            the_betas=beta_parameters,
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([result, result, result])
        self.assertTrue(jnp.allclose(jax_rv.function, expected_result))

        jax_2 = calculate_single_formula(
            model_elements=model_elements,
            the_betas={'beta': 1.0},
            gradient=True,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        expected_result = sum([result, result, result])
        self.assertTrue(jnp.allclose(jax_2.function, expected_result))

        self.assertTrue(jnp.allclose(jax_2.gradient, jax_mc.gradient, rtol=1e-3))

        jax_3 = calculate_single_formula(
            model_elements=model_elements,
            the_betas={'beta': 1.0},
            gradient=True,
            hessian=True,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([result, result, result])
        self.assertTrue(jnp.allclose(jax_3.function, expected_result))

        self.assertTrue(jnp.allclose(jax_3.gradient, jax_mc.gradient, rtol=1e-3))
        self.assertTrue(jnp.allclose(jax_3.hessian, jax_mc.hessian, rtol=1e-3))

        jax_4 = calculate_single_formula(
            model_elements=model_elements,
            the_betas={'beta': 1.0},
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_result = sum([result, result, result])
        self.assertTrue(jnp.allclose(jax_4.function, expected_result))

        self.assertTrue(jnp.allclose(jax_4.gradient, jax_mc.gradient, rtol=1e-3))
        self.assertTrue(jnp.allclose(jax_4.hessian, jax_mc.hessian, rtol=1e-3))
        self.assertTrue(jnp.allclose(jax_4.bhhh, jax_mc.bhhh, rtol=1e-3))

    def test_monte_carlo(self):
        random_term = Draws('random_term', 'UNIFORM')
        second_term = Draws('second_term', 'NORMAL')
        expression_1 = MonteCarlo(random_term + second_term)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements_1 = ModelElements.from_expression_and_weight(
            log_like=expression_1,
            weight=None,
            adapter=adapter,
            number_of_draws=10,
            use_jit=True,
        )

        expected_results = []

        jax_1 = calculate_single_formula(
            model_elements=model_elements_1,
            the_betas={},
            gradient=False,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        for obs in model_elements_1.draws_management.draws:
            integral = 0
            for draw in obs:
                integral += draw[0] + draw[1]
            expected_results.append(integral / len(obs))
        self.assertTrue(jnp.allclose(jax_1.function, sum(expected_results)))

        parameters = {'beta_1': 0.9}  # Example parameters
        expression_2 = MonteCarlo(self.beta_1 * random_term)
        expected_results = []
        expected_gradient = []
        expected_bhhh = []
        model_elements_2 = ModelElements.from_expression_and_weight(
            log_like=expression_2,
            weight=None,
            adapter=adapter,
            number_of_draws=10,
            use_jit=True,
        )
        jax_2 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        for obs in model_elements_2.draws_management.draws:
            integral = 0
            for draw in obs:
                integral += draw[0]
            expected_results.append(0.9 * integral / len(obs))
            expected_gradient.append(integral / len(obs))
            expected_bhhh.append(integral * integral / (len(obs) * len(obs)))
        self.assertTrue(jnp.allclose(jax_2.function, sum(expected_results)))
        expected_gradient_2 = jnp.array([sum(expected_gradient)])
        self.assertTrue(jnp.allclose(jax_2.gradient, expected_gradient_2))
        expected_hessian_2 = jnp.array([[0]])
        self.assertEqual(jax_2.hessian.shape, expected_hessian_2.shape)
        self.assertTrue(jnp.allclose(jax_2.hessian, expected_hessian_2))
        expected_bhhh_2 = jnp.array([[sum(expected_bhhh)]])
        self.assertEqual(jax_2.bhhh.shape, expected_bhhh_2.shape)
        self.assertTrue(jnp.allclose(jax_2.bhhh, expected_bhhh_2))

        # Test: gradient=True, hessian=False, bhhh=False
        jax_3 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=False,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        self.assertTrue(jnp.allclose(jax_3.function, sum(expected_results)))
        self.assertTrue(jnp.allclose(jax_3.gradient, expected_gradient_2))
        self.assertIsNone(jax_3.hessian)
        self.assertIsNone(jax_3.bhhh)

        # Test: gradient=True, hessian=True, bhhh=False
        jax_4 = calculate_single_formula(
            model_elements=model_elements_2,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=False,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        self.assertTrue(jnp.allclose(jax_4.function, sum(expected_results)))
        self.assertTrue(jnp.allclose(jax_4.gradient, expected_gradient_2))
        self.assertTrue(jnp.allclose(jax_4.hessian, expected_hessian_2))
        self.assertIsNone(jax_4.bhhh)

    def test_logit_1(self):
        value_1 = 0
        value_2 = 0
        parameters = {'beta_1': value_1, 'beta_2': value_2}
        beta_1 = Beta('beta_1', 0, None, None, 0)
        beta_2 = Beta('beta_2', 0, None, None, 0)
        utilities = {12: beta_1, 23: beta_2}
        the_logit = LogLogit(utilities, None, 12)
        expected_f = 3 * np.log(0.5)
        expected_g = np.array([1.5, -1.5])
        expected_h = np.array([[-0.75, 0.75], [0.75, -0.75]])
        expected_bhhh = 3 * np.outer([0.5, -0.5], [0.5, -0.5])
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=the_logit, weight=None, adapter=adapter, use_jit=True
        )
        jax = calculate_single_formula(
            model_elements=model_elements,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        self.assertTrue(jnp.allclose(jax.function, expected_f))
        self.assertTrue(jnp.allclose(jax.gradient, expected_g))
        self.assertEqual(jax.hessian.shape, expected_h.shape)
        self.assertTrue(jnp.allclose(jax.hessian, expected_h))
        self.assertEqual(jax.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax.bhhh, expected_bhhh))

    def test_logit_2(self):
        value_1 = 0.1
        value_2 = 0.2
        parameters = {'beta_1': value_1, 'beta_2': value_2}
        beta_1 = Beta('beta_1', 0, None, None, 0)
        beta_2 = Beta('beta_2', 0, None, None, 0)
        utilities = {12: beta_1, 23: beta_2}
        the_logit = LogLogit(utilities, None, 12)
        expected_function = -2.2331900596618652
        expected_gradient = np.array([1.5749376, -1.5749376])
        expected_hessian = np.array([[-0.7481281, 0.7481281], [0.7481281, -0.7481281]])
        expected_bhhh = np.array([[0.82680964, -0.8268095], [-0.8268095, 0.8268094]])
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=the_logit, weight=None, adapter=adapter, use_jit=True
        )
        jax = calculate_single_formula(
            model_elements=model_elements,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )

        self.assertTrue(jnp.allclose(jax.function, expected_function))
        self.assertTrue(jnp.allclose(jax.gradient, expected_gradient))
        self.assertEqual(jax.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax.hessian, expected_hessian))
        self.assertEqual(jax.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax.bhhh, expected_bhhh))

    def test_logit_3(self):
        value_1 = 0.1
        value_2 = 0.2
        parameters = {'beta_1': value_1, 'beta_2': value_2}
        beta_1 = Beta('beta_1', 0, None, None, 0)
        beta_2 = Beta('beta_2', 0, None, None, 0)
        utilities = {12: beta_1, 23: beta_2}
        av = {12: 1, 23: 0}
        the_logit = LogLogit(utilities, av, 12)
        adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )
        model_elements = ModelElements.from_expression_and_weight(
            log_like=the_logit, weight=None, adapter=adapter, use_jit=True
        )
        jax = calculate_single_formula(
            model_elements=model_elements,
            the_betas=parameters,
            gradient=True,
            hessian=True,
            bhhh=True,
            second_derivatives_mode=SecondDerivativesMode.ANALYTICAL,
            numerically_safe=False,
        )
        expected_function = 0.0
        expected_gradient = np.array([0.0, 0.0])
        expected_hessian = np.array([[0.0, 0.0], [0.0, 0.0]])
        expected_bhhh = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.assertTrue(jnp.allclose(jax.function, expected_function))
        self.assertTrue(jnp.allclose(jax.gradient, expected_gradient))
        self.assertEqual(jax.hessian.shape, expected_hessian.shape)
        self.assertTrue(jnp.allclose(jax.hessian, expected_hessian))
        self.assertEqual(jax.bhhh.shape, expected_bhhh.shape)
        self.assertTrue(jnp.allclose(jax.bhhh, expected_bhhh))


if __name__ == '__main__':
    unittest.main()
