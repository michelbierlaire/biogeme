"""Tests for parameter expression

Michel Bierlaire
Tue Mar 25 18:33:51 2025
"""

import unittest

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from biogeme.expressions import Beta, TypeOfElementaryExpression


class TestBetaParameter(unittest.TestCase):
    def test_constructor_valid(self):
        beta = Beta('beta1', 1.0, 0.0, 10.0, 0)
        self.assertEqual(beta.name, 'beta1')
        self.assertEqual(beta.init_value, 1.0)
        self.assertEqual(beta.lower_bound, 0.0)
        self.assertEqual(beta.upper_bound, 10.0)
        self.assertEqual(beta.status, 0)
        self.assertIsNone(beta.specific_id)

    def test_constructor_invalid_name(self):
        with self.assertRaises(BiogemeError):
            Beta(123, 1.0, None, None, 0)

    def test_constructor_invalid_value(self):
        with self.assertRaises(BiogemeError):
            Beta('beta1', 'invalid', None, None, 0)

    def test_expression_type(self):
        beta_free = Beta('beta1', 1.0, None, None, 0)
        self.assertEqual(
            beta_free.expression_type, TypeOfElementaryExpression.FREE_BETA
        )
        beta_fixed = Beta('beta2', 1.0, None, None, 1)
        self.assertEqual(
            beta_fixed.expression_type, TypeOfElementaryExpression.FIXED_BETA
        )

    def test_str_and_repr(self):
        beta = Beta('beta1', 1.0, None, None, 0)
        self.assertIn("Beta('beta1', 1.0", str(beta))
        self.assertIn('<Beta name=beta1 value=1.0', repr(beta))

    def test_safe_beta_id_raises(self):
        beta = Beta('beta1', 1.0, None, None, 0)
        with self.assertRaises(BiogemeError):
            _ = beta.safe_beta_id

    def test_safe_beta_id_returns(self):
        beta = Beta('beta1', 1.0, None, None, 0)
        beta.specific_id = 5
        self.assertEqual(beta.safe_beta_id, 5)

    def test_fix_betas_changes_value_and_status(self):
        beta = Beta('beta1', 1.0, None, None, 0)
        beta.fix_betas({'beta1': 2.5})
        self.assertEqual(beta.init_value, 2.5)
        self.assertEqual(beta.status, 1)

    def test_fix_betas_renames(self):
        beta = Beta('beta1', 1.0, None, None, 0)
        beta.fix_betas({'beta1': 2.5}, prefix='pre_', suffix='_suf')
        self.assertEqual(beta.name, 'pre_beta1_suf')

    def test_change_init_values_free(self):
        beta = Beta('beta1', 1.0, None, None, 0)
        beta.change_init_values({'beta1': 3.0})
        self.assertEqual(beta.init_value, 3.0)

    def test_change_init_values_fixed_logs_warning(self):
        beta = Beta('beta1', 1.0, None, None, 1)
        beta.change_init_values({'beta1': 2.0})
        self.assertEqual(beta.init_value, 2.0)

    def test_jax_function_returns_value(self):
        beta = Beta('beta1', 1.0, None, None, 0)
        beta.specific_id = 2
        fn = beta.recursive_construct_jax_function(numerically_safe=False)
        parameters = jnp.array([0.0, 1.0, 3.14])
        result = fn(parameters, None, None, None)
        self.assertEqual(result, 3.14)
