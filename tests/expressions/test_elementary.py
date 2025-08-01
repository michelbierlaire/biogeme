"""Tests for elementary expressions

Michel Bierlaire
Tue Mar 25 17:43:42 2025
"""

import unittest
import warnings

import jax.numpy as jnp

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Draws,
    RandomVariable,
    Variable,
    bioDraws,
    list_of_random_variables_in_expression,
    list_of_variables_in_expression,
)


class TestElementaryExpressions(unittest.TestCase):

    def test_elementary_dict_type_match(self):
        e = Variable('x')
        result = list_of_variables_in_expression(the_expression=e)
        self.assertIn(e, result)

    def test_elementary_dict_type_mismatch(self):
        e = Variable("x")
        result = list_of_random_variables_in_expression(the_expression=e)
        self.assertEqual(result, [])

    def test_variable_jax_function(self):
        v = Variable("income")
        v.specific_id = 2
        jax_fn = v.recursive_construct_jax_function(numerically_safe=False)
        one_row = jnp.array([10.0, 20.0, 30.0])
        result = jax_fn(None, one_row, None, None)
        self.assertEqual(result, 30.0)

    def test_variable_missing_id(self):
        v = Variable("age")
        with self.assertRaises(BiogemeError):
            _ = v.safe_variable_id

    def test_biodraws_jax_function(self):
        d = Draws("eps", "NORMAL")
        d.specific_id = 1
        jax_fn = d.recursive_construct_jax_function(numerically_safe=False)
        draws = jnp.array([[0.0, 0.5, 1.0]])
        result = jax_fn(None, None, draws, None)
        self.assertAlmostEqual(result, 0.5)

    def test_biodraws_missing_id(self):
        d = Draws("eps", "NORMAL")
        with self.assertRaises(BiogemeError):
            _ = d.safe_draw_id

    def test_random_variable_id_access(self):
        rv = RandomVariable("z")
        rv.rv_id = 0
        self.assertEqual(rv.rv_id, 0)

    def test_random_variable_missing_id(self):
        rv = RandomVariable("z")
        with self.assertRaises(BiogemeError):
            _ = rv.safe_rv_id

    def test_biodraws_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # ensure all warnings are caught

            _ = bioDraws("eps", "NORMAL")

            self.assertTrue(
                any(issubclass(warning.category, DeprecationWarning) for warning in w)
            )
            self.assertTrue(
                any("deprecated" in str(warning.message).lower() for warning in w)
            )
