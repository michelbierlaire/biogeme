"""Tests for unary expressions

Michel Bierlaire
Wed Mar 26 08:35:13 2025
"""

import unittest
import warnings

import jax.numpy as jnp
import numpy as np

from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    BelongsTo,
    Derive,
    Expression,
    IntegrateNormal,
    MonteCarlo,
    NormalCdf,
    PanelLikelihoodTrajectory,
    bioNormalCdf,
    cos,
    exp,
    log,
    logzero,
    sin,
)
from biogeme.expressions.power_constant import PowerConstant
from biogeme.expressions.unary_minus import UnaryMinus


class DummyNumeric(Expression):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def get_value(self) -> float:
        return self.value

    def recursive_construct_jax_function(self, numerically_safe: bool):
        def f(
            parameters: jnp.ndarray,
            one_row: jnp.ndarray,
            draws: jnp.ndarray,
            random_variables: jnp.ndarray,
        ) -> float:
            return self.value

        return f

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)


class TestUnaryExpressions(unittest.TestCase):
    def setUp(self):
        self.constant = DummyNumeric(2.0)
        self.zero = DummyNumeric(0.0)
        self.negative = DummyNumeric(-4.0)

    def test_unary_minus(self):
        expr = UnaryMinus(self.constant)
        self.assertEqual(expr.get_value(), -2.0)
        self.assertEqual(str(expr), f'(-{self.constant})')
        self.assertIn('UnaryMinus', repr(expr))

    def test_exp(self):
        expr = exp(self.constant)
        self.assertAlmostEqual(expr.get_value(), np.exp(2.0))
        self.assertIn('exp', repr(expr))

    def test_log(self):
        expr = log(self.constant)
        self.assertAlmostEqual(expr.get_value(), np.log(2.0))
        self.assertIn('log', repr(expr))

    def test_logzero_nonzero(self):
        expr = logzero(self.constant)
        self.assertAlmostEqual(expr.get_value(), np.log(2.0))

    def test_logzero_zero(self):
        expr = logzero(self.zero)
        self.assertEqual(expr.get_value(), 0.0)

    def test_sin(self):
        expr = sin(self.constant)
        self.assertAlmostEqual(expr.get_value(), np.sin(2.0))

    def test_cos(self):
        expr = cos(self.constant)
        self.assertAlmostEqual(expr.get_value(), np.cos(2.0))

    def test_power_constant_positive(self):
        expr = PowerConstant(self.constant, 2)
        self.assertEqual(expr.get_value(), 4.0)

    def test_power_constant_negative_integer(self):
        expr = PowerConstant(self.negative, 2)
        self.assertEqual(expr.get_value(), 16.0)

    def test_power_constant_negative_nonint(self):
        expr = PowerConstant(self.negative, 2.5)
        with self.assertRaises(BiogemeError):
            expr.get_value()

    def test_panel_likelihood_repr(self):
        expr = PanelLikelihoodTrajectory(self.constant)
        self.assertIn('PanelLikelihoodTrajectory', repr(expr))

    def test_normal_cdf_repr(self):
        expr = NormalCdf(self.constant)
        print(repr(expr))
        self.assertIn('NormalCdf', repr(expr))

    def test_monte_carlo_jax(self):
        expr = MonteCarlo(self.constant)
        jax_fn = expr.recursive_construct_jax_function(numerically_safe=False)
        draws = jnp.array([[1.0], [2.0], [3.0]])
        self.assertAlmostEqual(jax_fn(None, None, draws, None), 2.0)

    def test_draws_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = bioNormalCdf(self.constant)
            self.assertTrue(
                any(issubclass(wi.category, DeprecationWarning) for wi in w)
            )

    def test_derive_repr(self):
        expr = Derive(self.constant, "beta")
        self.assertIn("Derive", repr(expr))

    def test_integrate_repr(self):
        expr = IntegrateNormal(self.constant, "z")
        self.assertIn("Integrate", repr(expr))

    def test_belongsto_repr(self):
        expr = BelongsTo(self.constant, {1.0, 2.0})
        self.assertIn("BelongsTo", repr(expr))


if __name__ == '__main__':
    unittest.main()
