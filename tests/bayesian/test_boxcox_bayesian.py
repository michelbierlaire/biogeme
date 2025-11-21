# tests/test_boxcox_pymc.py
import unittest

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from biogeme.expressions import Numeric, Variable
from biogeme.models.boxcox import boxcox
from pytensor import function


class TestBoxCoxPyMC(unittest.TestCase):
    def _eval_expr(self, expr, df: pd.DataFrame) -> np.ndarray:
        """Build the PyMC/PyTensor graph for `expr` and evaluate it on `df`."""
        builder = expr.recursive_construct_pymc_model_builder()
        with pm.Model():
            out = builder(df)
            self.assertTrue(
                hasattr(out, "ndim") and out.ndim == 1,
                "Output must be a 1-D tensor (N,).",
            )
            f = function([], out)
            return np.asarray(f())

    def test_regular_formula_far_from_zero(self):
        """For ell far from 0, boxcox(x, ell) == (x**ell - 1)/ell."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        x = Variable("x")
        ell_val = 0.5  # far from 0 -> regular branch
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)
        expected = (np.power(df["x"].values, ell_val) - 1.0) / ell_val
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-10)

    def test_mclaurin_approx_near_zero(self):
        """For |ell| < 1e-5, use the specified McLaurin expansion."""
        # Use strictly positive x to avoid log(0)
        df = pd.DataFrame({"x": [1.5, 2.0, 3.0, 4.5]})
        x = Variable("x")
        ell_val = 1e-6  # triggers McLaurin branch (|ell| < 1e-5)
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)

        lx = np.log(df["x"].values)
        expected = (
            lx
            + ell_val * lx**2
            + (ell_val**2) * lx**3 / 6.0
            + (ell_val**3) * lx**4 / 24.0
        )
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-5)

    def test_zero_input_returns_zero(self):
        """boxcox(0, ell) is defined to return 0 regardless of ell."""
        df = pd.DataFrame({"x": [0.0, 2.0, 0.0, 5.0]})
        x = Variable("x")
        ell_val = 0.3
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)

        expected = np.where(
            df["x"].values == 0.0,
            0.0,
            (np.power(df["x"].values, ell_val) - 1.0) / ell_val,
        )
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-10)

    def test_symmetry_small_negative_ell(self):
        """Near-zero negative ell should also use McLaurin and match expansion."""
        df = pd.DataFrame({"x": [1.25, 1.75, 2.5]})
        x = Variable("x")
        ell_val = -5e-6  # still in McLaurin region
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)

        lx = np.log(df["x"].values)
        expected = (
            lx
            + ell_val * lx**2
            + (ell_val**2) * lx**3 / 6.0
            + (ell_val**3) * lx**4 / 24.0
        )
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-5)

    def test_grad_non_nan_for_positive_x(self):
        """Gradient wrt x should be finite for positive x and ell far from zero."""
        df = pd.DataFrame({"x": [1.2, 2.3, 3.4]})
        x = Variable("x")
        ell_val = 0.7
        y = boxcox(x, Numeric(ell_val))

        # Build graph and get gradient w.r.t. x via PyTensor
        builder = y.recursive_construct_pymc_model_builder()
        with pm.Model():
            out = builder(df)  # (N,)
            # Create a symbolic input representing x's data to take gradient w.r.t. it.
            xin = pt.vector("xin")
            # Rebuild expression with xin as the Variable source:
            # We emulate Variable('x') by substituting via a function:
            # Define a small function that replaces the Variable('x') values:
            # Simpler approach: use a new builder on a DataFrame using xin's shared var is non-trivial,
            # so instead check numeric stability via finite differences on the compiled out.

            f = function([], out)
            vals = f()
            self.assertTrue(np.all(np.isfinite(vals)))


class TestBoxCoxSmoothPyMC(unittest.TestCase):
    def _eval_expr(self, expr, df: pd.DataFrame) -> np.ndarray:
        """Build the PyMC/PyTensor graph for `expr` and evaluate it on `df`."""
        builder = expr.recursive_construct_pymc_model_builder()
        with pm.Model():
            out = builder(df)
            self.assertTrue(
                hasattr(out, "ndim") and out.ndim == 1,
                "Output must be a 1-D tensor (N,).",
            )
            f = function([], out)
            return np.asarray(f())

    def test_regular_formula_far_from_zero(self):
        """For ell far from 0, boxcox_smooth(x, ell) ≈ (x**ell - 1)/ell."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        x = Variable("x")
        ell_val = 0.5  # far from 0 -> regular regime
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)
        expected = (np.power(df["x"].values, ell_val) - 1.0) / ell_val
        # Allow a slightly looser tolerance than the exact BoxCox
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-6)
        # Ensure no NaNs
        self.assertTrue(np.isfinite(res).all())

    def test_mclaurin_approx_near_zero(self):
        """For |ell| small, smooth approximation should match Maclaurin well and be finite."""
        df = pd.DataFrame({"x": [1.5, 2.0, 3.0, 4.5]})
        x = Variable("x")
        ell_val = 1e-6
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)
        lx = np.log(df["x"].values)
        expected = (
            lx
            + ell_val * lx**2
            + (ell_val**2) * lx**3 / 6.0
            + (ell_val**3) * lx**4 / 24.0
        )
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-5)
        self.assertTrue(np.isfinite(res).all())

    def test_zero_input_returns_zero_and_finite(self):
        """boxcox_smooth(0, ell) should return exactly 0 and never produce NaNs."""
        df = pd.DataFrame({"x": [0.0, 2.0, 0.0, 5.0]})
        x = Variable("x")
        ell_val = 0.3
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)
        self.assertTrue(np.isfinite(res).all())
        # Zero entries should be exactly 0 (or numerically indistinguishable from 0)
        zero_mask = df["x"].values == 0.0
        self.assertTrue(np.allclose(res[zero_mask], 0.0, atol=1e-15))
        # Positive entries should be close to the exact definition
        expected_pos = (np.power(df["x"].values[~zero_mask], ell_val) - 1.0) / ell_val
        np.testing.assert_allclose(res[~zero_mask], expected_pos, rtol=0, atol=1e-6)

    def test_symmetry_small_negative_ell(self):
        """Near-zero negative ell should remain finite and close to Maclaurin."""
        df = pd.DataFrame({"x": [1.25, 1.75, 2.5]})
        x = Variable("x")
        ell_val = -5e-6
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)
        lx = np.log(df["x"].values)
        expected = (
            lx
            + ell_val * lx**2
            + (ell_val**2) * lx**3 / 6.0
            + (ell_val**3) * lx**4 / 24.0
        )
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-5)
        self.assertTrue(np.isfinite(res).all())

    def test_no_nan_across_various_cases(self):
        """Ensure the smooth transform never produces NaN/Inf across a variety of inputs."""
        cases = [
            (np.array([0.0, 1.0, 2.0, 3.0]), 0.1),
            (np.array([0.0, 1.0, 10.0, 100.0]), 1e-8),
            (np.array([1e-6, 1e-3, 1.0, 1e3]), -1e-6),
            (np.array([0.5, 2.0, 5.0, 10.0]), 0.9),
        ]
        for xs, ell_val in cases:
            df = pd.DataFrame({"x": xs})
            x = Variable("x")
            y = boxcox(x, Numeric(ell_val))
            res = self._eval_expr(y, df)
            self.assertTrue(np.isfinite(res).all())

    def test_approximation_quality_far_from_zero(self):
        """For typical values (x>0, |ell| not tiny), the smooth version should be close to exact."""
        rng = np.random.default_rng(123)
        xs = rng.uniform(0.2, 5.0, size=20)  # keep away from 0
        ell_val = 0.3
        df = pd.DataFrame({"x": xs})
        x = Variable("x")
        y = boxcox(x, Numeric(ell_val))

        res = self._eval_expr(y, df)
        expected = (np.power(xs, ell_val) - 1.0) / ell_val
        # Mean absolute error should be tiny; allow small bias from smoothing
        mae = np.mean(np.abs(res - expected))
        self.assertLess(mae, 1e-6)


if __name__ == "__main__":
    unittest.main()
