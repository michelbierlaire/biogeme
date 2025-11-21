# tests/test_elem_pymc.py
import unittest

import numpy as np
import pandas as pd
import pymc as pm

# If your project exposes expressions from a package root, use this:
from biogeme.expressions import Beta, Elem, Numeric, Variable
from pytensor import function


# Otherwise, fall back to explicit submodules:
# from biogeme.expressions.elem import Elem
# from biogeme.expressions.numeric_expressions import Numeric
# from biogeme.expressions.variable import Variable
# from biogeme.expressions.beta_parameters import Beta


class TestElemPyMCComputation(unittest.TestCase):
    def setUp(self):
        # Tiny DataFrame with 3 observations
        # Key k selects the branch per observation: 0, 1, 0
        self.df = pd.DataFrame(
            {
                "k": [0, 1, 0],
                "x": [1.0, 2.0, 3.0],
            }
        )

    def _eval(self, expr):
        """Build PyMC graph for expr and evaluate on self.df."""
        builder = expr.recursive_construct_pymc_model_builder()
        with pm.Model():  # <-- ensure a model context
            out = builder(self.df)  # expected shape: (N,)
            self.assertTrue(
                hasattr(out, "ndim") and out.ndim == 1, "Output must be 1-D (N,)."
            )
            f = function([], out)
            return f()

    def test_elem_selects_and_computes_numeric(self):
        """
        Elem should select the corresponding branch per observation AND compute
        the expression value. Here branch 0 -> 10 * x, branch 1 -> 20 * x.
        With k = [0, 1, 0] and x = [1, 2, 3], result = [10, 40, 30].
        """
        key = Variable("k")  # per-observation keys
        x = Variable("x")  # per-observation regressor
        branch0 = Numeric(10.0) * x  # 10 * x
        branch1 = Numeric(20.0) * x  # 20 * x

        expr = Elem({0: branch0, 1: branch1}, key)
        res = self._eval(expr)
        np.testing.assert_allclose(
            res, np.array([10.0, 40.0, 30.0]), rtol=0, atol=1e-12
        )

    def test_elem_with_beta_parameters_and_affine_forms(self):
        """
        Branch 0: (b0 * x + 1), with b0 fixed to 1.5
        Branch 1: (b1 * x - 2), with b1 fixed to -0.5
        k = [0,1,0], x = [1,2,3]
        Expected:
          n=0 -> 1.5*1 + 1 = 2.5
          n=1 -> -0.5*2 - 2 = -3.0
          n=2 -> 1.5*3 + 1 = 5.5
        """
        key = Variable("k")
        x = Variable("x")

        # Beta(name, start, lower, upper, status) with status=1 -> fixed parameter
        b0 = Beta("b0", 1.5, None, None, 1)  # fixed at 1.5
        b1 = Beta("b1", -0.5, None, None, 1)  # fixed at -0.5

        branch0 = b0 * x + Numeric(1.0)
        branch1 = b1 * x - Numeric(2.0)

        expr = Elem({0: branch0, 1: branch1}, key)
        res = self._eval(expr)
        np.testing.assert_allclose(res, np.array([2.5, -3.0, 5.5]), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
