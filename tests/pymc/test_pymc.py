import logging
import unittest

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from biogeme.database import Database
from biogeme.draws import DrawsManagement
from biogeme.expressions import (
    Beta,
    BinaryMax,
    BinaryMin,
    LogLogit,
    NormalCdf,
    Variable,
    cos,
    exp,
    log,
    logzero,
    sin,
)
from biogeme.expressions.power_constant import PowerConstant
from biogeme.model_elements import FlatPanelAdapter, ModelElements, RegularAdapter
from scipy.stats import norm

logging.basicConfig(level=logging.WARNING)


class TestPymcConstructExpressions(unittest.TestCase):
    def setUp(self):
        """Set up common data and Betas for PyMC tests."""
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

        self.adapter = (
            FlatPanelAdapter(database=self.database)
            if self.database.is_panel()
            else RegularAdapter(database=self.database)
        )

    # ---- helper ---------------------------------------------------------

    def _eval_expression(self, expr) -> np.ndarray:
        """Evaluate an Expression via its PyMC / PyTensor builder on `self.database`."""
        # Let ModelElements prepare the expression tree (for MonteCarlo, etc.).
        model_elements = ModelElements.from_expression_and_weight(
            log_like=expr,
            weight=None,
            adapter=self.adapter,
            use_jit=False,
        )
        root_expr = model_elements.loglikelihood

        with pm.Model():
            builder = root_expr.recursive_construct_pymc_model_builder()
            tensor = builder(self.database.dataframe)  # shape (N,)
            f = pytensor.function([], tensor)()
        return np.asarray(f)

    # ---- simple Betas / Variables --------------------------------------

    def test_beta_pymc(self):
        """Free Beta becomes a scalar RV; value is random but shared across observations."""
        beta = Beta('beta_1', 0.5, None, None, 0)
        values = np.asarray(self._eval_expression(beta))

        # If we get a scalar draw from the prior, broadcast it to all observations
        if values.ndim == 0:
            values = np.full(len(self.database.dataframe), float(values))

        # We cannot test the numerical value (it's a prior draw), but it must be
        # the same across all rows and finite.
        self.assertEqual(values.shape, (len(self.database.dataframe),))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.allclose(values, values[0]))

    def test_fixed_beta_pymc(self):
        """Fixed Beta evaluates to a deterministic constant."""
        fixed_beta = Beta(
            'fixed_beta', value=3.0, lowerbound=None, upperbound=None, status=1
        )

        values = np.asarray(self._eval_expression(fixed_beta))
        n = len(self.database.dataframe)

        if values.ndim == 0:
            values = np.full(n, float(values))

        self.assertEqual(values.shape, (n,))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(
            np.allclose(values, 3.0),
            msg=f"Fixed beta should evaluate to 3.0, got {values}",
        )

    def test_variable_pymc(self):
        income_expr = Variable('income')
        age_expr = Variable('age')
        choice_expr = Variable('choice')

        income_vals = self._eval_expression(income_expr)
        self.assertTrue(np.allclose(income_vals, np.array([1, 2, 3])))

        age_vals = self._eval_expression(age_expr)
        self.assertTrue(np.allclose(age_vals, np.array([10, 20, 30])))

        choice_vals = self._eval_expression(choice_expr)
        self.assertTrue(np.allclose(choice_vals, np.array([0, 1, 0])))

    # ---- arithmetic: +, -, *, /, power ---------------------------------

    def test_plus_pymc(self):
        expr1 = Beta('beta_1', 0.9, None, None, 1) + Beta('beta_2', 1.5, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, 0.9 + 1.5)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = Beta('beta_2', 1.5, None, None, 1) + Variable('age')
        vals2 = self._eval_expression(expr2)
        expected2 = np.array([1.5 + 10, 1.5 + 20, 1.5 + 30])
        self.assertTrue(np.allclose(vals2, expected2))

        expr3 = (
            Variable('income') + Beta('beta_2', 1.5, None, None, 1) + Variable('age')
        )
        vals3 = self._eval_expression(expr3)
        expected3 = np.array([1 + 1.5 + 10, 2 + 1.5 + 20, 3 + 1.5 + 30])
        self.assertTrue(np.allclose(vals3, expected3))

    def test_minus_pymc(self):
        expr1 = Beta('beta_1', 0.9, None, None, 1) - Beta('beta_2', 1.5, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, 0.9 - 1.5)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = Beta('beta_2', 1.5, None, None, 1) - Variable('age')
        vals2 = self._eval_expression(expr2)
        expected2 = np.array([1.5 - 10, 1.5 - 20, 1.5 - 30])
        self.assertTrue(np.allclose(vals2, expected2))

        expr3 = (
            Variable('income') - Beta('beta_2', 1.5, None, None, 1) - Variable('age')
        )
        vals3 = self._eval_expression(expr3)
        expected3 = np.array([1 - 1.5 - 10, 2 - 1.5 - 20, 3 - 1.5 - 30])
        self.assertTrue(np.allclose(vals3, expected3))

    def test_times_pymc(self):
        expr1 = Beta('beta_1', 0.9, None, None, 1) * Beta('beta_2', 1.5, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, 0.9 * 1.5)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = Beta('beta_2', 1.5, None, None, 1) * Variable('age')
        vals2 = self._eval_expression(expr2)
        expected2 = np.array([1.5 * 10, 1.5 * 20, 1.5 * 30])
        self.assertTrue(np.allclose(vals2, expected2))

        expr3 = (
            Variable('income') * Beta('beta_2', 1.5, None, None, 1) * Variable('age')
        )
        vals3 = self._eval_expression(expr3)
        expected3 = np.array([1 * 1.5 * 10, 2 * 1.5 * 20, 3 * 1.5 * 30])
        self.assertTrue(np.allclose(vals3, expected3))

    def test_divide_pymc(self):
        expr1 = Beta('beta_1', 0.9, None, None, 1) / Beta('beta_2', 1.5, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, 0.9 / 1.5)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = Beta('beta_2', 1.5, None, None, 1) / Variable('age')
        vals2 = self._eval_expression(expr2)
        expected2 = np.array([1.5 / 10, 1.5 / 20, 1.5 / 30])
        self.assertTrue(np.allclose(vals2, expected2))

        expr3 = (
            Variable('income') / Beta('beta_2', 1.5, None, None, 1) / Variable('age')
        )
        vals3 = self._eval_expression(expr3)
        expected3 = np.array([1 / 1.5 / 10, 2 / 1.5 / 20, 3 / 1.5 / 30])
        self.assertTrue(np.allclose(vals3, expected3))

    def test_power_pymc(self):
        # beta_1 ** beta_2
        expr1 = Beta('beta_1', 0.9, None, None, 1) ** Beta('beta_2', 1.5, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, 0.9**1.5)
        self.assertTrue(np.allclose(vals1, expected1))

        # beta_2 ** age
        expr2 = Beta('beta_2', 1.5, None, None, 1) ** Variable('age')
        vals2 = self._eval_expression(expr2)
        expected2 = np.array([1.5**10, 1.5**20, 1.5**30])
        self.assertTrue(np.allclose(vals2, expected2))

    # ---- min / max / logical ops ---------------------------------------

    def test_min_pymc(self):
        expr1 = BinaryMin(
            Beta('beta_1', 0.9, None, None, 1), Beta('beta_2', 1.5, None, None, 1)
        )
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, min(0.9, 1.5))
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = BinaryMin(Beta('beta_2', 1.5, None, None, 1), Variable('age'))
        vals2 = self._eval_expression(expr2)
        expected2 = np.array([1.5, 1.5, 1.5])  # 1.5 < 10,20,30
        self.assertTrue(np.allclose(vals2, expected2))

    def test_max_pymc(self):
        expr1 = BinaryMax(
            Beta('beta_1', 0.9, None, None, 1), Beta('beta_2', 1.5, None, None, 1)
        )
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, max(0.9, 1.5))
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = BinaryMax(Beta('beta_2', 1.5, None, None, 1), Variable('age'))
        vals2 = self._eval_expression(expr2)
        expected2 = np.array([10, 20, 30])  # age > 1.5
        self.assertTrue(np.allclose(vals2, expected2))

    def test_and_pymc(self):
        expr1 = Beta('beta_1', 0.0, None, None, 1) & Beta('beta_2', 1.5, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.zeros(3)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = Beta('beta_1', 1.5, None, None, 1) * (
            Variable('age') & Variable('income')
        )
        vals2 = self._eval_expression(expr2)
        # age>0 and income>0 for all rows → AND is 1 everywhere
        expected2 = np.full(3, 1.5)
        self.assertTrue(np.allclose(vals2, expected2))

    def test_or_pymc(self):
        expr1 = Beta('beta_1', 0.0, None, None, 1) | Beta('beta_2', 1.5, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.ones(3)  # 0 OR 1.5 → True → 1
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = Beta('beta_1', 1.5, None, None, 1) * (
            Variable('age') | Variable('income')
        )
        vals2 = self._eval_expression(expr2)
        # age>0 or income>0 for all rows → OR is 1 everywhere
        expected2 = np.full(3, 1.5)
        self.assertTrue(np.allclose(vals2, expected2))

    def test_unary_minus_pymc(self):
        expr1 = -Beta('beta_1', -1.0, None, None, 1)
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, 1.0)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = -Variable('age')
        vals2 = self._eval_expression(expr2)
        expected2 = -np.array([10, 20, 30])
        self.assertTrue(np.allclose(vals2, expected2))

    # ---- elementwise functions: NormalCdf, exp, sin, cos, log, logzero --

    def test_normal_cdf_pymc(self):
        value_1 = 0.0
        value_2 = -1.0
        cdf_1 = norm.cdf(value_1)
        cdf_2 = norm.cdf(value_2)

        expr1 = NormalCdf(Beta('beta_1', value_1, None, None, 1))
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, cdf_1)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = NormalCdf(Beta('beta_2', value_2, None, None, 1))
        vals2 = self._eval_expression(expr2)
        expected2 = np.full(3, cdf_2)
        self.assertTrue(np.allclose(vals2, expected2))

    def test_exp_pymc(self):
        v1, v2 = 0.0, -1.0
        e1, e2 = np.exp(v1), np.exp(v2)

        expr1 = exp(Beta('beta_1', v1, None, None, 1))
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, e1)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = exp(Beta('beta_2', v2, None, None, 1))
        vals2 = self._eval_expression(expr2)
        expected2 = np.full(3, e2)
        self.assertTrue(np.allclose(vals2, expected2))

    def test_sin_pymc(self):
        v1, v2 = 0.0, -1.0
        s1, s2 = np.sin(v1), np.sin(v2)

        expr1 = sin(Beta('beta_1', v1, None, None, 1))
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, s1)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = sin(Beta('beta_2', v2, None, None, 1))
        vals2 = self._eval_expression(expr2)
        expected2 = np.full(3, s2)
        self.assertTrue(np.allclose(vals2, expected2))

    def test_cos_pymc(self):
        v1, v2 = 0.0, -1.0
        c1, c2 = np.cos(v1), np.cos(v2)

        expr1 = cos(Beta('beta_1', v1, None, None, 1))
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, c1)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = cos(Beta('beta_2', v2, None, None, 1))
        vals2 = self._eval_expression(expr2)
        expected2 = np.full(3, c2)
        self.assertTrue(np.allclose(vals2, expected2))

    def test_log_pymc(self):
        v1, v2 = 1.0, 2.0
        l1, l2 = np.log(v1), np.log(v2)

        expr1 = log(Beta('beta_1', v1, None, None, 1))
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, l1)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = log(Beta('beta_2', v2, None, None, 1))
        vals2 = self._eval_expression(expr2)
        expected2 = np.full(3, l2)
        self.assertTrue(np.allclose(vals2, expected2))

    def test_logzero_pymc(self):
        v1, v2 = 0.0, 2.0
        l1, l2 = 0.0, np.log(v2)

        expr1 = logzero(Beta('beta_1', v1, None, None, 1))
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, l1)
        self.assertTrue(np.allclose(vals1, expected1))

        expr2 = logzero(Beta('beta_2', v2, None, None, 1))
        vals2 = self._eval_expression(expr2)
        expected2 = np.full(3, l2)
        self.assertTrue(np.allclose(vals2, expected2))

    # ---- PowerConstant --------------------------------------------------

    def test_power_constant_pymc(self):
        # square
        expr1 = PowerConstant(Beta('beta_1', -2.0, None, None, 1), 2)
        vals1 = self._eval_expression(expr1)
        expected1 = np.full(3, 4.0)
        self.assertTrue(np.allclose(vals1, expected1))

        # cube
        expr2 = PowerConstant(Beta('beta_1', -2.0, None, None, 1), 3)
        vals2 = self._eval_expression(expr2)
        expected2 = np.full(3, -8.0)
        self.assertTrue(np.allclose(vals2, expected2))

        # numerically safe branch around 0 for negative exponent
        expr3 = PowerConstant(Beta('beta_2', 0.0, None, None, 1), -3.1)
        vals3 = self._eval_expression(expr3)
        expected3 = np.zeros(3)
        self.assertTrue(np.allclose(vals3, expected3))

    # ---- LogLogit -------------------------------------------------------

    def test_loglogit_pymc(self):
        # Match test_logit_1 / _2 / _3 patterns for the *value* only
        value_1 = 0.1
        value_2 = 0.2
        beta_1 = Beta('beta_1', value_1, None, None, 1)
        beta_2 = Beta('beta_2', value_2, None, None, 1)
        # Ensure utilities are 1-D tensors (one value per observation) by
        # adding a dummy 0 * Variable('age') term. This keeps the numerical
        # value constant across observations while satisfying the LogLogit
        # PyMC builder shape requirements.
        u1 = beta_1 + 0 * Variable('age')
        u2 = beta_2 + 0 * Variable('age')
        utilities = {12: u1, 23: u2}

        # All alts available
        expr = LogLogit(utilities, None, 12)
        vals = self._eval_expression(expr)

        # For each observation log P(choice=12)
        # P(12) = exp(v1) / (exp(v1) + exp(v2)) with v1=0.1, v2=0.2
        v1, v2 = value_1, value_2
        p12 = np.exp(v1) / (np.exp(v1) + np.exp(v2))
        expected = np.full(3, np.log(p12))
        self.assertTrue(np.allclose(vals, expected))

        # Case where only 12 is available → probability 1 → log 1 = 0
        av = {12: 1, 23: 0}
        expr_av = LogLogit(utilities, av, 12)
        vals_av = self._eval_expression(expr_av)
        expected_av = np.zeros(3)
        self.assertTrue(np.allclose(vals_av, expected_av))
