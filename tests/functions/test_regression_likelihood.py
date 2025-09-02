# tests/test_linear_regression.py
import unittest
from unittest.mock import patch

from biogeme.expressions import (
    Beta,
    Expression,
    LinearTermTuple,
    LinearUtility,
    Variable,
)
from biogeme.likelihood.linear_regression import (
    build_normalized_formula,
    regression_likelihood,
)

# Import the module under test


class TestLinearRegressionHelpers(unittest.TestCase):
    def setUp(self):
        # Simple, symbolic set-up (no data evaluation required)
        self.y = Variable('y')
        self.x1 = Variable('x1')
        self.x2 = Variable('x2')

        self.b1 = Beta('b1', 0.0, None, None, 0)
        self.b2 = Beta('b2', 0.0, None, None, 0)

        # Use a separate symbol for the scale parameter
        self.sigma = Variable('sigma')

        self.indep = [self.x1, self.x2]
        self.coefs = [self.b1, self.b2]
        self.linear_terms = LinearUtility(
            [LinearTermTuple(beta=b, x=x) for b, x in zip(self.coefs, self.indep)]
        )

        # Expected symbolic dot product and standardized residual (built independently)
        self.expected_dot = LinearUtility(
            [
                LinearTermTuple(beta=self.b1, x=self.x1),
                LinearTermTuple(beta=self.b2, x=self.x2),
            ]
        )
        self.expected_std_resid = (self.y - self.expected_dot) / self.sigma

    # -------- build_formula --------

    def test_build_formula_symbolic_structure(self):
        expr = build_normalized_formula(
            dependent_variable=self.y,
            linear_terms=self.linear_terms,
            scale_parameter=self.sigma,
        )
        # Compare the expression structure via its repr (stable enough for biogeme Expressions)
        self.assertIsInstance(expr, Expression)
        self.assertEqual(repr(expr), repr(self.expected_std_resid))

    # -------- regression_likelihood --------
    @patch('biogeme.likelihood.linear_regression.normalpdf')
    def test_regression_likelihood_calls_normalpdf_with_expected_argument(
        self, mock_normalpdf
    ):
        # Make the stub return a recognizable token
        mock_normalpdf.side_effect = lambda arg: ('normalpdf_called_with', repr(arg))

        result = regression_likelihood(
            dependent_variable=self.y,
            linear_terms=self.linear_terms,
            scale_parameter=self.sigma,
        )

        # Ensure normalpdf was called once
        self.assertEqual(mock_normalpdf.call_count, 1)
        called_arg = mock_normalpdf.call_args[0][0]

        # The argument should be the standardized residual that build_formula would construct
        self.assertEqual(repr(called_arg), repr(self.expected_std_resid))

        # And the return value should be whatever normalpdf returned
        self.assertEqual(
            result, ('normalpdf_called_with', repr(self.expected_std_resid))
        )

    # -------- additional sanity tests --------
    def test_symbolic_contains_scale_in_denominator(self):
        expr = build_normalized_formula(
            dependent_variable=self.y,
            linear_terms=self.linear_terms,
            scale_parameter=self.sigma,
        )
        # Weak but useful check that the scale parameter symbol appears in the denominator
        self.assertIn('sigma', repr(expr))

    def test_symbolic_linear_combination_present(self):
        expr = build_normalized_formula(
            dependent_variable=self.y,
            linear_terms=self.linear_terms,
            scale_parameter=self.sigma,
        )
        # Ensure the names of variables and betas appear in the representation
        txt = repr(expr)
        for token in ('y', 'x1', 'x2', 'b1', 'b2'):
            self.assertIn(token, txt)


if __name__ == '__main__':
    unittest.main()
