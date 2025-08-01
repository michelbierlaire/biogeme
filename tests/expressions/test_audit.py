"""Tests for audit functions

Michel Bierlaire
Fri Mar 28 17:35:27 2025
"""

import unittest

from biogeme.expressions import (
    BelongsTo,
    Beta,
    Draws,
    IntegrateNormal,
    LogLogit,
    MonteCarlo,
    RandomVariable,
    Variable,
)
from biogeme.expressions.audit import audit_expression


class TestAuditFunctions(unittest.TestCase):
    def test_montecarlo_with_draws(self):
        expr = MonteCarlo(Draws('eps', 'NORMAL'))
        errors, warnings = audit_expression(expr)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_montecarlo_without_draws(self):
        expr = MonteCarlo(Beta('beta_1', 1.0, None, None, 0))
        errors, warnings = audit_expression(expr)
        self.assertTrue(any('does not contain any Draws' in e for e in errors))

    def test_montecarlo_nested(self):
        expr = MonteCarlo(MonteCarlo(Draws('eps', 'NORMAL')))
        errors, warnings = audit_expression(expr)
        self.assertTrue(any('cannot contain another MonteCarlo' in e for e in errors))

    def test_integrate_with_random_variable(self):
        expr = IntegrateNormal(RandomVariable('x'), 'x')
        errors, warnings = audit_expression(expr)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_integrate_without_random_variable(self):
        expr = IntegrateNormal(Beta('beta_1', 1.0, None, None, 0), 'x')
        errors, warnings = audit_expression(expr)
        self.assertFalse(errors)
        self.assertTrue(
            any('does not contain any RandomVariable' in w for w in warnings)
        )

    def test_belongsto_with_non_integer(self):
        expr = BelongsTo(Variable('x'), {1.0, 2.5, 3})
        errors, warnings = audit_expression(expr)
        self.assertTrue(any('not integer' in w for w in warnings))

    def test_loglogit_with_mismatched_keys(self):
        util = {1: Beta('b1', 1.0, None, None, 0)}
        av = {2: Beta('a1', 1.0, None, None, 0)}
        expr = LogLogit(util, av, 1)
        errors, warnings = audit_expression(expr)
        self.assertTrue(any('Incompatible list of alternatives' in e for e in errors))

    def test_loglogit_with_matching_keys(self):
        util = {1: Beta('b1', 1.0, None, None, 0)}
        av = {1: Beta('a1', 1.0, None, None, 0)}
        expr = LogLogit(util, av, 1)
        errors, warnings = audit_expression(expr)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])


if __name__ == '__main__':
    unittest.main()
