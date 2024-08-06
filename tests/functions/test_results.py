"""
Test the results module

:author: Michel Bierlaire
:date: Wed Aug 25 09:41:24 2021
"""

# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest
import pandas as pd
import biogeme.biogeme as bio
import biogeme.database as db
from biogeme.results import bioResults, calc_p_value, Beta as BetaResult
from biogeme.expressions import Beta, Variable, exp


class TestResults(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(
            {
                'Person': [1, 1, 1, 2, 2],
                'Exclude': [0, 0, 1, 0, 1],
                'Variable1': [1, 2, 3, 4, 5],
                'Variable2': [10, 20, 30, 40, 50],
                'Choice': [1, 2, 3, 1, 2],
                'Av1': [0, 1, 1, 1, 1],
                'Av2': [1, 1, 1, 1, 1],
                'Av3': [0, 1, 1, 1, 1],
            }
        )
        myData = db.Database('test', df)

        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        likelihood = -(beta1**2) * Variable1 - exp(beta2 * beta1) * Variable2 - beta2**4
        simul = beta1 / Variable1 + beta2 / Variable2
        dictOfExpressions = {
            'log_like': likelihood,
            'beta1': beta1,
            'simul': simul,
        }
        my_biogeme = bio.BIOGEME(myData, dictOfExpressions)
        my_biogeme.generate_html = False
        my_biogeme.generate_pickle = False
        my_biogeme.saveIterations = False
        my_biogeme.modelName = 'simpleExample'
        my_biogeme.bootstrap_samples = 10
        self.results = my_biogeme.estimate(run_bootstrap=True)

    def test_calcPValue(self):
        p = calc_p_value(1.96)
        self.assertAlmostEqual(p, 0.05, 2)

    def test_isBoundActive(self):
        beta1 = BetaResult('beta1', -1, (-1, 1))
        self.assertTrue(beta1.is_bound_active())
        beta2 = BetaResult('beta2', 0, (-1, 1))
        self.assertFalse(beta2.is_bound_active())


class TestBioResultsWithoutData(unittest.TestCase):
    def test_no_data_provided(self):
        # Test that no data is provided and the object is created successfully
        result = bioResults()
        self.assertIsNone(
            result.data, "Expected 'data' to be None when no data is provided"
        )

    def test_warning_logged_no_data(self):
        with self.assertLogs('biogeme.results', level='WARNING') as log:
            bioResults()
            self.assertIn(
                'WARNING:biogeme.results:Results: no data provided', log.output
            )

    def test_algorithm_has_not_converged(self):
        result = bioResults()
        self.assertFalse(
            result.algorithm_has_converged(),
            "Expected algorithm_has_converged to be False",
        )

    def test_variance_covariance_missing(self):
        result = bioResults()
        self.assertTrue(
            result.variance_covariance_missing(),
            "Expected variance_covariance_missing to be True",
        )

    def test_write_pickle_without_data(self):
        result = bioResults()
        with self.assertRaises(AttributeError):
            result.write_pickle()

    def test_short_summary_without_data(self):
        result = bioResults()
        summary = result.short_summary()
        self.assertIn('No estimation result is available', summary)

    def test_get_latex_without_data(self):
        result = bioResults()
        latex = result.get_latex()
        self.assertIn('No estimation result is available', latex)

    def test_get_html_without_data(self):
        result = bioResults()
        html = result.get_html()
        self.assertIn('No estimation result is available', html)


if __name__ == '__main__':
    unittest.main()
