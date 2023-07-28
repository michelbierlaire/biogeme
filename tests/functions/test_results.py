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
import biogeme.results as res
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
        likelihood = (
            -(beta1**2) * Variable1 - exp(beta2 * beta1) * Variable2 - beta2**4
        )
        simul = beta1 / Variable1 + beta2 / Variable2
        dictOfExpressions = {
            'loglike': likelihood,
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
        p = res.calcPValue(1.96)
        self.assertAlmostEqual(p, 0.05, 2)

    def test_isBoundActive(self):
        beta1 = res.beta('beta1', -1, (-1, 1))
        self.assertTrue(beta1.is_bound_active())
        beta2 = res.beta('beta2', 0, (-1, 1))
        self.assertFalse(beta2.is_bound_active())


if __name__ == '__main__':
    unittest.main()
