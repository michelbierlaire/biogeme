"""
Test the optimization module

:author: Michel Bierlaire
:data: Wed Apr 29 17:45:19 2020
"""

# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring


import random as rnd
import unittest

import numpy as np

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable
from biogeme.parameters import Parameters
from test_data import getData


class test_optimization(unittest.TestCase):
    def setUp(self):
        np.random.seed(90267)
        rnd.seed(90267)
        Choice = Variable('Choice')
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', 0, None, None, 0)
        beta2 = Beta('beta2', 0, None, None, 0)
        V1 = beta1 * Variable1
        V2 = beta2 * Variable2
        V3 = 0
        V = {1: V1, 2: V2, 3: V3}

        self.likelihood = models.loglogit(V, av=None, i=Choice)

        self.scipy_configuration = Parameters()
        self.scipy_configuration.set_value(
            name='optimization_algorithm', value='scipy', section='Estimation'
        )
        self.ls_configuration = Parameters()
        self.ls_configuration.set_value(
            name='optimization_algorithm', value='LS-newton', section='Estimation'
        )
        self.tr_configuration = Parameters()
        self.tr_configuration.set_value(
            name='optimization_algorithm', value='TR-newton', section='Estimation'
        )
        self.simple_bounds_configuration = Parameters()
        self.simple_bounds_configuration.set_value(
            name='optimization_algorithm', value='simple_bounds', section='Estimation'
        )

    def testBioScipy(self):
        my_biogeme = bio.BIOGEME(
            getData(1),
            self.likelihood,
            parameters=self.scipy_configuration,
            generate_html=False,
            generate_yaml=False,
            save_iterations=False,
        )
        my_biogeme.model_name = 'simpleExample'
        results = my_biogeme.estimate()
        beta = results.get_beta_values()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)

    def _testBioNewtonLineSearch(self):
        my_biogeme = bio.BIOGEME(
            getData(1),
            self.likelihood,
            parameters=self.ls_configuration,
            generate_html=False,
            generate_yaml=False,
            save_iterations=False,
        )
        my_biogeme.model_name = 'simpleExample'
        results = my_biogeme.estimate()
        beta = results.get_beta_values()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)

    def testBioNewtonTrustRegion(self):
        my_biogeme = bio.BIOGEME(
            getData(1),
            self.likelihood,
            parameters=self.tr_configuration,
            generate_html=False,
            generate_yaml=False,
            save_iterations=False,
        )
        my_biogeme.model_name = 'simpleExample'
        results = my_biogeme.estimate()
        beta = results.get_beta_values()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)

    def testBioNewtonSimpleBounds(self):
        my_biogeme = bio.BIOGEME(
            getData(1),
            self.likelihood,
            parameters=self.simple_bounds_configuration,
            generate_html=False,
            generate_yaml=False,
            save_iterations=False,
        )
        my_biogeme.model_name = 'simpleExample'
        results = my_biogeme.estimate()
        beta = results.get_beta_values()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)


if __name__ == '__main__':
    unittest.main()
