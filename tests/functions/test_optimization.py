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


import os
import shutil
import unittest
import random as rnd

import numpy as np
import tempfile

import biogeme.biogeme as bio
from biogeme import models
import biogeme.optimization as opt
import biogeme.exceptions as excep
from biogeme.expressions import Variable, Beta
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

        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.scipy_file = os.path.join(self.test_dir, 'scipy.toml')
        with open(self.scipy_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "scipy"', file=f)
        self.ls_file = os.path.join(self.test_dir, 'line_search.toml')
        with open(self.ls_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "LS-newton"', file=f)
        self.tr_file = os.path.join(self.test_dir, 'trust_region.toml')
        with open(self.tr_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "TR-newton"', file=f)
        self.simple_bounds_file = os.path.join(self.test_dir, 'simple_bounds.toml')
        with open(self.simple_bounds_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "simple_bounds"', file=f)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testBioScipy(self):
        my_biogeme = bio.BIOGEME(
            getData(1), self.likelihood, parameter_file=self.scipy_file
        )
        my_biogeme.modelName = 'simpleExample'
        my_biogeme.generateHtml = False
        my_biogeme.generatePickle = False
        my_biogeme.saveIterations = False
        results = my_biogeme.estimate()
        beta = results.getBetaValues()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)

    def testBioNewtonLineSearch(self):
        my_biogeme = bio.BIOGEME(
            getData(1), self.likelihood, parameter_file=self.ls_file
        )
        my_biogeme.modelName = 'simpleExample'
        my_biogeme.generateHtml = False
        my_biogeme.generatePickle = False
        my_biogeme.saveIterations = False
        results = my_biogeme.estimate()
        beta = results.getBetaValues()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)

    def testBioNewtonTrustRegion(self):
        my_biogeme = bio.BIOGEME(
            getData(1), self.likelihood, parameter_file=self.tr_file
        )
        my_biogeme.modelName = 'simpleExample'
        my_biogeme.generateHtml = False
        my_biogeme.generatePickle = False
        my_biogeme.saveIterations = False
        results = my_biogeme.estimate()
        beta = results.getBetaValues()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)

    def testBioNewtonSimpleBounds(self):
        my_biogeme = bio.BIOGEME(
            getData(1), self.likelihood, parameter_file=self.simple_bounds_file
        )
        my_biogeme.modelName = 'simpleExample'
        my_biogeme.generateHtml = False
        my_biogeme.generatePickle = False
        my_biogeme.saveIterations = False
        results = my_biogeme.estimate()
        beta = results.getBetaValues()
        self.assertAlmostEqual(beta['beta1'], 0.144546, 3)
        self.assertAlmostEqual(beta['beta2'], 0.023502, 3)


if __name__ == '__main__':
    unittest.main()
