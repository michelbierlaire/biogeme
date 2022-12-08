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
import biogeme.algorithms as algo
import biogeme.optimization as opt
import biogeme.exceptions as excep
from biogeme.expressions import Variable, Beta
from test_data import getData


class rosenbrock(algo.functionToMinimize):
    def __init__(self):
        self.x = None

    def setVariables(self, x):
        self.x = x

    def f(self, batch=None):
        if batch is not None:
            raise excep.biogemeError('This function is not data driven.')
        n = len(self.x)
        f = sum(
            100.0 * (self.x[i + 1] - self.x[i] ** 2) ** 2
            + (1.0 - self.x[i]) ** 2
            for i in range(n - 1)
        )
        return f

    def g(self):
        n = len(self.x)
        g = np.zeros(n)
        for i in range(n - 1):
            g[i] = (
                g[i]
                - 400 * self.x[i] * (self.x[i + 1] - self.x[i] ** 2)
                - 2 * (1 - self.x[i])
            )
            g[i + 1] = g[i + 1] + 200 * (self.x[i + 1] - self.x[i] ** 2)
        return g

    def h(self):
        n = len(self.x)
        H = np.zeros((n, n))
        for i in range(n - 1):
            H[i][i] = H[i][i] - 400 * self.x[i + 1] + 1200 * self.x[i] ** 2 + 2
            H[i + 1][i] = H[i + 1][i] - 400 * self.x[i]
            H[i][i + 1] = H[i][i + 1] - 400 * self.x[i]
            H[i + 1][i + 1] = H[i + 1][i + 1] + 200
        return H

    def f_g(self, batch=None):
        if batch is not None:
            raise excep.biogemeError('This function is not data driven.')
        return self.f(), self.g()

    def f_g_h(self, batch=None):
        if batch is not None:
            raise excep.biogemeError('This function is not data driven.')
        return self.f(), self.g(), self.h()

    def f_g_bhhh(self, batch=None):
        raise excep.biogemeError('This function is not data driven.')


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
        self.theFunction = rosenbrock()

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
        self.simple_bounds_file = os.path.join(
            self.test_dir, 'simple_bounds.toml'
        )
        with open(self.simple_bounds_file, 'w', encoding='utf-8') as f:
            print('[Estimation]', file=f)
            print('optimization_algorithm = "simple_bounds"', file=f)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testSchnabelEskow(self):
        A = np.array(
            [
                [0.3571, -0.1030, 0.0274, -0.0459],
                [-0.1030, 0.2525, 0.0736, -0.3845],
                [0.0274, 0.0736, 0.2340, -0.2878],
                [-0.0459, -0.3845, -0.2878, 0.5549],
            ]
        )
        L, E, P = algo.schnabelEskow(A)
        self.assertAlmostEqual(L[0][0], 0.7449161, 5)
        diff = P @ L @ L.T @ P.T - E - A
        for i in list(diff.flatten()):
            self.assertAlmostEqual(i, 0.0, 5)

    def testSchnabelEskow2(self):
        A = np.array(
            [
                [1890.3, -1705.6, -315.8, 3000.3],
                [-1705.6, 1538.3, 284.9, -2706.6],
                [-315.8, 284.9, 52.5, -501.2],
                [3000.3, -2706.6, -501.2, 4760.8],
            ]
        )
        L, E, P = algo.schnabelEskow(A)
        self.assertAlmostEqual(L[0][0], 6.89985507e01, 5)
        diff = P @ L @ L.T @ P.T - E - A
        for i in list(diff.flatten()):
            self.assertAlmostEqual(i, 0.0, 5)

    def testLineSearch(self):
        x = np.array([-1.5, 1.5])
        self.theFunction.setVariables(x)
        f, g = self.theFunction.f_g()
        alpha, _ = algo.lineSearch(self.theFunction, x, f, g, -g)
        self.assertAlmostEqual(alpha, 0.0009765625)

    def testNewton(self):
        x0 = np.array([-1.5, 1.5])
        xstar, _ = algo.newtonLineSearch(self.theFunction, x0)
        for i in list(xstar):
            self.assertAlmostEqual(i, 1, 4)

    def testNewtonTrustRegion(self):
        x0 = np.array([-1.5, 1.5])
        xstar, _ = algo.newtonTrustRegion(self.theFunction, x0)
        for i in list(xstar):
            self.assertAlmostEqual(i, 1, 4)

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
