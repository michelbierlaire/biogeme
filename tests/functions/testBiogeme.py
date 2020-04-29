"""
Test the biogeme module

:author: Michel Bierlaire
:data: Wed Apr 29 18:32:42 2020

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
import random as rnd
import numpy as np
import biogeme.biogeme as bio
from biogeme.expressions import Variable, Beta, exp
from testData import myData1

class testBiogeme(unittest.TestCase):
    def setUp(self):
        np.random.seed(90267)
        rnd.seed(90267)

        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        likelihood = -beta1**2 * Variable1 - exp(beta2 * beta1) * Variable2 - beta2**4
        simul = beta1 / Variable1 + beta2 / Variable2
        dictOfExpressions = {'loglike':likelihood,
                             'beta1':beta1,
                             'simul':simul}

        self.myBiogeme = bio.BIOGEME(myData1, dictOfExpressions)
        self.myBiogeme.modelName = 'simpleExample'

    def test_calculateInitLikelihood(self):
        res = self.myBiogeme.calculateInitLikelihood()
        self.assertAlmostEqual(res, -115.30029248549191, 5)

    def test_calculateLikelihood(self):
        x = self.myBiogeme.betaInitValues
        xplus = [v + 1 for v in x]
        res = self.myBiogeme.calculateLikelihood(xplus, scaled=False)
        self.assertEqual(res, -555)

    def test_calculateLikelihoodAndDerivatives(self):
        x = self.myBiogeme.betaInitValues
        xplus = [v + 1 for v in x]
        f, g, h, bhhh = self.myBiogeme.calculateLikelihoodAndDerivatives(xplus,
                                                                         scaled=False,
                                                                         hessian=True,
                                                                         bhhh=True)
        f_true = -555.0
        g_true = [-450., -540.]
        h_true = [[-1350., -150.], [-150., -540.]]
        bhhh_true = [[49500., 48600.], [48600., 58320.]]
        self.assertEqual(f_true, f)
        self.assertListEqual(g_true, g.tolist())
        self.assertListEqual(h_true, h.tolist())
        self.assertListEqual(bhhh_true, bhhh.tolist())

    def test_likelihoodFiniteDifferenceHessian(self):
        x = self.myBiogeme.betaInitValues
        xplus = [v + 1 for v in x]
        h = self.myBiogeme.likelihoodFiniteDifferenceHessian(xplus)
        h_true = [[-1380.00020229, -150.], [-150.0000451, -540.00005396]]
        for row, row_true in zip(h, h_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 5)

    def test_checkDerivatives(self):
        _, _, _, gdiff, hdiff = self.myBiogeme.checkDerivatives()
        gdiff_true = [-5.42793187e-06, 2.60800035e-05]
        hdiff_true = [[-8.04552171e-06, 7.36597983e-09], [-1.61387920e-07, 2.22928137e-05]]
        for col, col_true in zip(gdiff, gdiff_true):
            self.assertAlmostEqual(col, col_true, 5)
        for row, row_true in zip(hdiff, hdiff_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 5)

    def test_estimate(self):
        results = self.myBiogeme.estimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, -67.0654904797005, 5)

    def test_simulate(self):
        results = self.myBiogeme.estimate()
        s = self.myBiogeme.simulate(results.getBetaValues())
        self.assertAlmostEqual(s.loc[0, 'loglike'], -6.09223388037049, 5)

    def test_confidenceIntervals(self):
        results = self.myBiogeme.estimate(bootstrap=10)
        drawsFromBetas = results.getBetasForSensitivityAnalysis(self.myBiogeme.freeBetaNames)
        s = self.myBiogeme.simulate(results.getBetaValues())
        left, right = self.myBiogeme.confidenceIntervals(drawsFromBetas)
        self.assertLess(left.loc[0, 'loglike'], s.loc[0, 'loglike'])
        self.assertGreater(right.loc[0, 'loglike'], s.loc[0, 'loglike'])

if __name__ == '__main__':
    unittest.main()
