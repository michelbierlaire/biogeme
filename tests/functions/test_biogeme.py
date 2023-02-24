"""
Test the biogeme module

:author: Michel Bierlaire
:date: Wed Apr 29 18:32:42 2020

"""
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import unittest
import random as rnd
import numpy as np
import biogeme.biogeme as bio
import biogeme.exceptions as excep
from biogeme.expressions import (
    Variable,
    Beta,
    exp,
    bioDraws,
    PanelLikelihoodTrajectory,
    Numeric
)
from test_data import getData


class test_biogeme(unittest.TestCase):
    def setUp(self):
        np.random.seed(90267)
        rnd.seed(90267)

        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        self.likelihood = (
            -(beta1**2) * Variable1
            - exp(beta2 * beta1) * Variable2
            - beta2**4
        )
        simul = beta1 / Variable1 + beta2 / Variable2
        self.dictOfExpressions = {
            'loglike': self.likelihood,
            'weight': Numeric(1),
            'beta1': beta1,
            'simul': simul,
        }

        self.myData = getData(1)
        self.myBiogeme = bio.BIOGEME(self.myData, self.dictOfExpressions)
        self.myBiogeme.generateHtml = False
        self.myBiogeme.generatePickle = False
        self.myBiogeme.saveIterations = False
        self.myBiogeme.modelName = 'simpleExample'

    def test_ctor(self):
        # Test obsolete parameters
        aBiogeme = bio.BIOGEME(
            self.myData,
            self.dictOfExpressions,
            suggestScales=False,
            seed=123
        )
        self.assertEqual(aBiogeme.seed_param, 123)
        wrong_data = getData(1)
        wrong_data.data.loc['Person', 0] = np.nan
        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                wrong_data,
                self.dictOfExpressions,
            )

        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                self.myData,
                'wrong_object',
            )

        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                self.myData,
                {'loglike': 'wrong_object'},
            )
            
        wrong_expression = (
            Variable('Variable1') *
            PanelLikelihoodTrajectory(Beta('beta1', -1.0, -3, 3, 0))
        )
        panel_data = getData(1)
        panel_data.panel('Person')
        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                panel_data,
                wrong_expression,
            )

        cBiogeme = bio.BIOGEME(
            panel_data,
            self.dictOfExpressions,
        )

    def test_parameters(self):
        self.myBiogeme.algorithm_name = 'scipy'
        self.assertEqual(self.myBiogeme.algorithm_name, 'scipy')

        self.myBiogeme.identification_threshold = 1.0e-5
        self.assertEqual(self.myBiogeme.identification_threshold, 1.0e-5)

        self.myBiogeme.seed_param = 123
        self.assertEqual(self.myBiogeme.seed_param, 123)

        self.myBiogeme.save_iterations = False
        self.assertEqual(self.myBiogeme.save_iterations, False)

        self.myBiogeme.saveIterations = False
        self.assertEqual(self.myBiogeme.saveIterations, False)

        self.myBiogeme.skip_audit = True
        self.assertEqual(self.myBiogeme.skip_audit, True)

        self.myBiogeme.missing_data = 921967
        self.assertEqual(self.myBiogeme.missing_data, 921967)

        self.myBiogeme.missingData = 921967
        self.assertEqual(self.myBiogeme.missingData, 921967)

        self.myBiogeme.number_of_threads = 921967
        self.assertEqual(self.myBiogeme.number_of_threads, 921967)

        self.myBiogeme.number_of_draws = 921967
        self.assertEqual(self.myBiogeme.number_of_draws, 921967)

        self.myBiogeme.numberOfDraws = 921967
        self.assertEqual(self.myBiogeme.numberOfDraws, 921967)

        self.myBiogeme.only_robust_stats = True
        self.assertEqual(self.myBiogeme.only_robust_stats, True)

        self.myBiogeme.generate_html = False
        self.assertEqual(self.myBiogeme.generate_html, False)
        
        self.myBiogeme.generateHtml = False
        self.assertEqual(self.myBiogeme.generateHtml, False)
        
        self.myBiogeme.generate_pickle = False
        self.assertEqual(self.myBiogeme.generate_pickle, False)
        
        self.myBiogeme.generatePickle = False
        self.assertEqual(self.myBiogeme.generatePickle, False)
        
        self.myBiogeme.tolerance = 1.0e-5
        self.assertEqual(self.myBiogeme.tolerance, 1.0e-5)
        
        self.myBiogeme.second_derivatives = 0.3
        self.assertEqual(self.myBiogeme.second_derivatives, 0.3)
        
        self.myBiogeme.infeasible_cg = True
        self.assertEqual(self.myBiogeme.infeasible_cg, True)
        
    def test_saveIterationsFileName(self):
        f = self.myBiogeme._saveIterationsFileName()
        self.assertEqual(f, '__simpleExample.iter')

    def test_generateDraws(self):
        self.assertIsNone(self.myData.theDraws)
        ell = bioDraws('test', 'NORMAL')
        b = bio.BIOGEME(self.myData, ell, skipAudit=True)
        b._generateDraws(10)
        self.assertTupleEqual(self.myData.theDraws.shape, (5, 10, 1))
        ell2 = bioDraws('test', 'NORMAL') + bioDraws('test2', 'UNIFORM')
        b2 = bio.BIOGEME(self.myData, ell2, skipAudit=True)
        b2._generateDraws(20)
        self.assertTupleEqual(self.myData.theDraws.shape, (5, 20, 2))

    def test_getBoundsOnBeta(self):
        b = self.myBiogeme.getBoundsOnBeta('beta1')
        self.assertTupleEqual(b, (-3, 3))
        b = self.myBiogeme.getBoundsOnBeta('beta2')
        self.assertTupleEqual(b, (-3, 10))

    def test_calculateNullLoglikelihood(self):
        null_ell = self.myBiogeme.calculateNullLoglikelihood({1: 1, 2: 1})
        self.assertAlmostEqual(null_ell, -3.4657359027997265, 2)
        null_ell_2 = self.myBiogeme.calculateNullLoglikelihood(
            {1: 1, 2: 1, 3: 1}
        )
        self.assertAlmostEqual(null_ell_2, -5.493061443340549, 2)

    def test_calculateInitLikelihood(self):
        res = self.myBiogeme.calculateInitLikelihood()
        self.assertAlmostEqual(res, -115.30029248549191, 5)

    def test_calculateLikelihood(self):
        x = self.myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        res = self.myBiogeme.calculateLikelihood(xplus, scaled=False)
        self.assertEqual(res, -555)

    def test_calculateLikelihoodAndDerivatives(self):
        x = self.myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        f, g, h, bhhh = self.myBiogeme.calculateLikelihoodAndDerivatives(
            xplus, scaled=False, hessian=True, bhhh=True
        )
        f_true = -555.0
        g_true = [-450.0, -540.0]
        h_true = [[-1350.0, -150.0], [-150.0, -540.0]]
        bhhh_true = [[49500.0, 48600.0], [48600.0, 58320.0]]
        self.assertEqual(f_true, f)
        self.assertListEqual(g_true, g.tolist())
        self.assertListEqual(h_true, h.tolist())
        self.assertListEqual(bhhh_true, bhhh.tolist())

    def test_likelihoodFiniteDifferenceHessian(self):
        x = self.myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        h = self.myBiogeme.likelihoodFiniteDifferenceHessian(xplus)
        h_true = [[-1380.00020229, -150.0], [-150.0000451, -540.00005396]]
        for row, row_true in zip(h, h_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 5)

    def test_checkDerivatives(self):
        _, _, _, gdiff, hdiff = self.myBiogeme.checkDerivatives()
        gdiff_true = [-5.42793187e-06, 2.60800035e-05]
        hdiff_true = [
            [-8.04552171e-06, 7.36597983e-09],
            [-1.61387920e-07, 2.22928137e-05],
        ]
        for col, col_true in zip(gdiff, gdiff_true):
            self.assertAlmostEqual(col, col_true, 5)
        for row, row_true in zip(hdiff, hdiff_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 5)

    def test_estimate(self):
        self.myBiogeme.numberOfThreads = 1
        results = self.myBiogeme.estimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, -67.0654904797005, 5)

    def test_simulate(self):
        results = self.myBiogeme.estimate()
        s = self.myBiogeme.simulate(results.getBetaValues())
        self.assertAlmostEqual(s.loc[0, 'loglike'], -6.092208083991222, 3)

    def test_changeInitValues(self):
        self.myBiogeme.changeInitValues({'beta2': -100, 'beta1': 3.14156})
        self.assertListEqual(
            self.myBiogeme.id_manager.free_betas_values, [3.14156, -100]
        )

    def test_confidenceIntervals(self):
        results = self.myBiogeme.estimate(bootstrap=10)
        drawsFromBetas = results.getBetasForSensitivityAnalysis(
            self.myBiogeme.id_manager.free_betas.names
        )
        s = self.myBiogeme.simulate(results.getBetaValues())
        left, right = self.myBiogeme.confidenceIntervals(drawsFromBetas)
        self.assertLess(left.loc[0, 'loglike'], s.loc[0, 'loglike'])
        self.assertGreater(right.loc[0, 'loglike'], s.loc[0, 'loglike'])


if __name__ == '__main__':
    unittest.main()
