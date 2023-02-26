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
    MonteCarlo,
    RandomVariable,
    PanelLikelihoodTrajectory,
    Numeric
)
from test_data import getData


class test_biogeme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(90267)
        rnd.seed(90267)

        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        cls.likelihood = (
            - (beta1 * Variable1)**2
            - (beta2 * Variable2)**2
        )
        simul = (beta1 + 2 * beta2) / Variable1 + (beta2 + 2 * beta1) / Variable2
        cls.dict_of_expressions = {
            'loglike': cls.likelihood,
            'weight': Numeric(1),
            'beta1': beta1,
            'simul': simul,
        }
        beta1_without_bounds = Beta('beta1_without_bounds', -1.0, None, None, 0)
        beta2_without_bounds = Beta('beta2_without_bounds', 2.0, None, None, 0)
        cls.likelihood_without_bounds = (
            - (beta1_without_bounds * Variable1)**2
            - (beta2_without_bounds * Variable2)**2
        )
        cls.dict_of_expressions_without_bounds = {
            'loglike': cls.likelihood_without_bounds,
            'weight': Numeric(1),
            'beta1': beta1_without_bounds,
        }

        cls.myData = getData(1)

        cls.myPanelData = getData(1)
        cls.myPanelData.panel('Person')

    @classmethod
    def tearDownClass(cls):
        for file_name in os.listdir('.'):
            if (
                    file_name.endswith('.pickle') or
                    file_name.endswith('.html') or 
                    file_name.endswith('.iter') or
                    file_name.endswith('.log')
            ):
                os.remove(file_name)
                
    def test_ctor(self):
        # Test obsolete parameters
        aBiogeme = bio.BIOGEME(
            self.myData,
            self.dict_of_expressions,
        )
        wrong_data = getData(1)
        wrong_data.data.loc['Person', 0] = np.nan
        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                wrong_data,
                self.dict_of_expressions,
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
        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                self.myPanelData,
                wrong_expression,
            )

        cBiogeme = bio.BIOGEME(
            self.myPanelData,
            self.dict_of_expressions,
        )

    def test_old_parameters(self):
        b1 = bio.BIOGEME(self.myData, self.dict_of_expressions, numberOfDraws=12)
        self.assertEqual(b1.number_of_draws, 12)

        b2 = bio.BIOGEME(self.myData, self.dict_of_expressions, numberOfThreads=12)
        self.assertEqual(b2.number_of_threads, 12)

        b3 = bio.BIOGEME(self.myData, self.dict_of_expressions, seed=12)
        self.assertEqual(b3.seed_param, 12)
        
        b4 = bio.BIOGEME(self.myData, self.dict_of_expressions, missingData=12)
        self.assertEqual(b4.missing_data, 12)

        b5 = bio.BIOGEME(self.myData, self.dict_of_expressions, suggestScales=True)

    def test_saved_iterations(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme._loadSavedIteration()
        # Remove the file and try to loaed it again
        myBiogeme._loadSavedIteration()

    def test_random_init_values(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.setRandomInitValues(defaultBound=10)
        for v in myBiogeme.id_manager.free_betas_values:
            self.assertLessEqual(v, 10)
            self.assertGreaterEqual(v, -10)
        
    def test_parameters(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.algorithm_name = 'scipy'
        self.assertEqual(myBiogeme.algorithm_name, 'scipy')

        myBiogeme.identification_threshold = 1.0e-5
        self.assertEqual(myBiogeme.identification_threshold, 1.0e-5)

        myBiogeme.seed_param = 123
        self.assertEqual(myBiogeme.seed_param, 123)

        myBiogeme.save_iterations = False
        self.assertEqual(myBiogeme.save_iterations, False)

        myBiogeme.saveIterations = False
        self.assertEqual(myBiogeme.saveIterations, False)

        myBiogeme.missing_data = 921967
        self.assertEqual(myBiogeme.missing_data, 921967)

        myBiogeme.missingData = 921967
        self.assertEqual(myBiogeme.missingData, 921967)

        myBiogeme.number_of_threads = 921967
        self.assertEqual(myBiogeme.number_of_threads, 921967)

        myBiogeme.number_of_draws = 921967
        self.assertEqual(myBiogeme.number_of_draws, 921967)

        myBiogeme.numberOfDraws = 921967
        self.assertEqual(myBiogeme.numberOfDraws, 921967)

        myBiogeme.only_robust_stats = True
        self.assertEqual(myBiogeme.only_robust_stats, True)

        myBiogeme.generate_html = False
        self.assertEqual(myBiogeme.generate_html, False)
        
        myBiogeme.generateHtml = False
        self.assertEqual(myBiogeme.generateHtml, False)
        
        myBiogeme.generate_pickle = False
        self.assertEqual(myBiogeme.generate_pickle, False)
        
        myBiogeme.generatePickle = False
        self.assertEqual(myBiogeme.generatePickle, False)
        
        myBiogeme.tolerance = 1.0e-5
        self.assertEqual(myBiogeme.tolerance, 1.0e-5)
        
        myBiogeme.second_derivatives = 0.3
        self.assertEqual(myBiogeme.second_derivatives, 0.3)
        
        myBiogeme.infeasible_cg = True
        self.assertEqual(myBiogeme.infeasible_cg, True)

        myBiogeme.initial_radius = 3.14
        self.assertEqual(myBiogeme.initial_radius, 3.14)

        myBiogeme.steptol = 3.14
        self.assertEqual(myBiogeme.steptol, 3.14)
        
        myBiogeme.enlarging_factor = 3.14
        self.assertEqual(myBiogeme.enlarging_factor, 3.14)
        
        myBiogeme.maxiter = 314
        self.assertEqual(myBiogeme.maxiter, 314)

        myBiogeme.dogleg = True
        self.assertEqual(myBiogeme.dogleg, True)

    def test_free_beta_names(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        result = myBiogeme.freeBetaNames()
        expected_result = ['beta1', 'beta2']
        self.assertListEqual(result, expected_result)

    def test_bounds_on_beta(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        result = myBiogeme.getBoundsOnBeta('beta2')
        expected_result = -3, 10
        self.assertTupleEqual(result, expected_result)
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.getBoundsOnBeta('wrong_name')
        
    def test_saveIterationsFileName(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.generateHtml = False
        myBiogeme.generatePickle = False
        myBiogeme.saveIterations = False
        myBiogeme.modelName = 'simpleExample'
        f = myBiogeme._saveIterationsFileName()
        self.assertEqual(f, '__simpleExample.iter')

    def test_generateDraws(self):
        self.assertIsNone(self.myData.theDraws)
        ell = bioDraws('test', 'NORMAL')
        with self.assertRaises(excep.biogemeError):
            b = bio.BIOGEME(self.myData, ell)
        b = bio.BIOGEME(self.myData, ell, skip_audit=True)
        b._generateDraws(10)
        self.assertTupleEqual(self.myData.theDraws.shape, (5, 10, 1))
        ell2 = bioDraws('test', 'NORMAL') + bioDraws('test2', 'UNIFORM')
        b2 = bio.BIOGEME(self.myData, ell2, skip_audit=True)
        b2._generateDraws(20)
        self.assertTupleEqual(self.myData.theDraws.shape, (5, 20, 2))

    def test_prepare_database(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        with self.assertRaises(ValueError):
            myBiogeme._prepareDatabaseForFormula(sample=-10)
        myBiogeme._prepareDatabaseForFormula(sample=1)
        self.assertEqual(myBiogeme.database.getSampleSize(), 5)
        b = bio.BIOGEME(self.myPanelData, self.dict_of_expressions)
        sample_before = b.database.getSampleSize()
        print(f'Sample before: {sample_before}')
        b._prepareDatabaseForFormula(sample=0.5)
        self.assertEqual(b.database.getSampleSize(), sample_before * 0.5)
        
    def test_random_variable(self):
        rv = RandomVariable('omega')
        with self.assertRaises(excep.biogemeError):
            b = bio.BIOGEME(self.myData, rv)
        
    def test_getBoundsOnBeta(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        b = myBiogeme.getBoundsOnBeta('beta1')
        self.assertTupleEqual(b, (-3, 3))
        b = myBiogeme.getBoundsOnBeta('beta2')
        self.assertTupleEqual(b, (-3, 10))

    def test_calculateNullLoglikelihood(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        null_ell = myBiogeme.calculateNullLoglikelihood({1: 1, 2: 1})
        self.assertAlmostEqual(null_ell, -3.4657359027997265, 2)
        null_ell_2 = myBiogeme.calculateNullLoglikelihood(
            {1: 1, 2: 1, 3: 1}
        )
        self.assertAlmostEqual(null_ell_2, -5.493061443340549, 2)

    def test_calculateInitLikelihood(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        res = myBiogeme.calculateInitLikelihood()
        self.assertAlmostEqual(res, -22055, 5)

    def test_calculateLikelihood(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        x = myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        res = myBiogeme.calculateLikelihood(xplus, scaled=False)
        self.assertEqual(res, -49500)
        with self.assertRaises(ValueError):
            _ = myBiogeme.calculateLikelihood([1], scaled=False)
        

    def test_calculateLikelihoodAndDerivatives(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        x = myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        myBiogeme.save_iterations = True
        f, g, h, bhhh = myBiogeme.calculateLikelihoodAndDerivatives(
            xplus, scaled=False, hessian=True, bhhh=True
        )
        f_true = -49500
        g_true = [0.0, -33000.0]
        h_true = [[0.0, 0.0], [0.0, -11000.0]]
        bhhh_true = [[0.0, 0.0], [0.0, 352440000.0]]
        self.assertEqual(f_true, f)
        self.assertListEqual(g_true, g.tolist())
        self.assertListEqual(h_true, h.tolist())
        self.assertListEqual(bhhh_true, bhhh.tolist())
        with self.assertRaises(ValueError):
            _ = myBiogeme.calculateLikelihoodAndDerivatives([1], scaled=False)
        myBiogeme.database.data = myBiogeme.database.data[0: 0]
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.calculateLikelihoodAndDerivatives(x, scaled=True)
        
    def test_likelihoodFiniteDifferenceHessian(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        x = myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        h = myBiogeme.likelihoodFiniteDifferenceHessian(xplus)
        h_true = [[-110, 0.0], [0.0, -11000]]
        for row, row_true in zip(h, h_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 3)

    def test_checkDerivatives(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        _, _, _, gdiff, hdiff = myBiogeme.checkDerivatives()
        gdiff_true = [-0.0031248546037545566, -0.05513429641723633]
        hdiff_true = [
            [0.0, 7.36597983e-09],
            [-1.61387920e-07, -1.3038514225627296e-05],
        ]
        for col, col_true in zip(gdiff, gdiff_true):
            self.assertAlmostEqual(col, col_true, 5)
        for row, row_true in zip(hdiff, hdiff_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 5)

    def test_estimate(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.generateHtml = False
        myBiogeme.generatePickle = False
        myBiogeme.saveIterations = False
        myBiogeme.modelName = 'simpleExample'
        myBiogeme.numberOfThreads = 1
        myBiogeme.algorithm_name = 'simple_bounds'
        results = myBiogeme.estimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        panelBiogeme = bio.BIOGEME(self.myPanelData, self.dict_of_expressions)
        results = panelBiogeme.estimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        
        myBiogeme.algorithm_name = 'scipy'
        myBiogeme.saveIterations = True
        results = myBiogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        # We try to recycle, while there is no pickle file yet.
        results = myBiogeme.estimate(recycle=True)
        # We estimate the model twice to generate two pickle files. 
        myBiogeme.generatePickle = True
        results = myBiogeme.estimate()
        results = myBiogeme.estimate()
        results = myBiogeme.estimate(recycle=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme.generatePickle=False

        myBiogeme_without_bounds = bio.BIOGEME(
            self.myData,
            self.dict_of_expressions_without_bounds
        )
        myBiogeme_without_bounds.generateHtml = False
        myBiogeme_without_bounds.generatePickle = False
        myBiogeme_without_bounds.saveIterations = False

        
        myBiogeme_without_bounds.algorithm_name = 'TR-newton'
        results = myBiogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme_without_bounds.algorithm_name = 'TR-BFGS'
        results = myBiogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme_without_bounds.algorithm_name = 'LS-newton'
        results = myBiogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme_without_bounds.algorithm_name = 'LS-BFGS'
        results = myBiogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.estimate(algorithm='any_algo')
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.estimate(algoParameters='any_param')
            
        aBiogeme = bio.BIOGEME(self.myData, {'loglike': Numeric(0)})
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.estimate()

        aBiogeme.loglike = None
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.estimate()

    def test_quickEstimate(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.generateHtml = False
        myBiogeme.generatePickle = False
        myBiogeme.saveIterations = False
        myBiogeme.modelName = 'simpleExample'
        myBiogeme.numberOfThreads = 1
        myBiogeme.algorithm_name = 'simple_bounds'
        results = myBiogeme.quickEstimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        panelBiogeme = bio.BIOGEME(self.myPanelData, self.dict_of_expressions)
        results = panelBiogeme.quickEstimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        
        myBiogeme.algorithm_name = 'scipy'
        myBiogeme.saveIterations = True
        results = myBiogeme.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        # We try to recycle, while there is no pickle file yet.
        results = myBiogeme.quickEstimate(recycle=True)
        # We quickEstimate the model twice to generate two pickle files. 
        myBiogeme.generatePickle = True
        results = myBiogeme.quickEstimate()
        results = myBiogeme.quickEstimate()
        results = myBiogeme.quickEstimate(recycle=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme.generatePickle=False

        myBiogeme_without_bounds = bio.BIOGEME(
            self.myData,
            self.dict_of_expressions_without_bounds
        )
        myBiogeme_without_bounds.generateHtml = False
        myBiogeme_without_bounds.generatePickle = False
        myBiogeme_without_bounds.saveIterations = False

        
        myBiogeme_without_bounds.algorithm_name = 'TR-newton'
        results = myBiogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme_without_bounds.algorithm_name = 'TR-BFGS'
        results = myBiogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme_without_bounds.algorithm_name = 'LS-newton'
        results = myBiogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        myBiogeme_without_bounds.algorithm_name = 'LS-BFGS'
        results = myBiogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.quickEstimate(algorithm='any_algo')
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.quickEstimate(algoParameters='any_param')
            
        aBiogeme = bio.BIOGEME(self.myData, {'loglike': Numeric(0)})
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.quickEstimate()

        aBiogeme.loglike = None
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.quickEstimate()

            
    def test_simulate(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        results = myBiogeme.estimate()
        s = myBiogeme.simulate(results.getBetaValues())
        self.assertAlmostEqual(s.loc[0, 'loglike'], 0, 3)

        s = myBiogeme.simulate()
        self.assertAlmostEqual(s.loc[0, 'loglike'], 0, 3)

        s = myBiogeme.simulate({'any_beta': 0.1})
        self.assertAlmostEqual(s.loc[0, 'loglike'], 0, 3)
        
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.simulate('wrong_object')
        
        myPanelBiogeme = bio.BIOGEME(self.myPanelData, self.dict_of_expressions)
        results = myPanelBiogeme.estimate()
        with self.assertRaises(excep.biogemeError):
            s = myPanelBiogeme.simulate(results.getBetaValues())

        myPanelBiogeme = bio.BIOGEME(
            self.myPanelData,
            {
                'Simul': MonteCarlo(PanelLikelihoodTrajectory(bioDraws('test', 'NORMAL')))
            }
        )
        s = myPanelBiogeme.simulate()
            

    def test_changeInitValues(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.changeInitValues({'beta2': -100, 'beta1': 3.14156})
        self.assertListEqual(
            myBiogeme.id_manager.free_betas_values, [3.14156, -100]
        )

    def test_confidenceIntervals(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        results = myBiogeme.estimate(bootstrap=100)
        drawsFromBetas = results.getBetasForSensitivityAnalysis(
            myBiogeme.id_manager.free_betas.names
        )
        s = myBiogeme.simulate(results.getBetaValues())
        left, right = myBiogeme.confidenceIntervals(drawsFromBetas)
        self.assertLessEqual(left.loc[0, 'loglike'], s.loc[0, 'loglike'])
        self.assertGreaterEqual(right.loc[0, 'loglike'], s.loc[0, 'loglike'])

    def test_validate(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        results = myBiogeme.estimate()
        validation_data = self.myData.split(slices=2)
        validation_results = myBiogeme.validate(results, validation_data)
        self.assertAlmostEqual(validation_results[0]['Loglikelihood'].sum(), 0, 3)
        self.assertAlmostEqual(validation_results[1]['Loglikelihood'].sum(), 0, 3)

        b = bio.BIOGEME(self.myPanelData, self.dict_of_expressions)
        results = b.estimate()
        validation_data = self.myPanelData.split(slices=2)
        with self.assertRaises(excep.biogemeError):
            validation_results = b.validate(results, validation_data)

    def test_optimize(self):
        # Here, we test only the special cases, as it has been called
        # several times by estimate
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.optimize()
        myBiogeme._algorithm = None
        with self.assertRaises(excep.biogemeError):
            myBiogeme.optimize()

    def test_logfile(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.createLogFile(verbosity=2)

    def test_print(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        result = str(myBiogeme)[0:20]
        expected_result = 'biogemeModelDefaultN'
        self.assertEqual(result, expected_result)


    def test_files(self):
        myBiogeme = bio.BIOGEME(self.myData, self.dict_of_expressions)
        myBiogeme.modelName = 'name_for_file'
        myBiogeme.estimate()
        result = myBiogeme.files_of_type('html', all_files=False)
        expected_result = ['name_for_file.html']
        self.assertListEqual(result, expected_result)
        result = myBiogeme.files_of_type('html', all_files=True)
        self.assertGreaterEqual(len(result), 1)
if __name__ == '__main__':
    unittest.main()
