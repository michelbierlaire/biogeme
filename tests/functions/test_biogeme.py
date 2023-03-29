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
    bioDraws,
    MonteCarlo,
    RandomVariable,
    PanelLikelihoodTrajectory,
    Numeric,
)
from test_data import getData, getPanelData


class TestBiogeme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(90267)
        rnd.seed(90267)

    @classmethod
    def tearDownClass(cls):
        for file_name in os.listdir('.'):
            if (
                file_name.endswith('.pickle')
                or file_name.endswith('.html')
                or file_name.endswith('.iter')
                or file_name.endswith('.log')
            ):
                os.remove(file_name)

    def get_dict_of_expressions(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        likelihood = -((beta1 * Variable1) ** 2) - (beta2 * Variable2) ** 2
        simul = (beta1 + 2 * beta2) / Variable1 + (beta2 + 2 * beta1) / Variable2
        dict_of_expressions = {
            'loglike': likelihood,
            'weight': Numeric(1),
            'beta1': beta1,
            'simul': simul,
        }
        return dict_of_expressions

    def get_biogeme_instance(self):
        data = getData(1)
        return bio.BIOGEME(data, self.get_dict_of_expressions())

    def get_biogeme_instance_without_bounds(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, None, None, 0)
        beta2 = Beta('beta2', 2.0, None, None, 0)
        likelihood = -((beta1 * Variable1) ** 2) - (beta2 * Variable2) ** 2
        simul = (beta1 + 2 * beta2) / Variable1 + (beta2 + 2 * beta1) / Variable2
        dict_of_expressions = {
            'loglike': likelihood,
            'weight': Numeric(1),
            'beta1': beta1,
            'simul': simul,
        }
        data = getData(1)
        return bio.BIOGEME(data, dict_of_expressions)

    def get_panel_instance(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        likelihood = -((beta1 * Variable1) ** 2) - (beta2 * Variable2) ** 2
        simul = (beta1 + 2 * beta2) / Variable1 + (beta2 + 2 * beta1) / Variable2
        dict_of_expressions = {
            'loglike': likelihood,
            'weight': Numeric(1),
            'beta1': beta1,
            'simul': simul,
        }
        data = getData(1)
        data.panel('Person')
        return bio.BIOGEME(data, dict_of_expressions)

    def test_ctor(self):
        # Test obsolete parameters
        aBiogeme = bio.BIOGEME(
            getData(1),
            self.get_dict_of_expressions(),
        )
        wrong_data = getData(1)
        wrong_data.data.loc['Person', 0] = np.nan
        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                wrong_data,
                self.get_dict_of_expressions(),
            )

        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                getData(1),
                'wrong_object',
            )

        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                getData(1),
                {'loglike': 'wrong_object'},
            )

        wrong_expression = Variable('Variable1') * PanelLikelihoodTrajectory(
            Beta('beta1', -1.0, -3, 3, 0)
        )
        with self.assertRaises(excep.biogemeError):
            bBiogeme = bio.BIOGEME(
                getPanelData(1),
                wrong_expression,
            )

        cBiogeme = bio.BIOGEME(
            getPanelData(1),
            self.get_dict_of_expressions(),
        )

    def test_old_parameters(self):
        b1 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), numberOfDraws=12)
        self.assertEqual(b1.number_of_draws, 12)

        b2 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), numberOfThreads=12)
        self.assertEqual(b2.number_of_threads, 12)

        b3 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), seed=12)
        self.assertEqual(b3.seed_param, 12)

        b4 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), missingData=12)
        self.assertEqual(b4.missing_data, 12)

        b5 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), suggestScales=True)

    def test_saved_iterations(self):
        myBiogeme = self.get_biogeme_instance()
        myBiogeme._loadSavedIteration()
        # Remove the file and try to loaed it again
        myBiogeme._loadSavedIteration()

    def test_random_init_values(self):
        myBiogeme = self.get_biogeme_instance()
        myBiogeme.setRandomInitValues(defaultBound=10)
        for v in myBiogeme.id_manager.free_betas_values:
            self.assertLessEqual(v, 10)
            self.assertGreaterEqual(v, -10)

    def test_parameters(self):
        myBiogeme = self.get_biogeme_instance()
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
        myBiogeme = self.get_biogeme_instance()
        result = myBiogeme.freeBetaNames()
        expected_result = ['beta1', 'beta2']
        self.assertListEqual(result, expected_result)

    def test_bounds_on_beta(self):
        myBiogeme = self.get_biogeme_instance()
        result = myBiogeme.getBoundsOnBeta('beta2')
        expected_result = -3, 10
        self.assertTupleEqual(result, expected_result)
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.getBoundsOnBeta('wrong_name')

    def test_saveIterationsFileName(self):
        myBiogeme = self.get_biogeme_instance()
        myBiogeme.generateHtml = False
        myBiogeme.generatePickle = False
        myBiogeme.saveIterations = False
        myBiogeme.modelName = 'simpleExample'
        f = myBiogeme._saveIterationsFileName()
        self.assertEqual(f, '__simpleExample.iter')

    def test_generateDraws(self):
        the_data = getData(1)
        self.assertIsNone(the_data.theDraws)
        ell = bioDraws('test', 'NORMAL')
        with self.assertRaises(excep.biogemeError):
            b = bio.BIOGEME(the_data, ell)
        b = bio.BIOGEME(the_data, ell, skip_audit=True)
        b._generateDraws(10)
        self.assertTupleEqual(the_data.theDraws.shape, (5, 10, 1))
        ell2 = bioDraws('test', 'NORMAL') + bioDraws('test2', 'UNIFORM')
        b2 = bio.BIOGEME(the_data, ell2, skip_audit=True)
        b2._generateDraws(20)
        self.assertTupleEqual(the_data.theDraws.shape, (5, 20, 2))

    def test_random_variable(self):
        rv = RandomVariable('omega')
        with self.assertRaises(excep.biogemeError):
            b = bio.BIOGEME(getData(1), rv)

    def test_getBoundsOnBeta(self):
        myBiogeme = self.get_biogeme_instance()
        b = myBiogeme.getBoundsOnBeta('beta1')
        self.assertTupleEqual(b, (-3, 3))
        b = myBiogeme.getBoundsOnBeta('beta2')
        self.assertTupleEqual(b, (-3, 10))

    def test_calculateNullLoglikelihood(self):
        myBiogeme = self.get_biogeme_instance()
        null_ell = myBiogeme.calculateNullLoglikelihood({1: 1, 2: 1})
        self.assertAlmostEqual(null_ell, -3.4657359027997265, 2)
        null_ell_2 = myBiogeme.calculateNullLoglikelihood({1: 1, 2: 1, 3: 1})
        self.assertAlmostEqual(null_ell_2, -5.493061443340549, 2)

    def test_calculateInitLikelihood(self):
        myBiogeme = self.get_biogeme_instance()
        res = myBiogeme.calculateInitLikelihood()
        self.assertAlmostEqual(res, -22055, 5)

    def test_calculateLikelihood(self):
        myBiogeme = self.get_biogeme_instance()
        x = myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        res = myBiogeme.calculateLikelihood(xplus, scaled=False)
        self.assertEqual(res, -49500)
        with self.assertRaises(ValueError):
            _ = myBiogeme.calculateLikelihood([1], scaled=False)

    def test_calculateLikelihoodAndDerivatives(self):
        myBiogeme = self.get_biogeme_instance()
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
        myBiogeme.database.data = myBiogeme.database.data[0:0]
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.calculateLikelihoodAndDerivatives(x, scaled=True)

    def test_likelihoodFiniteDifferenceHessian(self):
        myBiogeme = self.get_biogeme_instance()
        x = myBiogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        h = myBiogeme.likelihoodFiniteDifferenceHessian(xplus)
        h_true = [[-110, 0.0], [0.0, -11000]]
        for row, row_true in zip(h, h_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 3)

    def test_checkDerivatives(self):
        myBiogeme = self.get_biogeme_instance()
        _, _, _, gdiff, hdiff = myBiogeme.checkDerivatives([1, 10])
        gdiff_true = [-0.00032656, 0.00545755]
        hdiff_true = [[6.49379217e-09, 0.00000000e00], [0.00000000e00, -1.39698386e-06]]
        for col, col_true in zip(gdiff, gdiff_true):
            self.assertAlmostEqual(col, col_true, 3)
        for row, row_true in zip(hdiff, hdiff_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 3)

    def test_estimate(self):
        myBiogeme = self.get_biogeme_instance()
        myBiogeme.generateHtml = False
        myBiogeme.generatePickle = False
        myBiogeme.saveIterations = False
        myBiogeme.modelName = 'simpleExample'
        myBiogeme.numberOfThreads = 1
        myBiogeme.algorithm_name = 'simple_bounds'
        results = myBiogeme.estimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        panelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
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
        myBiogeme.generatePickle = False

        myBiogeme_without_bounds = self.get_biogeme_instance_without_bounds()
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

        aBiogeme = bio.BIOGEME(getData(1), {'loglike': Numeric(0)})
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.estimate()

        aBiogeme.loglike = None
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.estimate()

    def test_quickEstimate(self):
        myBiogeme = self.get_biogeme_instance()
        myBiogeme.generateHtml = False
        myBiogeme.generatePickle = False
        myBiogeme.saveIterations = False
        myBiogeme.modelName = 'simpleExample'
        myBiogeme.numberOfThreads = 1
        myBiogeme.algorithm_name = 'simple_bounds'
        results = myBiogeme.quickEstimate(bootstrap=10)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        panelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
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
        myBiogeme.generatePickle = False

        myBiogeme_without_bounds = self.get_biogeme_instance_without_bounds()
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

        aBiogeme = bio.BIOGEME(getData(1), {'loglike': Numeric(0)})
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.quickEstimate()

        aBiogeme.loglike = None
        with self.assertRaises(excep.biogemeError):
            _ = aBiogeme.quickEstimate()

    def test_simulate(self):
        myBiogeme = self.get_biogeme_instance()
        results = myBiogeme.estimate()
        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.simulate(theBetaValues=None)

        s = myBiogeme.simulate(results.getBetaValues())
        self.assertAlmostEqual(s.loc[0, 'loglike'], 0, 3)

        the_betas = results.getBetaValues()
        the_betas['any_beta'] = 0.1
        s = myBiogeme.simulate(the_betas)
        self.assertAlmostEqual(s.loc[0, 'loglike'], 0, 3)

        with self.assertRaises(excep.biogemeError):
            _ = myBiogeme.simulate('wrong_object')

        myPanelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
        results = myPanelBiogeme.estimate()
        with self.assertRaises(excep.biogemeError):
            s = myPanelBiogeme.simulate(results.getBetaValues())

        myPanelBiogeme = bio.BIOGEME(
            getPanelData(1),
            {
                'Simul': MonteCarlo(
                    PanelLikelihoodTrajectory(bioDraws('test', 'NORMAL'))
                )
            },
        )
        s = myPanelBiogeme.simulate(results.getBetaValues())

    def test_changeInitValues(self):
        myBiogeme = self.get_biogeme_instance()
        myBiogeme.changeInitValues({'beta2': -100, 'beta1': 3.14156})
        self.assertListEqual(myBiogeme.id_manager.free_betas_values, [3.14156, -100])

    def test_confidenceIntervals(self):
        myBiogeme = self.get_biogeme_instance()
        results = myBiogeme.estimate(bootstrap=100)
        drawsFromBetas = results.getBetasForSensitivityAnalysis(
            myBiogeme.id_manager.free_betas.names
        )
        s = myBiogeme.simulate(results.getBetaValues())
        left, right = myBiogeme.confidenceIntervals(drawsFromBetas)
        self.assertLessEqual(left.loc[0, 'loglike'], s.loc[0, 'loglike'])
        self.assertGreaterEqual(right.loc[0, 'loglike'], s.loc[0, 'loglike'])

    def test_validate(self):
        my_data = getData(1)
        myBiogeme = self.get_biogeme_instance()
        results = myBiogeme.estimate()
        validation_data = my_data.split(slices=2)
        validation_results = myBiogeme.validate(results, validation_data)
        self.assertAlmostEqual(validation_results[0]['Loglikelihood'].sum(), 0, 3)
        self.assertAlmostEqual(validation_results[1]['Loglikelihood'].sum(), 0, 3)

        b = self.get_panel_instance()
        results = b.estimate()
        panel_data = getPanelData(1)
        validation_data = panel_data.split(slices=2)
        with self.assertRaises(excep.biogemeError):
            validation_results = b.validate(results, validation_data)

    def test_optimize(self):
        # Here, we test only the special cases, as it has been called
        # several times by estimate
        myBiogeme = self.get_biogeme_instance()
        myBiogeme.optimize()
        myBiogeme._algorithm = None
        with self.assertRaises(excep.biogemeError):
            myBiogeme.optimize()

    def test_print(self):
        myBiogeme = self.get_biogeme_instance()
        result = str(myBiogeme)[0:20]
        expected_result = 'biogemeModelDefaultN'
        self.assertEqual(result, expected_result)

    def test_files(self):
        myBiogeme = self.get_biogeme_instance()
        myBiogeme.modelName = 'name_for_file'
        myBiogeme.estimate()
        result = myBiogeme.files_of_type('html', all_files=False)
        expected_result = ['name_for_file.html']
        self.assertListEqual(result, expected_result)
        result = myBiogeme.files_of_type('html', all_files=True)
        self.assertGreaterEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
