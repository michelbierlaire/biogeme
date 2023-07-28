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
from biogeme.catalog import Catalog
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
        with self.assertRaises(excep.BiogemeError):
            bBiogeme = bio.BIOGEME(
                wrong_data,
                self.get_dict_of_expressions(),
            )

        with self.assertRaises(excep.BiogemeError):
            bBiogeme = bio.BIOGEME(
                getData(1),
                'wrong_object',
            )

        with self.assertRaises(excep.BiogemeError):
            bBiogeme = bio.BIOGEME(
                getData(1),
                {'loglike': 'wrong_object'},
            )

        wrong_expression = Variable('Variable1') * PanelLikelihoodTrajectory(
            Beta('beta1', -1.0, -3, 3, 0)
        )
        with self.assertRaises(excep.BiogemeError):
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
        my_biogeme = self.get_biogeme_instance()
        my_biogeme._loadSavedIteration()
        # Remove the file and try to loaed it again
        my_biogeme._loadSavedIteration()

    def test_random_init_values(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.setRandomInitValues(defaultBound=10)
        for v in my_biogeme.id_manager.free_betas_values:
            self.assertLessEqual(v, 10)
            self.assertGreaterEqual(v, -10)

    def test_parameters(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.algorithm_name = 'scipy'
        self.assertEqual(my_biogeme.algorithm_name, 'scipy')

        my_biogeme.identification_threshold = 1.0e-5
        self.assertEqual(my_biogeme.identification_threshold, 1.0e-5)

        my_biogeme.seed_param = 123
        self.assertEqual(my_biogeme.seed_param, 123)

        my_biogeme.save_iterations = False
        self.assertEqual(my_biogeme.save_iterations, False)

        my_biogeme.saveIterations = False
        self.assertEqual(my_biogeme.saveIterations, False)

        my_biogeme.missing_data = 921967
        self.assertEqual(my_biogeme.missing_data, 921967)

        my_biogeme.missingData = 921967
        self.assertEqual(my_biogeme.missingData, 921967)

        my_biogeme.number_of_threads = 921967
        self.assertEqual(my_biogeme.number_of_threads, 921967)

        my_biogeme.number_of_draws = 921967
        self.assertEqual(my_biogeme.number_of_draws, 921967)

        my_biogeme.numberOfDraws = 921967
        self.assertEqual(my_biogeme.numberOfDraws, 921967)

        my_biogeme.only_robust_stats = True
        self.assertEqual(my_biogeme.only_robust_stats, True)

        my_biogeme.generate_html = False
        self.assertEqual(my_biogeme.generate_html, False)

        my_biogeme.generate_html = False
        self.assertEqual(my_biogeme.generate_html, False)

        my_biogeme.generate_pickle = False
        self.assertEqual(my_biogeme.generate_pickle, False)

        my_biogeme.generate_pickle = False
        self.assertEqual(my_biogeme.generate_pickle, False)

        my_biogeme.tolerance = 1.0e-5
        self.assertEqual(my_biogeme.tolerance, 1.0e-5)

        my_biogeme.second_derivatives = 0.3
        self.assertEqual(my_biogeme.second_derivatives, 0.3)

        my_biogeme.infeasible_cg = True
        self.assertEqual(my_biogeme.infeasible_cg, True)

        my_biogeme.initial_radius = 3.14
        self.assertEqual(my_biogeme.initial_radius, 3.14)

        my_biogeme.steptol = 3.14
        self.assertEqual(my_biogeme.steptol, 3.14)

        my_biogeme.enlarging_factor = 3.14
        self.assertEqual(my_biogeme.enlarging_factor, 3.14)

        my_biogeme.maxiter = 314
        self.assertEqual(my_biogeme.maxiter, 314)

        my_biogeme.dogleg = True
        self.assertEqual(my_biogeme.dogleg, True)

    def test_free_beta_names(self):
        my_biogeme = self.get_biogeme_instance()
        result = my_biogeme.free_beta_names()
        expected_result = ['beta1', 'beta2']
        self.assertListEqual(result, expected_result)

    def test_bounds_on_beta(self):
        my_biogeme = self.get_biogeme_instance()
        result = my_biogeme.getBoundsOnBeta('beta2')
        expected_result = -3, 10
        self.assertTupleEqual(result, expected_result)
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.getBoundsOnBeta('wrong_name')

    def test_saveIterationsFileName(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.generate_html = False
        my_biogeme.generate_pickle = False
        my_biogeme.saveIterations = False
        my_biogeme.modelName = 'simpleExample'
        f = my_biogeme._saveIterationsFileName()
        self.assertEqual(f, '__simpleExample.iter')

    def test_generateDraws(self):
        the_data = getData(1)
        self.assertIsNone(the_data.theDraws)
        ell = bioDraws('test', 'NORMAL')
        with self.assertRaises(excep.BiogemeError):
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
        with self.assertRaises(excep.BiogemeError):
            b = bio.BIOGEME(getData(1), rv)

    def test_getBoundsOnBeta(self):
        my_biogeme = self.get_biogeme_instance()
        b = my_biogeme.getBoundsOnBeta('beta1')
        self.assertTupleEqual(b, (-3, 3))
        b = my_biogeme.getBoundsOnBeta('beta2')
        self.assertTupleEqual(b, (-3, 10))

    def test_calculateNullLoglikelihood(self):
        my_biogeme = self.get_biogeme_instance()
        null_ell = my_biogeme.calculateNullLoglikelihood({1: 1, 2: 1})
        self.assertAlmostEqual(null_ell, -3.4657359027997265, 2)
        null_ell_2 = my_biogeme.calculateNullLoglikelihood({1: 1, 2: 1, 3: 1})
        self.assertAlmostEqual(null_ell_2, -5.493061443340549, 2)

    def test_calculateInitLikelihood(self):
        my_biogeme = self.get_biogeme_instance()
        res = my_biogeme.calculateInitLikelihood()
        self.assertAlmostEqual(res, -22055, 5)

    def test_calculateLikelihood(self):
        my_biogeme = self.get_biogeme_instance()
        x = my_biogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        res = my_biogeme.calculateLikelihood(xplus, scaled=False)
        self.assertEqual(res, -49500)
        with self.assertRaises(ValueError):
            _ = my_biogeme.calculateLikelihood([1], scaled=False)

    def test_calculateLikelihoodAndDerivatives(self):
        my_biogeme = self.get_biogeme_instance()
        x = my_biogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        my_biogeme.save_iterations = True
        f, g, h, bhhh = my_biogeme.calculateLikelihoodAndDerivatives(
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
            _ = my_biogeme.calculateLikelihoodAndDerivatives([1], scaled=False)
        my_biogeme.database.data = my_biogeme.database.data[0:0]
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.calculateLikelihoodAndDerivatives(x, scaled=True)

    def test_likelihoodFiniteDifferenceHessian(self):
        my_biogeme = self.get_biogeme_instance()
        x = my_biogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        h = my_biogeme.likelihoodFiniteDifferenceHessian(xplus)
        h_true = [[-110, 0.0], [0.0, -11000]]
        for row, row_true in zip(h, h_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 3)

    def test_checkDerivatives(self):
        my_biogeme = self.get_biogeme_instance()
        _, _, _, gdiff, hdiff = my_biogeme.checkDerivatives([1, 10])
        gdiff_true = [-0.00032656, 0.00545755]
        hdiff_true = [[6.49379217e-09, 0.00000000e00], [0.00000000e00, -1.39698386e-06]]
        for col, col_true in zip(gdiff, gdiff_true):
            self.assertAlmostEqual(col, col_true, 3)
        for row, row_true in zip(hdiff, hdiff_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 3)

    def test_estimate(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.generate_html = False
        my_biogeme.generate_pickle = False
        my_biogeme.saveIterations = False
        my_biogeme.modelName = 'simpleExample'
        my_biogeme.numberOfThreads = 1
        my_biogeme.algorithm_name = 'simple_bounds'
        my_biogeme.bootstrap_samples = 10
        results = my_biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        panelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
        panelBiogeme.bootstrap_samples = 10
        results = panelBiogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)

        my_biogeme.algorithm_name = 'scipy'
        my_biogeme.saveIterations = True
        results = my_biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        # We try to recycle, while there is no pickle file yet.
        results = my_biogeme.estimate(recycle=True)
        # We estimate the model twice to generate two pickle files.
        my_biogeme.generate_pickle = True
        results = my_biogeme.estimate()
        results = my_biogeme.estimate()
        results = my_biogeme.estimate(recycle=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme.generate_pickle = False

        my_biogeme_without_bounds = self.get_biogeme_instance_without_bounds()
        my_biogeme_without_bounds.generate_html = False
        my_biogeme_without_bounds.generate_pickle = False
        my_biogeme_without_bounds.saveIterations = False

        my_biogeme_without_bounds.algorithm_name = 'TR-newton'
        results = my_biogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'TR-BFGS'
        results = my_biogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'LS-newton'
        results = my_biogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'LS-BFGS'
        results = my_biogeme_without_bounds.estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)

        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.estimate(algorithm='any_algo')
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.estimate(algoParameters='any_param')

        aBiogeme = bio.BIOGEME(getData(1), {'loglike': Numeric(0)})
        with self.assertRaises(excep.BiogemeError):
            _ = aBiogeme.estimate()

        aBiogeme.loglike = None
        with self.assertRaises(excep.BiogemeError):
            _ = aBiogeme.estimate()

    def test_quickEstimate(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.generate_html = False
        my_biogeme.generate_pickle = False
        my_biogeme.saveIterations = False
        my_biogeme.modelName = 'simpleExample'
        my_biogeme.numberOfThreads = 1
        my_biogeme.algorithm_name = 'simple_bounds'
        my_biogeme.bootstrap_samples = 10
        results = my_biogeme.quickEstimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        panelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
        panelBiogeme.bootstrap_samples = 10
        results = panelBiogeme.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)

        my_biogeme.algorithm_name = 'scipy'
        my_biogeme.saveIterations = True
        results = my_biogeme.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        # We try to recycle, while there is no pickle file yet.
        results = my_biogeme.quickEstimate(recycle=True)
        # We quickEstimate the model twice to generate two pickle files.
        my_biogeme.generate_pickle = True
        results = my_biogeme.quickEstimate()
        results = my_biogeme.quickEstimate()
        results = my_biogeme.quickEstimate(recycle=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme.generate_pickle = False

        my_biogeme_without_bounds = self.get_biogeme_instance_without_bounds()
        my_biogeme_without_bounds.generate_html = False
        my_biogeme_without_bounds.generate_pickle = False
        my_biogeme_without_bounds.saveIterations = False

        my_biogeme_without_bounds.algorithm_name = 'TR-newton'
        results = my_biogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'TR-BFGS'
        results = my_biogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'LS-newton'
        results = my_biogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'LS-BFGS'
        results = my_biogeme_without_bounds.quickEstimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)

        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.quickEstimate(algorithm='any_algo')
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.quickEstimate(algoParameters='any_param')

        aBiogeme = bio.BIOGEME(getData(1), {'loglike': Numeric(0)})
        with self.assertRaises(excep.BiogemeError):
            _ = aBiogeme.quickEstimate()

        aBiogeme.loglike = None
        with self.assertRaises(excep.BiogemeError):
            _ = aBiogeme.quickEstimate()

    def test_simulate(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.algorithm_name = 'simple_bounds'
        results = my_biogeme.estimate()
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.simulate(theBetaValues=None)

        s = my_biogeme.simulate(results.getBetaValues())
        self.assertAlmostEqual(s.loc[0, 'loglike'], 0, 3)

        the_betas = results.getBetaValues()
        the_betas['any_beta'] = 0.1
        s = my_biogeme.simulate(the_betas)
        self.assertAlmostEqual(s.loc[0, 'loglike'], 0, 3)

        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.simulate('wrong_object')

        myPanelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
        results = myPanelBiogeme.estimate()
        with self.assertRaises(excep.BiogemeError):
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
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.change_init_values({'beta2': -100, 'beta1': 3.14156})
        self.assertListEqual(my_biogeme.id_manager.free_betas_values, [3.14156, -100])

    def test_confidenceIntervals(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.bootstrap_samples = 100
        results = my_biogeme.estimate(run_bootstrap=True)
        drawsFromBetas = results.getBetasForSensitivityAnalysis(
            my_biogeme.id_manager.free_betas.names
        )
        s = my_biogeme.simulate(results.getBetaValues())
        left, right = my_biogeme.confidenceIntervals(drawsFromBetas)
        self.assertLessEqual(left.loc[0, 'loglike'], s.loc[0, 'loglike'])
        self.assertGreaterEqual(right.loc[0, 'loglike'], s.loc[0, 'loglike'])

    def test_validate(self):
        my_data = getData(1)
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.algorithm_name = 'simple_bounds'
        results = my_biogeme.estimate()
        validation_data = my_data.split(slices=2)
        validation_results = my_biogeme.validate(results, validation_data)
        self.assertAlmostEqual(validation_results[0]['Loglikelihood'].sum(), 0, 3)
        self.assertAlmostEqual(validation_results[1]['Loglikelihood'].sum(), 0, 3)

        b = self.get_panel_instance()
        results = b.estimate()
        panel_data = getPanelData(1)
        validation_data = panel_data.split(slices=2)
        with self.assertRaises(excep.BiogemeError):
            validation_results = b.validate(results, validation_data)

    def test_optimize(self):
        # Here, we test only the special cases, as it has been called
        # several times by estimate
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.algorithm_name = 'simple_bounds'
        my_biogeme.optimize()
        my_biogeme._algorithm = None
        with self.assertRaises(excep.BiogemeError):
            my_biogeme.optimize()

    def test_print(self):
        my_biogeme = self.get_biogeme_instance()
        result = str(my_biogeme)[0:20]
        expected_result = 'biogemeModelDefaultN'
        self.assertEqual(result, expected_result)

    def test_files(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.algorithm_name = 'simple_bounds'
        my_biogeme.modelName = 'name_for_file'
        my_biogeme.generate_html = True
        my_biogeme.estimate()
        result = my_biogeme.files_of_type('html', all_files=False)
        expected_result = ['name_for_file.html']
        self.assertListEqual(result, expected_result)
        result = my_biogeme.files_of_type('html', all_files=True)
        self.assertGreaterEqual(len(result), 1)

    def test_from_configuration(self):
        beta_1 = Beta('beta_1', 0, None, None, 0)
        beta_2 = Beta('beta_2', 0, None, None, 0)
        catalog = Catalog.from_dict(
            catalog_name='the_catalog',
            dict_of_expressions={
                'beta_1': beta_1,
                'beta_2': beta_2,
            },
        )

        # Valid constructor
        config_id = 'the_catalog:beta_2'
        the_biogeme = bio.BIOGEME.from_configuration(
            config_id=config_id,
            expression=catalog,
            database=getData(1),
        )
        self.assertIs(the_biogeme.loglike, catalog)

        # Invalid constructor: typo
        config_id = 'the_catalog:bta_2'
        with self.assertRaises(excep.BiogemeError):
            _ = bio.BIOGEME.from_configuration(
                config_id=config_id,
                expression=catalog,
                database=getData(1),
            )

        # Invalid constructor: wrong id
        config_id = 'wrong_id'
        with self.assertRaises(excep.BiogemeError):
            _ = bio.BIOGEME.from_configuration(
                config_id=config_id,
                expression=catalog,
                database=getData(1),
            )


if __name__ == '__main__':
    unittest.main()
