"""
Test the biogeme module

:author: Michel Bierlaire
:date: Wed Apr 29 18:32:42 2020

"""

import biogeme.biogeme_logging as blog

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
from biogeme.function_output import FunctionOutput
from test_data import getData, getPanelData

logger = blog.get_screen_logger(level=blog.INFO)


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
            'log_like': likelihood,
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
            'log_like': likelihood,
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
            'log_like': likelihood,
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
                {'log_like': 'wrong_object'},
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

        b3 = bio.BIOGEME(
            getData(1), self.get_dict_of_expressions(), parameter_file='myfile.toml'
        )
        self.assertEqual(b3.biogeme_parameters.file_name, 'myfile.toml')

        b4 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), missingData=12)
        self.assertEqual(b4.missing_data, 12)

        b5 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), suggestScales=True)

        b6 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), userNotes='blabla')
        self.assertEqual(b6.user_notes, 'blabla')

        b7 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), generateHtml=False)
        self.assertEqual(b7.generate_html, False)

        b8 = bio.BIOGEME(
            getData(1), self.get_dict_of_expressions(), saveIterations=True
        )
        self.assertEqual(b8.save_iterations, True)

        b9 = bio.BIOGEME(getData(1), self.get_dict_of_expressions(), seed_param=12)
        self.assertEqual(b9.seed, 12)

    def test_saved_iterations(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme._load_saved_iteration()
        # Remove the file and try to loaed it again
        my_biogeme._load_saved_iteration()

    def test_random_init_values(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.set_random_init_values(default_bound=10)
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
        result = my_biogeme.free_beta_names
        expected_result = ['beta1', 'beta2']
        self.assertListEqual(result, expected_result)

    def test_get_beta_values(self):
        my_biogeme = self.get_biogeme_instance()
        result = my_biogeme.get_beta_values()
        expected_result = {'beta1': -1.0, 'beta2': 2.0}
        self.assertDictEqual(result, expected_result)

    def test_bounds_on_beta(self):
        my_biogeme = self.get_biogeme_instance()
        result = my_biogeme.get_bounds_on_beta('beta2')
        expected_result = -3, 10
        self.assertTupleEqual(result, expected_result)
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.get_bounds_on_beta('wrong_name')

    def test_saveIterationsFileName(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.generate_html = False
        my_biogeme.generate_pickle = False
        my_biogeme.saveIterations = False
        my_biogeme.modelName = 'simpleExample'
        f = my_biogeme._save_iterations_file_name()
        self.assertEqual(f, '__simpleExample.iter')

    def test_generateDraws(self):
        the_data = getData(1)
        self.assertIsNone(the_data.theDraws)
        ell = bioDraws('test', 'NORMAL')
        with self.assertRaises(excep.BiogemeError):
            b = bio.BIOGEME(the_data, ell)
        b = bio.BIOGEME(the_data, ell, skip_audit=True)
        b._generate_draws(10)
        self.assertTupleEqual(the_data.theDraws.shape, (5, 10, 1))
        ell2 = bioDraws('test', 'NORMAL') + bioDraws('test2', 'UNIFORM')
        b2 = bio.BIOGEME(the_data, ell2, skip_audit=True)
        b2._generate_draws(20)
        self.assertTupleEqual(the_data.theDraws.shape, (5, 20, 2))

    def test_random_variable(self):
        rv = RandomVariable('omega')
        with self.assertRaises(excep.BiogemeError):
            b = bio.BIOGEME(getData(1), rv)

    def test_getBoundsOnBeta(self):
        my_biogeme = self.get_biogeme_instance()
        b = my_biogeme.get_bounds_on_beta('beta1')
        self.assertTupleEqual(b, (-3, 3))
        b = my_biogeme.get_bounds_on_beta('beta2')
        self.assertTupleEqual(b, (-3, 10))

    def test_calculateNullLoglikelihood(self):
        my_biogeme = self.get_biogeme_instance()
        null_ell = my_biogeme.calculate_null_loglikelihood({1: 1, 2: 1})
        self.assertAlmostEqual(null_ell, -3.4657359027997265, 2)
        null_ell_2 = my_biogeme.calculate_null_loglikelihood({1: 1, 2: 1, 3: 1})
        self.assertAlmostEqual(null_ell_2, -5.493061443340549, 2)

    def test_calculateInitLikelihood(self):
        my_biogeme = self.get_biogeme_instance()
        res = my_biogeme.calculate_init_likelihood()
        self.assertAlmostEqual(res, -22055, 5)

    def test_calculateLikelihood(self):
        my_biogeme = self.get_biogeme_instance()
        x = my_biogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        res = my_biogeme.calculate_likelihood(xplus, scaled=False)
        self.assertEqual(res, -49500)
        with self.assertRaises(ValueError):
            _ = my_biogeme.calculate_likelihood([1], scaled=False)

    def test_calculateLikelihoodAndDerivatives(self):
        my_biogeme = self.get_biogeme_instance()
        x = my_biogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        my_biogeme.save_iterations = True
        the_function_output: FunctionOutput = (
            my_biogeme.calculate_likelihood_and_derivatives(
                xplus, scaled=False, hessian=True, bhhh=True
            )
        )
        f_true = -49500
        g_true = [0.0, -33000.0]
        h_true = [[-110.0, 0.0], [0.0, -11000.0]]
        bhhh_true = [[0.0, 0.0], [0.0, 352440000.0]]
        self.assertEqual(f_true, the_function_output.function)
        self.assertListEqual(g_true, the_function_output.gradient.tolist())
        self.assertListEqual(h_true, the_function_output.hessian.tolist())
        self.assertListEqual(bhhh_true, the_function_output.bhhh.tolist())
        with self.assertRaises(ValueError):
            _ = my_biogeme.calculate_likelihood_and_derivatives([1], scaled=False)
        my_biogeme.database.data = my_biogeme.database.data[0:0]
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.calculate_likelihood_and_derivatives(x, scaled=True)

    def test_likelihoodFiniteDifferenceHessian(self):
        my_biogeme = self.get_biogeme_instance()
        x = my_biogeme.id_manager.free_betas_values
        xplus = [v + 1 for v in x]
        h = my_biogeme.likelihood_finite_difference_hessian(xplus)
        h_true = [[-110, 0.0], [0.0, -11000]]
        for row, row_true in zip(h, h_true):
            for col, col_true in zip(row, row_true):
                self.assertAlmostEqual(col, col_true, 3)

    def test_checkDerivatives(self):
        my_biogeme = self.get_biogeme_instance()
        _, _, _, gdiff, hdiff = my_biogeme.check_derivatives([1, 10])
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
        # We estimate the model twice to generate two pickle .py.
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

        aBiogeme = bio.BIOGEME(getData(1), {'log_like': Numeric(0)})
        with self.assertRaises(excep.BiogemeError):
            _ = aBiogeme.estimate()

        aBiogeme.log_like = None
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
        results = my_biogeme.quick_estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        panelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
        panelBiogeme.bootstrap_samples = 10
        results = panelBiogeme.quick_estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)

        my_biogeme.algorithm_name = 'scipy'
        my_biogeme.saveIterations = True
        results = my_biogeme.quick_estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        # We try to recycle, while there is no pickle file yet.
        results = my_biogeme.quick_estimate(recycle=True)
        # We quickEstimate the model twice to generate two pickle .py.
        my_biogeme.generate_pickle = True
        results = my_biogeme.quick_estimate()
        results = my_biogeme.quick_estimate()
        results = my_biogeme.quick_estimate(recycle=True)
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme.generate_pickle = False

        my_biogeme_without_bounds = self.get_biogeme_instance_without_bounds()
        my_biogeme_without_bounds.generate_html = False
        my_biogeme_without_bounds.generate_pickle = False
        my_biogeme_without_bounds.saveIterations = False

        my_biogeme_without_bounds.algorithm_name = 'TR-newton'
        results = my_biogeme_without_bounds.quick_estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'TR-BFGS'
        results = my_biogeme_without_bounds.quick_estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'LS-newton'
        results = my_biogeme_without_bounds.quick_estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)
        my_biogeme_without_bounds.algorithm_name = 'LS-BFGS'
        results = my_biogeme_without_bounds.quick_estimate()
        self.assertAlmostEqual(results.data.logLike, 0, 5)

        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.quick_estimate(algorithm='any_algo')
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.quick_estimate(algoParameters='any_param')

        aBiogeme = bio.BIOGEME(getData(1), {'log_like': Numeric(0)})
        with self.assertRaises(excep.BiogemeError):
            _ = aBiogeme.quick_estimate()

        aBiogeme.log_like = None
        with self.assertRaises(excep.BiogemeError):
            _ = aBiogeme.quick_estimate()

    def test_simulate(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.algorithm_name = 'simple_bounds'
        results = my_biogeme.estimate()
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.simulate(the_beta_values=None)
        my_biosim = self.get_biogeme_instance()
        s = my_biosim.simulate(results.get_beta_values())
        self.assertAlmostEqual(s.loc[0, 'log_like'], 0, 3)

        the_betas = results.get_beta_values()
        the_betas['any_beta'] = 0.1
        s = my_biosim.simulate(the_betas)
        self.assertAlmostEqual(s.loc[0, 'log_like'], 0, 3)

        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.simulate('wrong_object')

        myPanelBiogeme = bio.BIOGEME(getPanelData(1), self.get_dict_of_expressions())
        results = myPanelBiogeme.estimate()
        with self.assertRaises(excep.BiogemeError):
            s = myPanelBiogeme.simulate(results.get_beta_values())

        myPanelBiogeme = bio.BIOGEME(
            getPanelData(1),
            {
                'Simul': MonteCarlo(
                    PanelLikelihoodTrajectory(bioDraws('test', 'NORMAL'))
                )
            },
        )
        s = myPanelBiogeme.simulate(results.get_beta_values())

    def test_changeInitValues(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.change_init_values({'beta2': -100, 'beta1': 3.14156})
        self.assertListEqual(my_biogeme.id_manager.free_betas_values, [3.14156, -100])

    def test_confidenceIntervals(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.bootstrap_samples = 100
        results = my_biogeme.estimate(run_bootstrap=True)
        draws_from_betas = results.get_betas_for_sensitivity_analysis(
            my_biogeme.id_manager.free_betas.names
        )
        my_biosim = self.get_biogeme_instance()
        s = my_biosim.simulate(results.get_beta_values())
        left, right = my_biosim.confidence_intervals(draws_from_betas)
        self.assertLessEqual(left.loc[0, 'log_like'], s.loc[0, 'log_like'])
        self.assertGreaterEqual(right.loc[0, 'log_like'], s.loc[0, 'log_like'])

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
        expected_result = 'name_for_file.html'
        self.assertIn(expected_result, result, f"{expected_result} is not in the list")
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

        self.assertIs(the_biogeme.log_like, catalog)

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

    def test_report_array(self):
        my_biogeme = self.get_biogeme_instance()
        array = np.array([12, 45])
        result = my_biogeme.report_array(array)
        expected_result = 'beta1=12, beta2=45'
        self.assertEqual(result, expected_result)
        result = my_biogeme.report_array(array, with_names=False)
        expected_result = '12, 45'
        self.assertEqual(result, expected_result)

    def test_beta_values_dict_to_list(self):
        my_biogeme = self.get_biogeme_instance()
        result = my_biogeme.beta_values_dict_to_list()
        expected_result = [-1.0, 2.0]
        self.assertEqual(expected_result, result)
        beta_dict = {'beta1': 1.0, 'beta2': 2.0}
        result = my_biogeme.beta_values_dict_to_list(beta_dict=beta_dict)
        expected_result = [1.0, 2.0]
        self.assertEqual(expected_result, result)
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.beta_values_dict_to_list(beta_dict=3)
        beta_dict = {'beta1': 1.0}
        with self.assertRaises(excep.BiogemeError):
            _ = my_biogeme.beta_values_dict_to_list(beta_dict=beta_dict)


if __name__ == '__main__':

    unittest.main()
