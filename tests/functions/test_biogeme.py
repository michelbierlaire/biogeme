"""
Test the biogeme module

:author: Michel Bierlaire
:date: Wed Apr 29 18:32:42 2020

"""

import os
import random as rnd
import unittest

import biogeme.biogeme_logging as blog
import numpy as np
from biogeme.biogeme import BIOGEME
from biogeme.catalog import Catalog
from biogeme.exceptions import BiogemeError
from biogeme.expressions import (
    Beta,
    Numeric,
    PanelLikelihoodTrajectory,
    Variable,
    exp,
)
from biogeme.second_derivatives import SecondDerivativesMode
from biogeme.tools.files import files_of_type
from biogeme.validation import ValidationResult

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
                or file_name.endswith('.yaml')
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

    def get_dict_of_expressions_for_panel(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        likelihood = exp(-((beta1 * Variable1) ** 2) - (beta2 * Variable2) ** 2)
        simul = (beta1 + 2 * beta2) / Variable1 + (beta2 + 2 * beta1) / Variable2
        dict_of_expressions = {
            'log_like': PanelLikelihoodTrajectory(likelihood),
            'weight': Numeric(1),
            'beta1': beta1,
            'simul': PanelLikelihoodTrajectory(simul),
        }
        return dict_of_expressions

    def get_biogeme_instance(self):
        data = getData(1)
        return BIOGEME(data, self.get_dict_of_expressions())

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
        return BIOGEME(data, dict_of_expressions)

    def get_panel_instance(self):
        Variable1 = Variable('Variable1')
        Variable2 = Variable('Variable2')
        beta1 = Beta('beta1', -1.0, -3, 3, 0)
        beta2 = Beta('beta2', 2.0, -3, 10, 0)
        likelihood = -((beta1 * Variable1) ** 2) - (beta2 * Variable2) ** 2
        simul = (beta1 + 2 * beta2) / Variable1 + (beta2 + 2 * beta1) / Variable2
        dict_of_expressions = {
            'log_like': PanelLikelihoodTrajectory(likelihood),
            'weight': Numeric(1),
            'beta1': beta1,
            'simul': PanelLikelihoodTrajectory(simul),
        }
        data = getData(1)
        data.panel('Person')
        return BIOGEME(data, dict_of_expressions)

    def test_ctor(self):
        # Test obsolete parameters
        aBiogeme = BIOGEME(
            getData(1),
            self.get_dict_of_expressions(),
        )

        with self.assertRaises(BiogemeError):
            bBiogeme = BIOGEME(
                getData(1),
                'wrong_object',
            )

        with self.assertRaises(BiogemeError):
            bBiogeme = BIOGEME(
                getData(1),
                {'log_like': 'wrong_object'},
            )

        cBiogeme = BIOGEME(
            getPanelData(1),
            self.get_dict_of_expressions_for_panel(),
        )

    def test_saved_iterations(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme._load_saved_iteration()
        # Remove the file and try to loaed it again
        my_biogeme._load_saved_iteration()

    def test_random_init_values(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.set_random_init_values(default_bound=10)
        for (
            v
        ) in (
            my_biogeme.model_elements.expressions_registry.list_of_free_betas_init_values
        ):
            self.assertLessEqual(v, 10)
            self.assertGreaterEqual(v, -10)

    def test_free_beta_names(self):
        my_biogeme = self.get_biogeme_instance()
        result = my_biogeme.free_betas_names
        expected_result = ['beta1', 'beta2']
        self.assertListEqual(result, expected_result)

    def test_saveIterationsFileName(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.model_name = 'simpleExample'
        f = my_biogeme._save_iterations_file_name()
        self.assertEqual(f, '__simpleExample.iter')

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

    def test_estimate(self):
        data = getData(1)
        my_biogeme = BIOGEME(data, self.get_dict_of_expressions(), bootstrap_samples=10)
        results = my_biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_estimate_panel(self):
        likelihood = self.get_dict_of_expressions_for_panel()
        panel_biogeme = BIOGEME(
            getPanelData(1),
            likelihood,
            bootstrap_samples=10,
            save_iterations=False,
        )
        results = panel_biogeme.estimate(run_bootstrap=True)
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_estimate_scipy(self):
        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            bootstrap_samples=10,
            optimization_algorithm='scipy',
            save_iterations=True,
            generate_yaml=True,
        )
        my_biogeme.model_name = 'test_scipy'
        results = my_biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)
        # We try to recycle, while there is no yaml file yet.
        results = my_biogeme.estimate(recycle=True)
        # We estimate the model twice to generate two pickle .py.
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_estimate_tr_newton(self):
        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            bootstrap_samples=10,
            optimization_algorithm='TR-newton',
            save_iterations=False,
            generate_yaml=False,
        )
        results = my_biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_estimate_tr_bfgs(self):
        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            bootstrap_samples=10,
            optimization_algorithm='TR-BFGS',
            save_iterations=False,
            generate_yaml=False,
        )
        results = my_biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_estimate_ls_newton(self):
        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            bootstrap_samples=10,
            optimization_algorithm='LS-newton',
            save_iterations=False,
            generate_yaml=False,
        )
        results = my_biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_estimate_ls_bfgs(self):

        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            bootstrap_samples=10,
            optimization_algorithm='LS-BFGS',
            save_iterations=False,
            generate_yaml=False,
        )
        results = my_biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_algo_error(self):
        data = getData(1)
        with self.assertRaises(BiogemeError):
            _ = BIOGEME(
                data,
                self.get_dict_of_expressions(),
                bootstrap_samples=10,
                optimization_algorithm='any_algo',
                save_iterations=False,
                generate_yaml=False,
            )

        with self.assertRaises(BiogemeError):
            _ = BIOGEME(
                data,
                self.get_dict_of_expressions(),
                bootstrap_samples=10,
                wrong_parameter_name='any_algo',
                save_iterations=False,
                generate_yaml=False,
            )

    def test_trivial_loglikelihood(self):
        aBiogeme = BIOGEME(getData(1), {'log_like': Numeric(0)})
        with self.assertRaises(BiogemeError):
            _ = aBiogeme.estimate()

    def test_quickEstimate(self):
        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            save_iterations=False,
            generate_yaml=False,
            generate_html=False,
        )
        my_biogeme.model_name = 'simple_example'
        results = my_biogeme.quick_estimate()
        self.assertAlmostEqual(results.final_log_likelihood, 0, 5)

    def test_simulate(self):
        my_biogeme = self.get_biogeme_instance()
        results = my_biogeme.estimate()
        with self.assertRaises(BiogemeError):
            _ = my_biogeme.simulate(the_beta_values=None)
        my_biosim = self.get_biogeme_instance()
        s = my_biosim.simulate(results.get_beta_values())
        self.assertAlmostEqual(s.loc[0, 'log_like'], 0, 3)

        the_betas = results.get_beta_values()
        the_betas['any_beta'] = 0.1
        s = my_biosim.simulate(the_betas)
        self.assertAlmostEqual(s.loc[0, 'log_like'], 0, 3)

    def test_simulate_wrong(self):
        my_biogeme = self.get_biogeme_instance()
        with self.assertRaises(BiogemeError):
            _ = my_biogeme.simulate('wrong_object')

    def test_changeInitValues(self):
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.change_init_values({'beta2': -100, 'beta1': 3.14156})
        self.assertListEqual(
            my_biogeme.model_elements.expressions_registry.list_of_free_betas_init_values,
            [3.14156, -100],
        )

    def test_confidenceIntervals(self):
        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            save_iterations=False,
            generate_yaml=False,
            generate_html=False,
            bootstrap_samples=100,
        )
        results = my_biogeme.estimate(run_bootstrap=True)
        draws_from_betas = results.get_betas_for_sensitivity_analysis()
        my_biosim = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            save_iterations=False,
            generate_yaml=False,
            generate_html=False,
        )
        s = my_biosim.simulate(results.get_beta_values())
        left, right = my_biosim.confidence_intervals(draws_from_betas)
        self.assertLessEqual(left.loc[0, 'log_like'], s.loc[0, 'log_like'])
        self.assertGreaterEqual(right.loc[0, 'log_like'], s.loc[0, 'log_like'])

    def test_validate(self):
        my_data = getData(1)
        my_biogeme = self.get_biogeme_instance()
        my_biogeme.algorithm_name = 'simple_bounds'
        results = my_biogeme.estimate()
        validation_results: list[ValidationResult] = my_biogeme.validate(
            results, slices=5
        )
        self.assertAlmostEqual(
            validation_results[0]
            .simulated_values['log_like [validation fold 1]']
            .sum(),
            0,
            3,
        )
        self.assertAlmostEqual(
            validation_results[1]
            .simulated_values['log_like [validation fold 2]']
            .sum(),
            0,
            3,
        )

    def test_print(self):
        my_biogeme = self.get_biogeme_instance()
        result = str(my_biogeme)[0:20]
        expected_result = 'biogeme_model_defaul'
        self.assertEqual(result, expected_result)

    def test_files(self):
        data = getData(1)
        my_biogeme = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            save_iterations=False,
            generate_yaml=False,
            generate_html=True,
            bootstrap_samples=100,
        )
        my_biogeme.model_name = 'name_for_file'
        my_biogeme.estimate()
        result = files_of_type(
            name=my_biogeme.model_name, extension='html', all_files=False
        )
        expected_result = 'name_for_file.html'
        self.assertIn(expected_result, result, f"{expected_result} is not in the list")
        result = files_of_type('html', name=my_biogeme.model_name, all_files=True)
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
        the_biogeme = BIOGEME.from_configuration(
            config_id=config_id,
            multiple_expression=catalog,
            database=getData(1),
        )

        # Invalid constructor: typo
        config_id = 'the_catalog:bta_2'
        with self.assertRaises(BiogemeError):
            _ = BIOGEME.from_configuration(
                config_id=config_id,
                multiple_expression=catalog,
                database=getData(1),
            )

        # Invalid constructor: wrong id
        config_id = 'wrong_id'
        with self.assertRaises(BiogemeError):
            _ = BIOGEME.from_configuration(
                config_id=config_id,
                multiple_expression=catalog,
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

    def test_calculate_second_derivatives(self):

        data = getData(1)

        biogeme_0 = BIOGEME(data, self.get_dict_of_expressions())
        self.assertEqual(
            biogeme_0.second_derivatives_mode, SecondDerivativesMode.ANALYTICAL
        )

        biogeme_1 = BIOGEME(
            data, self.get_dict_of_expressions(), calculating_second_derivatives='never'
        )
        self.assertEqual(biogeme_1.second_derivatives_mode, SecondDerivativesMode.NEVER)

        biogeme_2 = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            calculating_second_derivatives='finite_differences',
        )
        self.assertEqual(
            biogeme_2.second_derivatives_mode, SecondDerivativesMode.FINITE_DIFFERENCES
        )

        biogeme_3 = BIOGEME(
            data,
            self.get_dict_of_expressions(),
            calculating_second_derivatives='analytical',
        )
        self.assertEqual(
            biogeme_3.second_derivatives_mode, SecondDerivativesMode.ANALYTICAL
        )

        with self.assertRaises(BiogemeError):
            _ = BIOGEME(
                data,
                self.get_dict_of_expressions(),
                calculating_second_derivatives='xxx',
            )


if __name__ == '__main__':

    unittest.main()
