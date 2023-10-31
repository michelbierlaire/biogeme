"""
Test the specification module

:author: Michel Bierlaire
:date: Tue Jun 20 08:56:40 2023
"""

import unittest
from biogeme.specification import Specification
from unittest.mock import MagicMock, patch
import biogeme.exceptions as excep
from biogeme.validity import Validity
from biogeme.configuration import Configuration
from biogeme.parameters import biogeme_parameters
from spec_swissmetro import logprob
from swissmetro_data import database


class TestSpecification(unittest.TestCase):
    def setUp(self):
        self.expression = logprob
        Specification.expression = logprob
        Specification.database = database
        self.config_id_1 = 'ASC:no_seg'
        self.config_id_2 = 'ASC:MALE'

    def test_constructor(self):
        config = self.expression.current_configuration()
        the_specification = Specification(config)
        the_results = the_specification.get_results()
        expected_loglike = -5331.252006916163
        loglike = the_results.data.logLike
        self.assertAlmostEqual(loglike, expected_loglike, 2)

    def test_from_string_id(self):
        the_specification = Specification.from_string_id(self.config_id_2)
        the_results = the_specification.get_results()
        expected_loglike = -5187.983411661233
        loglike = the_results.data.logLike
        self.assertAlmostEqual(loglike, expected_loglike, 2)

    def test_default_specification(self):
        the_specification = Specification.default_specification()
        the_results = the_specification.get_results()
        expected_loglike = -5331.252006916163
        loglike = the_results.data.logLike
        self.assertAlmostEqual(loglike, expected_loglike, 2)

    def test_describe(self):
        the_specification = Specification.from_string_id(self.config_id_2)
        expected_result = 'Final log likelihood:		-5187.983'
        description = the_specification.describe()[102:134]
        self.assertEqual(description, expected_result)

    @patch('biogeme.biogeme.BIOGEME.from_configuration')
    def test_validity_for_excessive_parameters(self, mock_from_configuration):
        """We use a mock Biogeme object instead of the real one"""
        biogeme_parameters.set_value(name='maximum_number_parameters', value=10)

        mock_biogeme = MagicMock()
        mock_biogeme.number_unknown_parameters.return_value = 11
        mock_from_configuration.return_value = mock_biogeme
        config = MagicMock(spec=Configuration)
        the_specification = Specification(config)
        self.assertFalse(the_specification.validity.status)
        self.assertIn('Too many parameters', the_specification.validity.reason)

    @patch('biogeme.biogeme.BIOGEME.from_configuration')
    def test_validity_for_non_converged_algorithm(self, mock_from_configuration):
        """We use a mock Biogeme object instead of the real one"""
        mock_biogeme = MagicMock()
        mock_biogeme.number_unknown_parameters.return_value = 4
        mock_from_configuration.return_value = mock_biogeme

        mock_results = MagicMock()
        mock_results.algorithm_has_converged.return_value = False
        mock_biogeme.quickEstimate.return_value = mock_results

        config = MagicMock(spec=Configuration)

        the_specification = Specification(config)
        self.assertFalse(the_specification.validity.status)
        self.assertIn(
            'Optimization algorithm has not converged',
            the_specification.validity.reason,
        )

    @patch('biogeme.biogeme.BIOGEME.from_configuration')
    def test_user_defined_validity_check(self, mock_from_configuration):
        mock_biogeme = MagicMock()
        mock_biogeme.number_unknown_parameters.return_value = 4
        mock_from_configuration.return_value = mock_biogeme

        def mock_validity_check(results):
            return Validity(status=False, reason='Mock validity failure')

        Specification.user_defined_validity_check = staticmethod(mock_validity_check)
        config = MagicMock(spec=Configuration)
        the_specification = Specification(config)
        self.assertFalse(the_specification.validity.status)
        self.assertIn('Mock validity failure', the_specification.validity.reason)

    def test_invalid_configuration(self):
        with self.assertRaises(excep.BiogemeError):
            Specification("Invalid Configuration")


if __name__ == '__main__':
    unittest.main()
