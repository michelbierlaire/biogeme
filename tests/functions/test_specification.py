"""
Test the specification module

:author: Michel Bierlaire
:date: Tue Jun 20 08:56:40 2023
"""

import unittest
from biogeme.specification import Specification
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


if __name__ == '__main__':
    unittest.main()
