import logging
import unittest

import numpy as np
from biogeme.multiobjectives import (
    AIC_BIC_dimension,
    aic_bic_dimension,
    loglikelihood_dimension,
)

logger = logging.getLogger('biogeme.deprecated')


class DummyResults:
    def __init__(
        self,
        final_log_likelihood=None,
        number_of_parameters=None,
        akaike_information_criterion=None,
        bayesian_information_criterion=None,
        raw_estimation_results=True,
    ):
        self.final_log_likelihood = final_log_likelihood
        self.number_of_parameters = number_of_parameters
        self.akaike_information_criterion = akaike_information_criterion
        self.bayesian_information_criterion = bayesian_information_criterion
        self.raw_estimation_results = raw_estimation_results


class TestMultiObjectives(unittest.TestCase):
    def test_loglikelihood_dimension_valid(self):
        res = DummyResults(final_log_likelihood=-123.45, number_of_parameters=4)
        result = loglikelihood_dimension(res)
        self.assertEqual(result, [123.45, 4])

    def test_loglikelihood_dimension_invalid(self):
        res = DummyResults(raw_estimation_results=None)
        result = loglikelihood_dimension(res)
        max_float = float(np.finfo(np.float32).max)
        self.assertEqual(result, [max_float, max_float])

    def test_aic_bic_dimension_valid(self):
        res = DummyResults(
            akaike_information_criterion=456.78,
            bayesian_information_criterion=478.90,
            number_of_parameters=5,
        )
        result = aic_bic_dimension(res)
        self.assertEqual(result, [456.78, 478.90, 5])

    def test_aic_bic_dimension_invalid(self):
        res = DummyResults(raw_estimation_results=None)
        result = aic_bic_dimension(res)
        max_float = float(np.finfo(np.float32).max)
        self.assertEqual(result, [max_float, max_float, max_float])

    def test_deprecated_AIC_BIC_dimension(self):
        res = DummyResults()
        with self.assertLogs(logger, level='WARNING') as cm:
            AIC_BIC_dimension(res)

            # exactly one WARNING; message contains both names
            self.assertEqual(len(cm.records), 1)
            self.assertEqual(cm.records[0].levelno, logging.WARNING)


if __name__ == '__main__':
    unittest.main()
