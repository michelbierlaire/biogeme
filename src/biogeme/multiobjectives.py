"""Implementation of some multi-objective functions.



:author: Michel Bierlaire
:date: Fri Jul 14 18:35:29 2023

A multi-objective function takes the estimation results, and returns
several indicators. The indicators should be such that the lower, the
better. If an indicator must be maximized, the opposite value should be
returned.

"""

import numpy as np

from biogeme.deprecated import deprecated
from biogeme.results_processing import EstimationResults


def loglikelihood_dimension(results: EstimationResults) -> list[float]:
    """Function returning the negative log likelihood and the number
    of parameters, designed for multi-objective optimization

    :param results: estimation results
    :type results: biogeme.results.bioResults
    """
    if results.raw_estimation_results is None:
        return [float(np.finfo(np.float32).max), float(np.finfo(np.float32).max)]
    return [-results.final_log_likelihood, results.number_of_parameters]


def aic_bic_dimension(results: EstimationResults) -> list[float]:
    """Function returning the AIC, BIC and the number
    of parameters, designed for multi-objective optimization

    :param results: estimation results
    :type results: biogeme.results.bioResults
    """
    if results.raw_estimation_results is None:
        return [
            float(np.finfo(np.float32).max),
            float(np.finfo(np.float32).max),
            float(np.finfo(np.float32).max),
        ]

    return [
        results.akaike_information_criterion,
        results.bayesian_information_criterion,
        results.number_of_parameters,
    ]


@deprecated(new_func=aic_bic_dimension)
def AIC_BIC_dimension(results: EstimationResults) -> list[float]:
    pass
