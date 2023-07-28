"""Implementation of some multi-objective functions.



:author: Michel Bierlaire
:date: Fri Jul 14 18:35:29 2023

A multi-objective function takes the estimation results, and returns
several indicators. The indicators should be such that the lower, the
better. If an indicator must be maximized, the oppoite value should be
returned.

"""


def loglikelihood_dimension(results):
    """Function returning the negative log likelihood and the number
    of parameters, designed for multi-objective optimization

    :param results: estimation results
    :type results: biogeme.results.bioResults
    """
    return [-results.data.logLike, results.data.nparam]


def AIC_BIC_dimension(results):
    """Function returning the AIC, BIC and the number
    of parameters, designed for multi-objective optimization

    :param results: estimation results
    :type results: biogeme.results.bioResults
    """
    return [results.data.akaike, results.data.bayesian, results.data.nparam]
