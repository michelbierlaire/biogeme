"""

Read of estimate
================

Function to estimate the parameters, or read them from a file, if available.

Michel Bierlaire, EPFL
Mon May 05 2025, 18:59:34
"""

import logging

from biogeme.biogeme import BIOGEME
from biogeme.results_processing import EstimationResults

logger = logging.getLogger(__name__)


def read_or_estimate(the_biogeme: BIOGEME, directory: str = '.') -> EstimationResults:
    """
    Function to estimate the parameters, or read them from a file, if available.

    :param the_biogeme: Biogeme object.
    :param directory: directory where the yaml file is supposed to be.

    :return: estimation results.
    """
    try:
        filename = f'{directory}/{the_biogeme.model_name}.yaml'
        logger.info('Results are read from the file {filename}.')
        results = EstimationResults.from_yaml_file(filename=filename)
    except FileNotFoundError:
        logger.info('Parameters are estimated.')
        results = the_biogeme.estimate()

    return results
