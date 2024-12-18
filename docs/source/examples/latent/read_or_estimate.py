"""

Read of estimate
================

Function to estimate the parameters, or read them from a file, if available.

:author: Michel Bierlaire, EPFL
:date: Sat Dec  9 17:20:49 2023
"""

import logging
from typing import Optional
from biogeme.biogeme import BIOGEME
from biogeme.results_processing import EstimationResults

logger = logging.getLogger(__name__)


def read_or_estimate(
    the_biogeme: BIOGEME, directory: Optional[str] = '.'
) -> EstimationResults:
    """
    Function to estimate the parameters, or read them from a file, if available.

    :param the_biogeme: Biogeme object.
    :param directory: directory where the pickle file is supposed to be.

    :return: estimation results.
    """
    try:
        filename = f'{directory}/{the_biogeme.modelName}.yaml'
        logger.info('Results are read from the file {filename}.')
        results = EstimationResults.from_yaml_file(filename=filename)
    except FileNotFoundError:
        logger.info('Parameters are estimated.')
        results = the_biogeme.estimate()

    return results
