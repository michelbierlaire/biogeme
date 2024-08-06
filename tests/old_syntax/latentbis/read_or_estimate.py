"""

Read or estimate
================

Function to estimate the parameters, or read them from a file, if available.

:author: Michel Bierlaire, EPFL
:date: Sat Dec  9 17:20:49 2023
"""
import logging
from typing import Optional
from biogeme.biogeme import BIOGEME
from biogeme.results import bioResults
import biogeme.exceptions as excep

logger = logging.getLogger(__name__)


def read_or_estimate(
    the_biogeme: BIOGEME, directory: Optional[str] = '.'
) -> bioResults:
    """
    Function to estimate the parameters, or read them from a file, if available.

    :param the_biogeme: Biogeme object.
    :param directory: directory where the pickle file is supposed to be.

    :return: estimation results.
    """
    try:
        filename = f'{directory}/{the_biogeme.modelName}.pickle'
        logger.info('Results are read from the file {filename}.')
        results = bioResults(pickleFile=filename)
    except excep.FileNotFoundError:
        logger.info('Parameters are estimated.')
        results = the_biogeme.estimate()

    return results
