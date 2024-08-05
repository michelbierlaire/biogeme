""" Function to check and process numeric expressions

:author: Michel Bierlaire
:date: Sat Sep  9 15:27:17 2023
"""

import logging

import numpy as np

from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)

EPSILON = np.sqrt(np.finfo(float).eps)


def is_numeric(obj) -> bool:
    """Checks if an object is numeric
    :param obj: obj to be checked
    :type obj: Object

    """
    return isinstance(obj, (int, float, bool))
