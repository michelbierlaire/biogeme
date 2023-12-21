""" Function to check and process numeric expressions

:author: Michel Bierlaire
:date: Sat Sep  9 15:27:17 2023
"""
import logging
from typing import Optional, Any
import numpy as np
from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)

MAX_VALUE = np.sqrt(np.finfo(float).max)
EPSILON = np.sqrt(np.finfo(float).eps)


def is_numeric(obj):
    """Checks if an object is numeric
    :param obj: obj to be checked
    :type obj: Object

    """
    return isinstance(obj, (int, float, bool))


def validate(value: float, modify: bool = True) -> float:
    """Check if the value is valid, in the sense that its absolute
    value is lower than the square root of the maximum floating point

    :param value: value to validate
    :param modify: if True, a valid value is generated. If False, an exception is triggered.
    """
    if value > MAX_VALUE:
        if modify:
            return MAX_VALUE
        raise BiogemeError(f'Value {value} is invalid. It cannot exceed {MAX_VALUE}')
    if value < -MAX_VALUE:
        if modify:
            return -MAX_VALUE
        raise BiogemeError(
            f'Value {value} is invalid. It cannot be lower than {-MAX_VALUE}'
        )
    return value
