"""Functions to verify the validity of parameters

:author: Michel Bierlaire
:date: Thu Dec  1 16:22:34 2022

"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import biogeme.optimization as opt
from biogeme.second_derivatives import SecondDerivativesMode

if TYPE_CHECKING:
    from biogeme.default_parameters import ParameterValue


def is_number(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is a number

    :param x: value of the parameter to check
    """
    if isinstance(x, numbers.Number):
        return True, None
    return False, 'Value must be a number'


def zero_one(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is between zero and one

    :param x: value of the parameter to check

    """
    check, msg = is_number(x)
    if not check:
        return check, msg
    if 0 <= x <= 1:
        return True, None
    return False, 'Value must be between zero and one'


def is_positive(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is positive

    :param x: value of the parameter to check
    """
    check, msg = is_number(x)
    if not check:
        return check, msg

    if x > 0:
        return True, None
    return False, 'Value must be positive'


def is_non_negative(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is non_negative

    :param x: value of the parameter to check

    """
    if x >= 0:
        return True, None
    return False, 'Value must be non negative'


def is_integer(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is integer

    :param x: value of the parameter to check
    """
    if isinstance(x, numbers.Integral):
        return True, None
    return False, 'Value must be an integer'


def check_algo_name(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is a valid algorithm name

    :param x: value of the parameter to check
    """
    if not isinstance(x, str):
        return False, f'Parameter must be a string: {x}'
    possibilities = ['automatic'] + list(opt.algorithms.keys())
    if x in possibilities:
        return True, None
    return False, f'Value must be in: {possibilities}'


def check_sampling_strategy(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is a valid strategy name

    :param x: value of the parameter to check
    """
    from biogeme.bayesian_estimation import SAMPLER_STRATEGIES_DESCRIPTION

    if not isinstance(x, str):
        return False, f'Parameter must be a string: {x}'
    possibilities = ['automatic'] + list(SAMPLER_STRATEGIES_DESCRIPTION.keys())
    if x in possibilities:
        return True, None
    return False, f'Value must be in: {possibilities}'


def check_calculating_second_derivatives(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is a valid way to calculate the second derivatives

    :param x: value of the parameter to check
    """
    if not isinstance(x, str):
        return False, f'Parameter must be a string: {x}'
    possibilities = [value.value for value in SecondDerivativesMode]
    if x in possibilities:
        return True, None
    return False, f'Value must be in: {possibilities}'


def is_boolean(x: ParameterValue) -> tuple[bool, str | None]:
    """Return true if x is a boolean

    :param x: value of the parameter to check
    :type x: float
    """
    if isinstance(x, bool):
        return True, None
    return False, 'Value must be boolean'
