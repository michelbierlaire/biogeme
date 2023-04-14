"""Defines a generic exception for Biogeme

:author: Michel Bierlaire

:date: Tue Mar 26 16:47:11 2019

"""

# Too constraining
# pylint: disable=invalid-name


class BiogemeError(Exception):
    """Defines a generic exception for Biogeme."""


class ValueOutOfRange(BiogemeError):
    """Value of out range error."""


class DuplicateError(BiogemeError):
    """duplicate error."""
