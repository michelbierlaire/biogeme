"""Defines exceptions for Biogeme

:author: Michel Bierlaire

:date: Fri Dec  8 20:34:09 2023

"""


class BiogemeError(Exception):
    """Defines a generic exception for Biogeme."""


class FileNotFound(BiogemeError):
    """Value of out range error."""


class ValueOutOfRange(BiogemeError):
    """Value of out range error."""


class DuplicateError(BiogemeError):
    """duplicate error."""


class NotImplementedError(BiogemeError):
    """Not implemented error."""
