""" Defines the types of the elementary expressions

:author: Michel Bierlaire
:date: Tue Mar  7 18:38:21 2023

"""
from enum import Enum, auto


class TypeOfElementaryExpression(Enum):
    """
    Defines the types of elementary expressions
    """

    VARIABLE = auto()
    BETA = auto()
    FREE_BETA = auto()
    FIXED_BETA = auto()
    RANDOM_VARIABLE = auto()
    DRAWS = auto()
