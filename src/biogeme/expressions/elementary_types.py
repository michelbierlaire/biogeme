"""Arithmetic expressions: types of elementary expressions

:author: Michel Bierlaire
:date: Tue Mar  7 18:38:21 2023

"""

import logging
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
    BAYESIAN_DRAWS = auto()


logger = logging.getLogger(__name__)
