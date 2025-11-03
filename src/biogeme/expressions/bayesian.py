"""Definitions useful for Bayesian estimation

Michel Bierlaire
Sat Oct 18 2025, 16:03:38
"""

from collections.abc import Callable
from enum import StrEnum, auto

import pandas as pd
import pytensor.tensor as pt

PymcModelBuilderType = Callable[[pd.DataFrame], pt.TensorVariable]


class Dimension(StrEnum):
    """Possible dimensions for model data in Biogeme."""

    OBSERVATIONS = auto()
    INDIVIDUALS = auto()
    ALTERNATIVES = auto()
