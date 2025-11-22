"""Definitions useful for Bayesian estimation

Michel Bierlaire
Sat Oct 18 2025, 16:03:38
"""

from typing import Protocol

import pandas as pd
import pytensor.tensor as pt

try:
    from enum import StrEnum  # Python 3.11+
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal backport of Python 3.11 StrEnum."""

        pass


class PymcModelBuilderType(Protocol):
    def __call__(self, dataframe: pd.DataFrame) -> pt.TensorVariable: ...


class Dimension(StrEnum):
    """Enumeration of coordinate dimension labels for MCMC models.

    Values can be used directly as strings in PyMC/ArviZ model definitions, e.g.:
        dims=(Dimension.OBS, Dimension.ALT)
    """

    ALT = "alt"
    OBS = "obs"
    INDIVIDUALS = "individuals"
