"""Specifies the way second derivatives have to be calculated

Michel Bierlaire
Tue Jun 24 2025, 15:16:20
"""

from enum import Enum


class SecondDerivativesMode(Enum):
    ANALYTICAL = "analytical"
    FINITE_DIFFERENCES = "finite_differences"
    NEVER = "never"
