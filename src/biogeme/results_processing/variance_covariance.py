"""
Possible approximations of the variance-covariance matrix
Michel Bierlaire
Mon Jun 23 2025, 17:21:45
"""

from enum import Enum


class EstimateVarianceCovariance(str, Enum):
    """Identifies the estimate of the variance-covariance matrix to be used."""

    RAO_CRAMER = "Rao-Cramer"
    ROBUST = "Robust"
    BOOTSTRAP = "Bootstrap"
    BHHH = "BHHH"

    def __str__(self) -> str:
        return self.value
