"""Defines the labels of the various dimensions involved in the MCMC.

Michel Bierlaire
Mon Nov 03 2025, 08:56:26
"""

from enum import StrEnum


class Dimension(StrEnum):
    """Enumeration of coordinate dimension labels for MCMC models.

    Values can be used directly as strings in PyMC/ArviZ model definitions, e.g.:
        dims=(Dimension.OBS, Dimension.ALT)
    """

    ALT = "alt"
    OBS = "obs"
