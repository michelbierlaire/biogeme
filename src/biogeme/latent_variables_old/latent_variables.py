"""Latent-variable specifications for Biogeme latent-variable models.

This module defines lightweight data structures used to *specify* latent
variables (what depends on what).

Identification constraints (normalizations) and build-time concerns (draw types,
parameter fixings, etc.) are handled outside of this module by:
- a normalization plan (set of constraints/fixings), and
- a builder that takes (spec, plan) and produces Biogeme expressions.

Michel Bierlaire
:Wed Mar 04 2026, 11:36:27
"""

from collections.abc import Iterable
from dataclasses import dataclass

from .structural_equation import StructuralEquation


@dataclass
class LatentVariable:
    """Define a latent variable with its structural equation and metadata.

    The structural equation is provided through a
    :class:`~biogeme.latent_variables.structural_equation.StructuralEquation`.

    :param name:
        Name of the latent variable.
    :param structural_equation:
        Structural equation specification (deterministic part and stochastic
        error term definition).
    :param indicators:
        Collection of indicator names linked to this latent variable.
    """

    name: str
    structural_equation: StructuralEquation
    indicators: Iterable[str]
