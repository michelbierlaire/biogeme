"""Structural equation specifications for latent variables.

This module defines :class:`StructuralEquation`, a *pure specification* object
used to describe the structural equation of a latent variable (what depends on
what).

It intentionally contains **no** Biogeme expression-building logic.
In particular, it does not:

- create parameters (e.g., ``Beta``),
- create simulation draws,
- apply positivity constraints,
- apply normalization / identification constraints.

All build-time concerns are handled by the builders package, which takes:

- a structural-equation specification,
- a build context (naming policy, estimation mode, draw types, factories), and
- an optional normalization plan,

and produces the corresponding Biogeme expressions.

See for example: ``biogeme.latent_variables.builders.structural``.

Michel Bierlaire
Thu Mar 05 2026, 11:26:07
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuralEquation:
    """Pure specification of a latent-variable structural equation.

    The structural equation specifies the *deterministic* covariates entering a
    latent variable. Any stochastic term, scaling (sigma), draw type, and
    identification constraints are build-time concerns and must be handled by
    the builders.

    Parameters
    ----------
    name:
        Name of the latent variable associated with this structural equation.
        It is expected to match the owning :class:`~biogeme.latent_variables.latent_variables.LatentVariable`.
    explanatory_variables:
        Iterable of variable names entering the deterministic part of the
        structural equation.
    """

    name: str
    explanatory_variables: Iterable[str]
