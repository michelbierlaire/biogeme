# normalization/parameter_refs.py
"""
Semantic references to model parameters that can be fixed by a normalization plan.

This module is part of the *normalization* layer. It contains **no Biogeme
expression building** logic and should remain independent of estimation mode.

A `ParameterRef` identifies a model parameter in a stable way, independent of
how it is ultimately represented (e.g., log-parameterizations, bounds, etc.).
Builders are responsible for mapping these references to Biogeme parameter names.

Michel Bierlaire
Wed Mar 04 2026, 16:30:17
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class ParameterRef(Protocol):
    """Protocol for semantic references to fixable model parameters."""

    def key(self) -> str:
        """Return a stable string key identifying the target parameter."""
        ...

    def __str__(self) -> str:  # pragma: no cover
        ...


@dataclass(frozen=True, slots=True)
class MeasurementIntercept:
    """Reference to a measurement intercept parameter for one indicator."""

    indicator: str

    def key(self) -> str:
        return f"measurement_intercept:{self.indicator}"

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True, slots=True)
class MeasurementLoading:
    """Reference to a measurement loading (LV coefficient) for one LV and indicator."""

    latent: str
    indicator: str

    def key(self) -> str:
        return f"measurement_loading:{self.latent}:{self.indicator}"

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True, slots=True)
class MeasurementSigma:
    """Reference to a measurement error scale (sigma) parameter for one indicator."""

    indicator: str

    def key(self) -> str:
        return f"measurement_sigma:{self.indicator}"

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True, slots=True)
class ThresholdFirst:
    """Reference to the first free threshold parameter (tau_1) for a threshold system.

    This target is meaningful only for non-symmetric (monotone) threshold systems
    that explicitly contain a free `tau_1` parameter in the builder.
    """

    type_name: str

    def key(self) -> str:
        return f"threshold_first:{self.type_name}"

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True, slots=True)
class StructuralSigma:
    """Reference to the structural error scale (sigma) of a latent variable.

    This is provided as an extension point. Whether it is used depends on the
    chosen identification strategy and builder support.
    """

    latent: str

    def key(self) -> str:
        return f"structural_sigma:{self.latent}"

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True, slots=True)
class StructuralCoefficient:
    """Reference to a coefficient of an explanatory variable in a structural equation.

    This target identifies the coefficient associated with one explanatory
    variable in the structural equation of one latent variable.
    """

    latent: str
    explanatory_variable: str

    def key(self) -> str:
        return f"structural_coefficient:{self.latent}:{self.explanatory_variable}"

    def __str__(self) -> str:
        return self.key()
