from __future__ import annotations

"""Pure specification objects for latent-variable models.

These classes describe *what the user wants* without committing to any
particular output format.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class MeasurementModel(str, Enum):
    """Measurement model used for one indicator."""

    GAUSSIAN = 'gaussian'
    ORDERED_PROBIT = 'ordered_probit'
    ORDERED_LOGIT = 'ordered_logit'


# Reusable specification for strictly positive parameters


@dataclass(frozen=True, slots=True)
class PositiveParameterSpec:
    """Specification of a strictly positive model parameter.

    The value is expressed on the natural scale of the parameter. Downstream
    generators may transform it internally, for example by using ``log`` for a
    positive reparameterization.

    :param start: Optional starting value on the natural scale.
    :param lower_bound: Optional lower bound on the natural scale.
    """

    start: float | None = None
    lower_bound: float | None = 0.0


@dataclass(frozen=True, slots=True)
class StructuralEquation:
    """Pure structural-equation specification.

    :param name: Name of the owning latent variable.
    :param intercept: Whether a structural intercept should be included.
    :param explanatory_variables: Variables entering the deterministic part.
    """

    name: str
    intercept: bool = True
    explanatory_variables: Iterable[str] = ()


@dataclass(frozen=True, slots=True)
class LatentVariable:
    """Pure latent-variable specification.

    :param name: Latent variable name.
    :param structural_equation: Structural equation specification.
    :param indicators: Names of indicators linked to this latent variable.
    :param structural_sigma: Optional specification of the structural standard
        deviation on its natural scale.
    """

    name: str
    structural_equation: StructuralEquation
    indicators: Iterable[str]
    structural_sigma: PositiveParameterSpec | None = None


@dataclass(frozen=True, slots=True)
class LikertType:
    """Shared indicator-type metadata.

    :param type_name: Type name.
    :param symmetric: Whether the threshold system is symmetric when used by
        ordinal indicators.
    :param categories: Ordered category labels.
    :param neutral_labels: Neutral or placeholder labels.
    """

    type_name: str
    symmetric: bool
    categories: list[int]
    neutral_labels: list[int]


@dataclass(frozen=True, slots=True)
class LikertIndicator:
    """Pure semantic indicator specification.

    :param name: Indicator name.
    :param statement: Human-readable text of the indicator.
    :param type_name: Shared type metadata name.
    """

    name: str
    statement: str
    type_name: str


@dataclass(frozen=True, slots=True)
class IndicatorMeasurementSpec:
    """Measurement specification attached to one indicator.

    This object describes *how* one semantic indicator is modeled in a given
    run, independently of the indicator definition itself.

    :param indicator_name: Name of the indicator being modeled.
    :param measurement_model: Statistical measurement model used for that
        indicator.
    :param measurement_sigma: Optional specification of the measurement scale
        on its natural scale.
    """

    indicator_name: str
    measurement_model: MeasurementModel
    measurement_sigma: PositiveParameterSpec | None = None


@dataclass(frozen=True, slots=True)
class MeasurementConfiguration:
    """Collection of indicator-level measurement specifications.

    :param specifications: Measurement specification for each modeled
        indicator.
    """

    specifications: Iterable[IndicatorMeasurementSpec]
