from __future__ import annotations

"""Semantic references to normalizable parameters."""

from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True, slots=True)
class ParameterRef:
    """Base semantic reference.

    Subclasses identify one semantically meaningful model parameter.
    """

    def key(self) -> tuple:
        return (self.__class__.__name__,)


@dataclass(frozen=True, slots=True)
class StructuralCoefficient(ParameterRef):
    latent_name: str
    variable_name: str

    def key(self) -> tuple:
        return self.__class__.__name__, self.latent_name, self.variable_name


@dataclass(frozen=True, slots=True)
class StructuralIntercept(ParameterRef):
    latent_name: str

    def key(self) -> tuple:
        return (self.__class__.__name__, self.latent_name)


@dataclass(frozen=True, slots=True)
class StructuralSigma(ParameterRef):
    latent_name: str

    def key(self) -> tuple:
        return (self.__class__.__name__, self.latent_name)


@dataclass(frozen=True, slots=True)
class MeasurementIntercept(ParameterRef):
    indicator_name: str

    def key(self) -> tuple:
        return (self.__class__.__name__, self.indicator_name)


@dataclass(frozen=True, slots=True)
class MeasurementLoading(ParameterRef):
    latent_name: str
    indicator_name: str

    def key(self) -> tuple:
        return (self.__class__.__name__, self.latent_name, self.indicator_name)


@dataclass(frozen=True, slots=True)
class MeasurementSigma(ParameterRef):
    indicator_name: str

    def key(self) -> tuple:
        return (self.__class__.__name__, self.indicator_name)


@dataclass(frozen=True, slots=True)
class ThresholdFirst(ParameterRef):
    type_name: str

    def key(self) -> tuple:
        return (self.__class__.__name__, self.type_name)


@dataclass(frozen=True, slots=True)
class ThresholdDelta(ParameterRef):
    type_name: str
    index: int

    def key(self) -> tuple:
        return (self.__class__.__name__, self.type_name, self.index)
