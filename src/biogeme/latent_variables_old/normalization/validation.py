# normalization/validation.py
"""
Validation utilities for normalization plans.

This module performs *structural* validation of a plan against a specification:
- targets exist (indicators, latent variables, types),
- targets are compatible with the specification (e.g., ThresholdFirst only for non-symmetric types),
- fixed values respect obvious domain constraints (e.g., sigma > 0 if fixed).

It does *not* attempt to prove identification.

Michel Bierlaire
Wed Mar 04 2026, 16:35:47

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, cast

from biogeme.latent_variables.latent_variables import LatentVariable
from biogeme.latent_variables.likert_indicators import (
    LikertIndicator,
    LikertType,
    MeasurementModel,
)

from .parameter_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
    ParameterRef,
    StructuralCoefficient,
    StructuralSigma,
    ThresholdFirst,
)
from .plan import NormalizationPlan


class DiagnosticLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A validation diagnostic."""

    level: DiagnosticLevel
    code: str
    message: str
    target: ParameterRef | None = None


def validate_plan(
    *,
    latent_variables: Iterable[LatentVariable],
    likert_indicators: Iterable[LikertIndicator],
    likert_types: Iterable[LikertType],
    plan: NormalizationPlan,
) -> list[Diagnostic]:
    """Validate a normalization plan against a specification.

    :param latent_variables:
        Latent-variable specifications.
    :param likert_indicators:
        Likert indicator specifications.
    :param likert_types:
        Likert type specifications.
    :param plan:
        Normalization plan to validate.
    :return:
        List of diagnostics (errors and warnings).
    """
    diags: list[Diagnostic] = []

    lv_names = {lv.name for lv in latent_variables}
    indicator_names = {ind.name for ind in likert_indicators}
    type_by_name = {t.type_name: t for t in likert_types}

    indicators_by_latent = {lv.name: set(lv.indicators) for lv in latent_variables}
    explanatory_variables_by_latent = {
        lv.name: set(lv.structural_equation.explanatory_variables)
        for lv in latent_variables
    }
    indicators_by_type = {
        t.type_name: {
            ind.name for ind in likert_indicators if ind.type_name == t.type_name
        }
        for t in likert_types
    }
    ordinal_indicators_by_type = {
        t.type_name: {
            ind.name
            for ind in likert_indicators
            if ind.type_name == t.type_name
            and ind.measurement_model
            in {
                MeasurementModel.ORDERED_PROBIT,
                MeasurementModel.ORDERED_LOGIT,
            }
        }
        for t in likert_types
    }
    gaussian_indicators_by_type = {
        t.type_name: {
            ind.name
            for ind in likert_indicators
            if ind.type_name == t.type_name
            and ind.measurement_model == MeasurementModel.GAUSSIAN
        }
        for t in likert_types
    }

    def _unknown_target_type(target: ParameterRef, value: float) -> list[Diagnostic]:
        _ = value  # value is not validated for unknown targets
        return [
            Diagnostic(
                level=DiagnosticLevel.WARNING,
                code="unknown_target_type",
                message=f"Unknown fixing target type: {type(target).__name__}.",
                target=target,
            )
        ]

    def _validate_measurement_intercept(
        target: MeasurementIntercept, value: float
    ) -> list[Diagnostic]:
        _ = value  # value is not validated for intercepts
        if target.indicator in indicator_names:
            return []
        return [
            Diagnostic(
                level=DiagnosticLevel.ERROR,
                code="unknown_indicator",
                message=f"Unknown indicator '{target.indicator}' in fixing '{target}'.",
                target=target,
            )
        ]

    def _validate_measurement_loading(
        target: MeasurementLoading, value: float
    ) -> list[Diagnostic]:
        _ = value  # value is not validated for loadings
        out: list[Diagnostic] = []
        if target.latent not in lv_names:
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="unknown_latent",
                    message=f"Unknown latent variable '{target.latent}' in fixing '{target}'.",
                    target=target,
                )
            )
        if target.indicator not in indicator_names:
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="unknown_indicator",
                    message=f"Unknown indicator '{target.indicator}' in fixing '{target}'.",
                    target=target,
                )
            )
        if (
            target.latent in indicators_by_latent
            and target.indicator in indicator_names
            and target.indicator not in indicators_by_latent[target.latent]
        ):
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="invalid_measurement_loading",
                    message=(
                        f"Indicator '{target.indicator}' is not attached to latent variable "
                        f"'{target.latent}' in fixing '{target}'."
                    ),
                    target=target,
                )
            )
        return out

    def _validate_measurement_sigma(
        target: MeasurementSigma, value: float
    ) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        if target.indicator not in indicator_names:
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="unknown_indicator",
                    message=f"Unknown indicator '{target.indicator}' in fixing '{target}'.",
                    target=target,
                )
            )
        if value <= 0:
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nonpositive_sigma",
                    message=f"Measurement sigma must be > 0, got {value} in fixing '{target}'.",
                    target=target,
                )
            )
        return out

    def _validate_threshold_first(
        target: ThresholdFirst, value: float
    ) -> list[Diagnostic]:
        _ = value  # value is not validated for tau_1
        lt = type_by_name.get(target.type_name)
        if lt is None:
            return [
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="unknown_type",
                    message=f"Unknown Likert type '{target.type_name}' in fixing '{target}'.",
                    target=target,
                )
            ]
        if not indicators_by_type.get(target.type_name):
            return [
                Diagnostic(
                    level=DiagnosticLevel.WARNING,
                    code="unused_type",
                    message=(
                        f"Likert type '{target.type_name}' is not used by any indicator."
                    ),
                    target=target,
                )
            ]
        if not ordinal_indicators_by_type.get(target.type_name):
            gaussian_indicators = sorted(gaussian_indicators_by_type[target.type_name])
            return [
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="threshold_first_not_applicable",
                    message=(
                        f"Fixing '{target}' is not applicable: type '{target.type_name}' is used "
                        f"only by Gaussian indicators {gaussian_indicators}, so no threshold system "
                        f"is defined for it."
                    ),
                    target=target,
                )
            ]
        if lt.symmetric:
            return [
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="threshold_first_not_applicable",
                    message=(
                        f"Fixing '{target}' is not applicable: Likert type "
                        f"'{target.type_name}' is symmetric and does not expose a free tau_1."
                    ),
                    target=target,
                )
            ]
        return []

    def _validate_structural_coefficient(
        target: StructuralCoefficient, value: float
    ) -> list[Diagnostic]:
        _ = value  # value is not validated for structural coefficients
        out: list[Diagnostic] = []
        if target.latent not in lv_names:
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="unknown_latent",
                    message=f"Unknown latent variable '{target.latent}' in fixing '{target}'.",
                    target=target,
                )
            )
        if (
            target.latent in explanatory_variables_by_latent
            and target.explanatory_variable
            not in explanatory_variables_by_latent[target.latent]
        ):
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="unknown_explanatory_variable",
                    message=(
                        f"Unknown explanatory variable '{target.explanatory_variable}' "
                        f"for latent variable '{target.latent}' in fixing '{target}'."
                    ),
                    target=target,
                )
            )
        return out

    def _validate_structural_sigma(
        target: StructuralSigma, value: float
    ) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        if target.latent not in lv_names:
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="unknown_latent",
                    message=f"Unknown latent variable '{target.latent}' in fixing '{target}'.",
                    target=target,
                )
            )
        if value <= 0:
            out.append(
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nonpositive_sigma",
                    message=f"Structural sigma must be > 0, got {value} in fixing '{target}'.",
                    target=target,
                )
            )
        return out

    Validator = Callable[[Any, float], list[Diagnostic]]
    validators: dict[type[ParameterRef], Validator] = {
        MeasurementIntercept: _validate_measurement_intercept,
        MeasurementLoading: _validate_measurement_loading,
        MeasurementSigma: _validate_measurement_sigma,
        ThresholdFirst: _validate_threshold_first,
        StructuralCoefficient: _validate_structural_coefficient,
        StructuralSigma: _validate_structural_sigma,
    }

    for fixing in plan.as_list():
        target = fixing.target
        value = fixing.value
        validator = validators.get(type(target), _unknown_target_type)
        diags.extend(validator(cast(Any, target), value))

    return diags
