from __future__ import annotations

"""Validation for specifications and normalization plans."""

from dataclasses import dataclass
from enum import Enum

from .model_spec import LatentVariable, LikertIndicator, LikertType
from .normalization_plan import NormalizationPlan
from .normalization_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
    StructuralCoefficient,
    StructuralIntercept,
    StructuralSigma,
    ThresholdDelta,
    ThresholdFirst,
)


class ValidationLevel(str, Enum):
    ERROR = 'error'
    WARNING = 'warning'


@dataclass(frozen=True, slots=True)
class Diagnostic:
    level: ValidationLevel
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    diagnostics: list[Diagnostic]

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.level == ValidationLevel.ERROR]

    def raise_for_errors(self) -> None:
        if self.errors:
            messages = '\n'.join(f'- {d.code}: {d.message}' for d in self.errors)
            raise ValueError(f'Validation failed:\n{messages}')


def validate_specification(
    *,
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
) -> ValidationResult:
    diagnostics: list[Diagnostic] = []
    lv_names = [lv.name for lv in latent_variables]
    if len(set(lv_names)) != len(lv_names):
        diagnostics.append(
            Diagnostic(
                ValidationLevel.ERROR,
                'duplicate_latent_variable',
                'Latent variable names must be unique.',
            )
        )
    ind_names = [ind.name for ind in likert_indicators]
    if len(set(ind_names)) != len(ind_names):
        diagnostics.append(
            Diagnostic(
                ValidationLevel.ERROR,
                'duplicate_indicator',
                'Indicator names must be unique.',
            )
        )
    type_names = [lt.type_name for lt in likert_types]
    if len(set(type_names)) != len(type_names):
        diagnostics.append(
            Diagnostic(
                ValidationLevel.ERROR,
                'duplicate_type',
                'Likert type names must be unique.',
            )
        )

    indicator_by_name = {ind.name: ind for ind in likert_indicators}
    type_by_name = {lt.type_name: lt for lt in likert_types}

    for lv in latent_variables:
        if lv.structural_equation.name != lv.name:
            diagnostics.append(
                Diagnostic(
                    ValidationLevel.ERROR,
                    'structural_name_mismatch',
                    f"Structural equation name '{lv.structural_equation.name}' does not match latent variable '{lv.name}'.",
                )
            )
        if not set(lv.indicators):
            diagnostics.append(
                Diagnostic(
                    ValidationLevel.WARNING,
                    'latent_without_indicator',
                    f"Latent variable '{lv.name}' has no indicator.",
                )
            )
        for indicator_name in lv.indicators:
            if indicator_name not in indicator_by_name:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_indicator',
                        f"Latent variable '{lv.name}' references unknown indicator '{indicator_name}'.",
                    )
                )
    for ind in likert_indicators:
        if ind.type_name not in type_by_name:
            diagnostics.append(
                Diagnostic(
                    ValidationLevel.ERROR,
                    'unknown_type',
                    f"Indicator '{ind.name}' refers to unknown type '{ind.type_name}'.",
                )
            )
    for lt in likert_types:
        if len(lt.categories) < 2:
            diagnostics.append(
                Diagnostic(
                    ValidationLevel.ERROR,
                    'too_few_categories',
                    f"Type '{lt.type_name}' must define at least two categories.",
                )
            )

    return ValidationResult(diagnostics)


def validate_normalization_plan(
    *,
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
    normalization_plan: NormalizationPlan | None,
) -> ValidationResult:
    if normalization_plan is None:
        return ValidationResult([])

    diagnostics: list[Diagnostic] = []
    lv_names = {lv.name for lv in latent_variables}
    indicator_names = {ind.name for ind in likert_indicators}
    type_by_name = {lt.type_name: lt for lt in likert_types}

    for fixing in normalization_plan:
        target = fixing.target
        if isinstance(target, StructuralCoefficient):
            if target.latent_name not in lv_names:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_latent',
                        f"Unknown latent variable '{target.latent_name}' in fixing '{target}'.",
                    )
                )
        elif isinstance(target, StructuralIntercept):
            if target.latent_name not in lv_names:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_latent',
                        f"Unknown latent variable '{target.latent_name}' in fixing '{target}'.",
                    )
                )
        elif isinstance(target, StructuralSigma):
            if target.latent_name not in lv_names:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_latent',
                        f"Unknown latent variable '{target.latent_name}' in fixing '{target}'.",
                    )
                )
        elif isinstance(target, MeasurementIntercept):
            if target.indicator_name not in indicator_names:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_indicator',
                        f"Unknown indicator '{target.indicator_name}' in fixing '{target}'.",
                    )
                )
        elif isinstance(target, MeasurementLoading):
            if target.latent_name not in lv_names:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_latent',
                        f"Unknown latent variable '{target.latent_name}' in fixing '{target}'.",
                    )
                )
            if target.indicator_name not in indicator_names:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_indicator',
                        f"Unknown indicator '{target.indicator_name}' in fixing '{target}'.",
                    )
                )
        elif isinstance(target, MeasurementSigma):
            if target.indicator_name not in indicator_names:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_indicator',
                        f"Unknown indicator '{target.indicator_name}' in fixing '{target}'.",
                    )
                )
        elif isinstance(target, ThresholdFirst):
            lt = type_by_name.get(target.type_name)
            if lt is None:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_type',
                        f"Unknown type '{target.type_name}' in fixing '{target}'.",
                    )
                )
            elif lt.symmetric:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'threshold_not_applicable',
                        f"Type '{target.type_name}' is symmetric, so fixing '{target}' is not applicable.",
                    )
                )
        elif isinstance(target, ThresholdDelta):
            if target.type_name not in type_by_name:
                diagnostics.append(
                    Diagnostic(
                        ValidationLevel.ERROR,
                        'unknown_type',
                        f"Unknown type '{target.type_name}' in fixing '{target}'.",
                    )
                )
        else:
            diagnostics.append(
                Diagnostic(
                    ValidationLevel.ERROR,
                    'unknown_fixing_target',
                    f'Unknown fixing target type: {type(target).__name__}.',
                )
            )

    return ValidationResult(diagnostics)
