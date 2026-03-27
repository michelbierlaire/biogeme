from __future__ import annotations

"""Resolved semantic model used by all outputs."""

from dataclasses import dataclass, field
from enum import Enum

from .context import EstimationMode
from .model_spec import MeasurementModel
from .normalization_refs import ParameterRef


class MeasurementErrorDistribution(str, Enum):
    GAUSSIAN = 'gaussian'
    LOGISTIC = 'logistic'


class ThresholdConstructionKind(str, Enum):
    SYMMETRIC = 'symmetric'
    MONOTONE = 'monotone'


class ParameterStatus(str, Enum):
    FIXED = 'fixed'
    FREE = 'free'


class ParameterCreationKind(str, Enum):
    NUMERIC_CONSTANT = 'numeric_constant'
    FREE_BETA = 'free_beta'
    FIXED_BETA = 'fixed_beta'
    LOG_EXP_BETA = 'log_exp_beta'
    BOUNDED_BETA = 'bounded_beta'


class PositivityStrategy(str, Enum):
    NONE = 'none'
    LOG_EXP = 'log_exp'
    LOWER_BOUND = 'lower_bound'


class ParameterRole(str, Enum):
    STRUCTURAL_COEFFICIENT = 'structural_coefficient'
    STRUCTURAL_SIGMA = 'structural_sigma'
    MEASUREMENT_INTERCEPT = 'measurement_intercept'
    MEASUREMENT_LOADING = 'measurement_loading'
    MEASUREMENT_SIGMA = 'measurement_sigma'
    THRESHOLD_FIRST = 'threshold_first'
    THRESHOLD_DELTA = 'threshold_delta'


class CutpointKind(str, Enum):
    FREE = 'free'
    FIXED = 'fixed'
    DERIVED = 'derived'


@dataclass(frozen=True, slots=True)
class ResolvedParameter:
    semantic_ref: ParameterRef | None
    final_name: str
    role: ParameterRole
    status: ParameterStatus
    fixed_value: float | None
    initial_value: float
    lower_bound: float | None
    upper_bound: float | None
    positivity_strategy: PositivityStrategy | None
    creation_kind: ParameterCreationKind
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ResolvedParameterRef:
    final_name: str
    semantic_ref: ParameterRef | None = None


@dataclass(frozen=True, slots=True)
class ResolvedConstant:
    value: float


@dataclass(frozen=True, slots=True)
class ResolvedLinearTerm:
    coefficient: ResolvedParameterRef | ResolvedConstant
    variable_name: str


@dataclass(frozen=True, slots=True)
class ResolvedLinearCombination:
    intercept: ResolvedParameterRef | ResolvedConstant | None
    terms: list[ResolvedLinearTerm]


@dataclass(frozen=True, slots=True)
class ResolvedStructuralEquation:
    latent_name: str
    expression_name: str
    terms: list[ResolvedLinearTerm]
    sigma: ResolvedParameterRef | None
    draw_name: str
    draw_type: str
    error_distribution: str


@dataclass(frozen=True, slots=True)
class ResolvedCutpoint:
    symbol_name: str
    kind: CutpointKind
    expression_text: str
    source_parameter_names: list[str]


@dataclass(frozen=True, slots=True)
class ResolvedThresholdSystem:
    type_name: str
    symmetric: bool
    categories: list[int]
    neutral_labels: list[int]
    construction_kind: ThresholdConstructionKind
    cutpoints: list[ResolvedCutpoint]
    used_by_indicators: list[str]
    normalization_notes: list[str]


@dataclass(frozen=True, slots=True)
class ResolvedMeasurementEquation:
    indicator_name: str
    statement: str
    type_name: str
    measurement_model: MeasurementModel
    systematic_part: ResolvedLinearCombination
    sigma: ResolvedParameterRef | None
    observed_variable_name: str
    threshold_system_name: str | None
    error_distribution: MeasurementErrorDistribution
    normalization_notes: list[str]


@dataclass(frozen=True, slots=True)
class ResolvedNormalizationRule:
    scope: str
    target_name: str
    value: float | str
    reason: str


@dataclass(frozen=True, slots=True)
class ResolvedNormalizationSummary:
    rules: list[ResolvedNormalizationRule]
    warnings: list[str]
    disclaimer: str


@dataclass(frozen=True, slots=True)
class ResolvedLatentVariable:
    name: str
    structural_equation: ResolvedStructuralEquation
    indicator_names: list[str]
    reference_indicator: str | None
    normalization_notes: list[str]


@dataclass(frozen=True, slots=True)
class ResolvedModelMetadata:
    estimation_mode: EstimationMode
    measurement_models_present: list[MeasurementModel]
    has_gaussian: bool
    has_ordered_probit: bool
    has_ordered_logit: bool
    has_ordinal: bool
    n_latent_variables: int
    n_indicators: int
    n_threshold_systems: int


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    metadata: ResolvedModelMetadata
    latent_variables: dict[str, ResolvedLatentVariable]
    measurement_equations: dict[str, ResolvedMeasurementEquation]
    threshold_systems: dict[str, ResolvedThresholdSystem]
    parameters: dict[str, ResolvedParameter]
    normalization: ResolvedNormalizationSummary
