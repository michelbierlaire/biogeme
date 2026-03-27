from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum

import pytest
from biogeme.latent_variables.model_spec import (
    LatentVariable,
    LikertIndicator,
    LikertType,
    PositiveParameterSpec,
    StructuralEquation,
)
from biogeme.latent_variables.normalization_plan import Fixing, NormalizationPlan
from biogeme.latent_variables.normalization_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
    ParameterRef,
    StructuralCoefficient,
    StructuralSigma,
    ThresholdDelta,
    ThresholdFirst,
)
from biogeme.latent_variables.validation import (
    Diagnostic,
    ValidationLevel,
    ValidationResult,
    validate_normalization_plan,
    validate_specification,
)


def make_latent_variable(
    *,
    name: str,
    structural_name: str | None = None,
    explanatory_variables: list[str] | None = None,
    indicators: list[str] | None = None,
    structural_sigma: PositiveParameterSpec | None = None,
) -> LatentVariable:
    return LatentVariable(
        name=name,
        structural_equation=StructuralEquation(
            name=name if structural_name is None else structural_name,
            explanatory_variables=(
                [] if explanatory_variables is None else explanatory_variables
            ),
        ),
        indicators=[] if indicators is None else indicators,
        structural_sigma=structural_sigma,
    )


def make_indicator(
    *, name: str, type_name: str, statement: str | None = None
) -> LikertIndicator:
    return LikertIndicator(
        name=name,
        statement=name if statement is None else statement,
        type_name=type_name,
    )


def make_type(
    *,
    type_name: str,
    symmetric: bool,
    categories: list[int],
    neutral_labels: list[int] | None = None,
) -> LikertType:
    return LikertType(
        type_name=type_name,
        symmetric=symmetric,
        categories=categories,
        neutral_labels=[] if neutral_labels is None else neutral_labels,
    )


class UnknownTarget(ParameterRef):
    marker: str

    def __init__(self, marker: str) -> None:
        object.__setattr__(self, "marker", marker)

    def key(self) -> tuple[str, str]:
        return ("UnknownTarget", self.marker)


def codes(result: ValidationResult) -> list[str]:
    return [diagnostic.code for diagnostic in result.diagnostics]


def messages(result: ValidationResult) -> list[str]:
    return [diagnostic.message for diagnostic in result.diagnostics]


def test_validation_level_is_string_enum() -> None:
    assert issubclass(ValidationLevel, str)
    assert issubclass(ValidationLevel, Enum)
    assert ValidationLevel.ERROR.value == "error"
    assert ValidationLevel.WARNING.value == "warning"


def test_diagnostic_is_frozen_and_slotted() -> None:
    diagnostic = Diagnostic(
        level=ValidationLevel.ERROR,
        code="some_code",
        message="some message",
    )

    assert diagnostic.level is ValidationLevel.ERROR
    assert diagnostic.code == "some_code"
    assert diagnostic.message == "some message"
    assert hasattr(Diagnostic, "__slots__")
    assert "__dict__" not in dir(diagnostic)

    with pytest.raises(FrozenInstanceError):
        diagnostic.code = "other"  # type: ignore[misc]


def test_validation_result_errors_filters_only_errors() -> None:
    error_1 = Diagnostic(ValidationLevel.ERROR, "e1", "first error")
    warning = Diagnostic(ValidationLevel.WARNING, "w1", "a warning")
    error_2 = Diagnostic(ValidationLevel.ERROR, "e2", "second error")

    result = ValidationResult([error_1, warning, error_2])

    assert result.errors == [error_1, error_2]


def test_validation_result_raise_for_errors_does_nothing_without_errors() -> None:
    result = ValidationResult(
        [Diagnostic(ValidationLevel.WARNING, "w1", "warning only")]
    )

    result.raise_for_errors()


def test_validation_result_raise_for_errors_raises_with_formatted_messages() -> None:
    result = ValidationResult(
        [
            Diagnostic(ValidationLevel.ERROR, "e1", "first error"),
            Diagnostic(ValidationLevel.WARNING, "w1", "ignored warning"),
            Diagnostic(ValidationLevel.ERROR, "e2", "second error"),
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"Validation failed:\n- e1: first error\n- e2: second error",
    ):
        result.raise_for_errors()


def test_validate_specification_returns_empty_result_for_valid_input() -> None:
    latent_variables = [
        make_latent_variable(
            name="LV1",
            explanatory_variables=["x1"],
            indicators=["ind1"],
        )
    ]
    likert_indicators = [make_indicator(name="ind1", type_name="type1")]
    likert_types = [
        make_type(
            type_name="type1",
            symmetric=False,
            categories=[1, 2, 3],
        )
    ]

    result = validate_specification(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
    )

    assert result.diagnostics == []
    assert result.errors == []


def test_validate_specification_detects_duplicate_latent_variable_names() -> None:
    result = validate_specification(
        latent_variables=[
            make_latent_variable(name="LV1", indicators=["ind1"]),
            make_latent_variable(name="LV1", indicators=["ind1"]),
        ],
        likert_indicators=[make_indicator(name="ind1", type_name="type1")],
        likert_types=[make_type(type_name="type1", symmetric=False, categories=[1, 2])],
    )

    assert codes(result) == ["duplicate_latent_variable"]
    assert messages(result) == ["Latent variable names must be unique."]


def test_validate_specification_detects_duplicate_indicator_names() -> None:
    result = validate_specification(
        latent_variables=[make_latent_variable(name="LV1", indicators=["ind1"])],
        likert_indicators=[
            make_indicator(name="ind1", type_name="type1"),
            make_indicator(name="ind1", type_name="type1"),
        ],
        likert_types=[make_type(type_name="type1", symmetric=False, categories=[1, 2])],
    )

    assert codes(result) == ["duplicate_indicator"]
    assert messages(result) == ["Indicator names must be unique."]


def test_validate_specification_detects_duplicate_type_names() -> None:
    result = validate_specification(
        latent_variables=[make_latent_variable(name="LV1", indicators=["ind1"])],
        likert_indicators=[make_indicator(name="ind1", type_name="type1")],
        likert_types=[
            make_type(type_name="type1", symmetric=False, categories=[1, 2]),
            make_type(type_name="type1", symmetric=True, categories=[1, 2, 3]),
        ],
    )

    assert codes(result) == ["duplicate_type"]
    assert messages(result) == ["Likert type names must be unique."]


def test_validate_specification_detects_structural_name_mismatch() -> None:
    result = validate_specification(
        latent_variables=[
            make_latent_variable(
                name="LV1",
                structural_name="OTHER",
                indicators=["ind1"],
            )
        ],
        likert_indicators=[make_indicator(name="ind1", type_name="type1")],
        likert_types=[make_type(type_name="type1", symmetric=False, categories=[1, 2])],
    )

    assert codes(result) == ["structural_name_mismatch"]
    assert messages(result) == [
        "Structural equation name 'OTHER' does not match latent variable 'LV1'."
    ]


def test_validate_specification_warns_when_latent_has_no_indicator() -> None:
    result = validate_specification(
        latent_variables=[make_latent_variable(name="LV1", indicators=[])],
        likert_indicators=[],
        likert_types=[],
    )

    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].level is ValidationLevel.WARNING
    assert result.diagnostics[0].code == "latent_without_indicator"
    assert result.diagnostics[0].message == "Latent variable 'LV1' has no indicator."


def test_validate_specification_detects_unknown_indicator_referenced_by_latent() -> (
    None
):
    result = validate_specification(
        latent_variables=[make_latent_variable(name="LV1", indicators=["ghost"])],
        likert_indicators=[],
        likert_types=[],
    )

    assert codes(result) == ["unknown_indicator"]
    assert messages(result) == [
        "Latent variable 'LV1' references unknown indicator 'ghost'."
    ]


def test_validate_specification_detects_unknown_type_referenced_by_indicator() -> None:
    result = validate_specification(
        latent_variables=[make_latent_variable(name="LV1", indicators=["ind1"])],
        likert_indicators=[make_indicator(name="ind1", type_name="missing_type")],
        likert_types=[],
    )

    assert codes(result) == ["unknown_type"]
    assert messages(result) == [
        "Indicator 'ind1' refers to unknown type 'missing_type'."
    ]


def test_validate_specification_detects_type_with_too_few_categories() -> None:
    result = validate_specification(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[make_type(type_name="bad_type", symmetric=False, categories=[1])],
    )

    assert codes(result) == ["too_few_categories"]
    assert messages(result) == ["Type 'bad_type' must define at least two categories."]


def test_validate_specification_can_accumulate_multiple_diagnostics_in_order() -> None:
    result = validate_specification(
        latent_variables=[
            make_latent_variable(
                name="LV1",
                structural_name="WRONG",
                indicators=[],
            ),
            make_latent_variable(
                name="LV1",
                indicators=["ghost"],
            ),
        ],
        likert_indicators=[
            make_indicator(name="ind1", type_name="missing"),
            make_indicator(name="ind1", type_name="missing"),
        ],
        likert_types=[
            make_type(type_name="type1", symmetric=False, categories=[1]),
            make_type(type_name="type1", symmetric=True, categories=[1, 2, 3]),
        ],
    )

    assert codes(result) == [
        "duplicate_latent_variable",
        "duplicate_indicator",
        "duplicate_type",
        "structural_name_mismatch",
        "latent_without_indicator",
        "unknown_indicator",
        "unknown_type",
        "unknown_type",
        "too_few_categories",
    ]


def test_validate_normalization_plan_returns_empty_result_when_plan_is_none() -> None:
    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=None,
    )

    assert result.diagnostics == []


def test_validate_normalization_plan_accepts_valid_targets_of_each_supported_type() -> (
    None
):
    latent_variables = [
        make_latent_variable(name="LV1", indicators=["ind1"]),
    ]
    likert_indicators = [
        make_indicator(name="ind1", type_name="type1"),
    ]
    likert_types = [
        make_type(type_name="type1", symmetric=False, categories=[1, 2, 3]),
    ]
    plan = NormalizationPlan(
        [
            Fixing(target=StructuralCoefficient("LV1", "x1"), value=0.0),
            Fixing(target=StructuralSigma("LV1"), value=1.0),
            Fixing(target=MeasurementIntercept("ind1"), value=0.0),
            Fixing(target=MeasurementLoading("LV1", "ind1"), value=1.0),
            Fixing(target=MeasurementSigma("ind1"), value=1.0),
            Fixing(target=ThresholdFirst("type1"), value=-1.0),
            Fixing(target=ThresholdDelta("type1", 1), value=1.0),
        ]
    )

    result = validate_normalization_plan(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        normalization_plan=plan,
    )

    assert result.diagnostics == []


def test_validate_normalization_plan_detects_unknown_latent_for_structural_coefficient() -> (
    None
):
    plan = NormalizationPlan(
        [Fixing(target=StructuralCoefficient("UNKNOWN", "x1"), value=0.0)]
    )

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_latent"]
    assert messages(result) == [
        "Unknown latent variable 'UNKNOWN' in fixing 'StructuralCoefficient(latent_name='UNKNOWN', variable_name='x1')'."
    ]


def test_validate_normalization_plan_detects_unknown_latent_for_structural_sigma() -> (
    None
):
    plan = NormalizationPlan([Fixing(target=StructuralSigma("UNKNOWN"), value=1.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_latent"]
    assert messages(result) == [
        "Unknown latent variable 'UNKNOWN' in fixing 'StructuralSigma(latent_name='UNKNOWN')'."
    ]


def test_validate_normalization_plan_detects_unknown_indicator_for_measurement_intercept() -> (
    None
):
    plan = NormalizationPlan([Fixing(target=MeasurementIntercept("ghost"), value=0.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_indicator"]
    assert messages(result) == [
        "Unknown indicator 'ghost' in fixing 'MeasurementIntercept(indicator_name='ghost')'."
    ]


def test_validate_normalization_plan_detects_unknown_latent_for_measurement_loading() -> (
    None
):
    plan = NormalizationPlan(
        [Fixing(target=MeasurementLoading("UNKNOWN", "ind1"), value=1.0)]
    )

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[make_indicator(name="ind1", type_name="type1")],
        likert_types=[make_type(type_name="type1", symmetric=False, categories=[1, 2])],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_latent"]
    assert messages(result) == [
        "Unknown latent variable 'UNKNOWN' in fixing 'MeasurementLoading(latent_name='UNKNOWN', indicator_name='ind1')'."
    ]


def test_validate_normalization_plan_detects_unknown_indicator_for_measurement_loading() -> (
    None
):
    plan = NormalizationPlan(
        [Fixing(target=MeasurementLoading("LV1", "ghost"), value=1.0)]
    )

    result = validate_normalization_plan(
        latent_variables=[make_latent_variable(name="LV1", indicators=[])],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_indicator"]
    assert messages(result) == [
        "Unknown indicator 'ghost' in fixing 'MeasurementLoading(latent_name='LV1', indicator_name='ghost')'."
    ]


def test_validate_normalization_plan_measurement_loading_can_emit_two_errors() -> None:
    plan = NormalizationPlan(
        [Fixing(target=MeasurementLoading("UNKNOWN", "ghost"), value=1.0)]
    )

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_latent", "unknown_indicator"]
    assert messages(result) == [
        "Unknown latent variable 'UNKNOWN' in fixing 'MeasurementLoading(latent_name='UNKNOWN', indicator_name='ghost')'.",
        "Unknown indicator 'ghost' in fixing 'MeasurementLoading(latent_name='UNKNOWN', indicator_name='ghost')'.",
    ]


def test_validate_normalization_plan_detects_unknown_indicator_for_measurement_sigma() -> (
    None
):
    plan = NormalizationPlan([Fixing(target=MeasurementSigma("ghost"), value=1.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_indicator"]
    assert messages(result) == [
        "Unknown indicator 'ghost' in fixing 'MeasurementSigma(indicator_name='ghost')'."
    ]


def test_validate_normalization_plan_detects_unknown_type_for_threshold_first() -> None:
    plan = NormalizationPlan([Fixing(target=ThresholdFirst("missing"), value=0.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_type"]
    assert messages(result) == [
        "Unknown type 'missing' in fixing 'ThresholdFirst(type_name='missing')'."
    ]


def test_validate_normalization_plan_detects_threshold_first_not_applicable_for_symmetric_type() -> (
    None
):
    plan = NormalizationPlan([Fixing(target=ThresholdFirst("sym_type"), value=0.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[
            make_type(type_name="sym_type", symmetric=True, categories=[1, 2, 3])
        ],
        normalization_plan=plan,
    )

    assert codes(result) == ["threshold_not_applicable"]
    assert messages(result) == [
        "Type 'sym_type' is symmetric, so fixing 'ThresholdFirst(type_name='sym_type')' is not applicable."
    ]


def test_validate_normalization_plan_accepts_threshold_first_for_nonsymmetric_type() -> (
    None
):
    plan = NormalizationPlan([Fixing(target=ThresholdFirst("mono_type"), value=-1.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[
            make_type(type_name="mono_type", symmetric=False, categories=[1, 2, 3])
        ],
        normalization_plan=plan,
    )

    assert result.diagnostics == []


def test_validate_normalization_plan_detects_unknown_type_for_threshold_delta() -> None:
    plan = NormalizationPlan([Fixing(target=ThresholdDelta("missing", 1), value=1.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_type"]
    assert messages(result) == [
        "Unknown type 'missing' in fixing 'ThresholdDelta(type_name='missing', index=1)'."
    ]


def test_validate_normalization_plan_detects_unknown_fixing_target_type() -> None:
    plan = NormalizationPlan([Fixing(target=UnknownTarget("weird"), value=1.0)])

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[],
        normalization_plan=plan,
    )

    assert codes(result) == ["unknown_fixing_target"]
    assert messages(result) == ["Unknown fixing target type: UnknownTarget."]


def test_validate_normalization_plan_can_accumulate_multiple_diagnostics_in_order() -> (
    None
):
    plan = NormalizationPlan(
        [
            Fixing(target=StructuralCoefficient("LV_X", "x1"), value=0.0),
            Fixing(target=MeasurementLoading("LV_Y", "ind_missing"), value=1.0),
            Fixing(target=ThresholdFirst("sym_type"), value=0.0),
            Fixing(target=ThresholdDelta("missing_type", 1), value=1.0),
            Fixing(target=UnknownTarget("u"), value=9.0),
        ]
    )

    result = validate_normalization_plan(
        latent_variables=[],
        likert_indicators=[],
        likert_types=[
            make_type(type_name="sym_type", symmetric=True, categories=[1, 2, 3])
        ],
        normalization_plan=plan,
    )

    assert codes(result) == [
        "unknown_latent",
        "unknown_indicator",
        "unknown_latent",
        "unknown_type",
        "threshold_not_applicable",
        "unknown_fixing_target",
    ]
