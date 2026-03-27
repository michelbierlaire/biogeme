from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum

import pytest
from biogeme.latent_variables.context import EstimationMode
from biogeme.latent_variables.model_spec import MeasurementModel
from biogeme.latent_variables.normalization_refs import ParameterRef
from biogeme.latent_variables.resolved import (
    CutpointKind,
    MeasurementErrorDistribution,
    ParameterCreationKind,
    ParameterRole,
    ParameterStatus,
    PositivityStrategy,
    ResolvedConstant,
    ResolvedCutpoint,
    ResolvedLatentVariable,
    ResolvedLinearCombination,
    ResolvedLinearTerm,
    ResolvedMeasurementEquation,
    ResolvedModel,
    ResolvedModelMetadata,
    ResolvedNormalizationRule,
    ResolvedNormalizationSummary,
    ResolvedParameter,
    ResolvedParameterRef,
    ResolvedStructuralEquation,
    ResolvedThresholdSystem,
    ThresholdConstructionKind,
)


class DummyParameterRef(ParameterRef):
    name: str

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "name", name)

    def key(self) -> tuple[str, str]:
        return ("DummyParameterRef", self.name)


def dummy_ref(name: str = "p") -> DummyParameterRef:
    return DummyParameterRef(name)


def test_measurement_error_distribution_enum_values() -> None:
    assert issubclass(MeasurementErrorDistribution, str)
    assert issubclass(MeasurementErrorDistribution, Enum)
    assert MeasurementErrorDistribution.GAUSSIAN.value == "gaussian"
    assert MeasurementErrorDistribution.LOGISTIC.value == "logistic"


def test_threshold_construction_kind_enum_values() -> None:
    assert issubclass(ThresholdConstructionKind, str)
    assert issubclass(ThresholdConstructionKind, Enum)
    assert ThresholdConstructionKind.SYMMETRIC.value == "symmetric"
    assert ThresholdConstructionKind.MONOTONE.value == "monotone"


def test_parameter_status_enum_values() -> None:
    assert issubclass(ParameterStatus, str)
    assert issubclass(ParameterStatus, Enum)
    assert ParameterStatus.FIXED.value == "fixed"
    assert ParameterStatus.FREE.value == "free"


def test_parameter_creation_kind_enum_values() -> None:
    assert issubclass(ParameterCreationKind, str)
    assert issubclass(ParameterCreationKind, Enum)
    assert ParameterCreationKind.NUMERIC_CONSTANT.value == "numeric_constant"
    assert ParameterCreationKind.FREE_BETA.value == "free_beta"
    assert ParameterCreationKind.FIXED_BETA.value == "fixed_beta"
    assert ParameterCreationKind.LOG_EXP_BETA.value == "log_exp_beta"
    assert ParameterCreationKind.BOUNDED_BETA.value == "bounded_beta"


def test_positivity_strategy_enum_values() -> None:
    assert issubclass(PositivityStrategy, str)
    assert issubclass(PositivityStrategy, Enum)
    assert PositivityStrategy.NONE.value == "none"
    assert PositivityStrategy.LOG_EXP.value == "log_exp"
    assert PositivityStrategy.LOWER_BOUND.value == "lower_bound"


def test_parameter_role_enum_values() -> None:
    assert issubclass(ParameterRole, str)
    assert issubclass(ParameterRole, Enum)
    assert ParameterRole.STRUCTURAL_COEFFICIENT.value == "structural_coefficient"
    assert ParameterRole.STRUCTURAL_SIGMA.value == "structural_sigma"
    assert ParameterRole.MEASUREMENT_INTERCEPT.value == "measurement_intercept"
    assert ParameterRole.MEASUREMENT_LOADING.value == "measurement_loading"
    assert ParameterRole.MEASUREMENT_SIGMA.value == "measurement_sigma"
    assert ParameterRole.THRESHOLD_FIRST.value == "threshold_first"
    assert ParameterRole.THRESHOLD_DELTA.value == "threshold_delta"


def test_cutpoint_kind_enum_values() -> None:
    assert issubclass(CutpointKind, str)
    assert issubclass(CutpointKind, Enum)
    assert CutpointKind.FREE.value == "free"
    assert CutpointKind.FIXED.value == "fixed"
    assert CutpointKind.DERIVED.value == "derived"


def test_resolved_parameter_fields_preserved() -> None:
    semantic_ref = dummy_ref("beta_time")
    notes = ["first", "second"]

    parameter = ResolvedParameter(
        semantic_ref=semantic_ref,
        final_name="beta_time",
        role=ParameterRole.STRUCTURAL_COEFFICIENT,
        status=ParameterStatus.FREE,
        fixed_value=None,
        initial_value=1.5,
        lower_bound=-2.0,
        upper_bound=3.0,
        positivity_strategy=PositivityStrategy.LOG_EXP,
        creation_kind=ParameterCreationKind.LOG_EXP_BETA,
        notes=notes,
    )

    assert parameter.semantic_ref is semantic_ref
    assert parameter.final_name == "beta_time"
    assert parameter.role is ParameterRole.STRUCTURAL_COEFFICIENT
    assert parameter.status is ParameterStatus.FREE
    assert parameter.fixed_value is None
    assert parameter.initial_value == 1.5
    assert parameter.lower_bound == -2.0
    assert parameter.upper_bound == 3.0
    assert parameter.positivity_strategy is PositivityStrategy.LOG_EXP
    assert parameter.creation_kind is ParameterCreationKind.LOG_EXP_BETA
    assert parameter.notes is notes


def test_resolved_parameter_notes_default_factory_gives_distinct_lists() -> None:
    first = ResolvedParameter(
        semantic_ref=None,
        final_name="p1",
        role=ParameterRole.MEASUREMENT_SIGMA,
        status=ParameterStatus.FIXED,
        fixed_value=1.0,
        initial_value=1.0,
        lower_bound=0.0,
        upper_bound=None,
        positivity_strategy=None,
        creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
    )
    second = ResolvedParameter(
        semantic_ref=None,
        final_name="p2",
        role=ParameterRole.MEASUREMENT_SIGMA,
        status=ParameterStatus.FIXED,
        fixed_value=2.0,
        initial_value=2.0,
        lower_bound=0.0,
        upper_bound=None,
        positivity_strategy=None,
        creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
    )

    assert first.notes == []
    assert second.notes == []
    assert first.notes is not second.notes


def test_resolved_parameter_is_frozen_and_slotted() -> None:
    parameter = ResolvedParameter(
        semantic_ref=None,
        final_name="p",
        role=ParameterRole.MEASUREMENT_INTERCEPT,
        status=ParameterStatus.FREE,
        fixed_value=None,
        initial_value=0.0,
        lower_bound=None,
        upper_bound=None,
        positivity_strategy=PositivityStrategy.NONE,
        creation_kind=ParameterCreationKind.FREE_BETA,
    )

    assert hasattr(ResolvedParameter, "__slots__")
    assert "__dict__" not in dir(parameter)

    with pytest.raises(FrozenInstanceError):
        parameter.final_name = "q"  # type: ignore[misc]


def test_resolved_parameter_ref_fields_and_defaults() -> None:
    semantic_ref = dummy_ref("alpha")

    ref = ResolvedParameterRef(final_name="alpha", semantic_ref=semantic_ref)
    ref_default = ResolvedParameterRef(final_name="beta")

    assert ref.final_name == "alpha"
    assert ref.semantic_ref is semantic_ref
    assert ref_default.final_name == "beta"
    assert ref_default.semantic_ref is None


def test_resolved_parameter_ref_is_frozen_and_slotted() -> None:
    ref = ResolvedParameterRef(final_name="alpha")

    assert hasattr(ResolvedParameterRef, "__slots__")
    assert "__dict__" not in dir(ref)

    with pytest.raises(FrozenInstanceError):
        ref.final_name = "beta"  # type: ignore[misc]


def test_resolved_constant_field_and_frozen_slots() -> None:
    constant = ResolvedConstant(value=3.14)

    assert constant.value == 3.14
    assert hasattr(ResolvedConstant, "__slots__")
    assert "__dict__" not in dir(constant)

    with pytest.raises(FrozenInstanceError):
        constant.value = 2.71  # type: ignore[misc]


def test_resolved_linear_term_accepts_parameter_ref() -> None:
    coefficient = ResolvedParameterRef(final_name="beta")

    term = ResolvedLinearTerm(
        coefficient=coefficient,
        variable_name="travel_time",
    )

    assert term.coefficient is coefficient
    assert term.variable_name == "travel_time"


def test_resolved_linear_term_accepts_constant() -> None:
    coefficient = ResolvedConstant(value=2.0)

    term = ResolvedLinearTerm(
        coefficient=coefficient,
        variable_name="cost",
    )

    assert term.coefficient is coefficient
    assert term.variable_name == "cost"


def test_resolved_linear_term_is_frozen_and_slotted() -> None:
    term = ResolvedLinearTerm(
        coefficient=ResolvedConstant(value=1.0),
        variable_name="x",
    )

    assert hasattr(ResolvedLinearTerm, "__slots__")
    assert "__dict__" not in dir(term)

    with pytest.raises(FrozenInstanceError):
        term.variable_name = "y"  # type: ignore[misc]


def test_resolved_linear_combination_fields_preserved() -> None:
    intercept = ResolvedParameterRef(final_name="alpha")
    terms = [
        ResolvedLinearTerm(
            coefficient=ResolvedParameterRef(final_name="beta1"),
            variable_name="x1",
        ),
        ResolvedLinearTerm(
            coefficient=ResolvedConstant(value=2.0),
            variable_name="x2",
        ),
    ]

    combo = ResolvedLinearCombination(
        intercept=intercept,
        terms=terms,
    )

    assert combo.intercept is intercept
    assert combo.terms is terms


def test_resolved_linear_combination_accepts_none_intercept() -> None:
    combo = ResolvedLinearCombination(
        intercept=None,
        terms=[],
    )

    assert combo.intercept is None
    assert combo.terms == []


def test_resolved_linear_combination_is_frozen_and_slotted() -> None:
    combo = ResolvedLinearCombination(
        intercept=None,
        terms=[],
    )

    assert hasattr(ResolvedLinearCombination, "__slots__")
    assert "__dict__" not in dir(combo)

    with pytest.raises(FrozenInstanceError):
        combo.intercept = ResolvedConstant(value=1.0)  # type: ignore[misc]


def test_resolved_structural_equation_fields_preserved() -> None:
    sigma = ResolvedParameterRef(final_name="sigma_lv")
    terms = [
        ResolvedLinearTerm(
            coefficient=ResolvedParameterRef(final_name="beta_income"),
            variable_name="income",
        )
    ]

    equation = ResolvedStructuralEquation(
        latent_name="LV1",
        expression_name="expr_lv1",
        terms=terms,
        sigma=sigma,
        draw_name="omega_lv1",
        draw_type="NORMAL_HALTON",
        error_distribution="gaussian",
    )

    assert equation.latent_name == "LV1"
    assert equation.expression_name == "expr_lv1"
    assert equation.terms is terms
    assert equation.sigma is sigma
    assert equation.draw_name == "omega_lv1"
    assert equation.draw_type == "NORMAL_HALTON"
    assert equation.error_distribution == "gaussian"


def test_resolved_structural_equation_accepts_none_sigma() -> None:
    equation = ResolvedStructuralEquation(
        latent_name="LV2",
        expression_name="expr_lv2",
        terms=[],
        sigma=None,
        draw_name="omega_lv2",
        draw_type="NORMAL",
        error_distribution="logistic",
    )

    assert equation.sigma is None
    assert equation.error_distribution == "logistic"


def test_resolved_structural_equation_is_frozen_and_slotted() -> None:
    equation = ResolvedStructuralEquation(
        latent_name="LV",
        expression_name="expr",
        terms=[],
        sigma=None,
        draw_name="omega",
        draw_type="NORMAL",
        error_distribution="gaussian",
    )

    assert hasattr(ResolvedStructuralEquation, "__slots__")
    assert "__dict__" not in dir(equation)

    with pytest.raises(FrozenInstanceError):
        equation.draw_name = "other"  # type: ignore[misc]


def test_resolved_cutpoint_fields_preserved() -> None:
    source_parameter_names = ["tau_1", "delta_2"]

    cutpoint = ResolvedCutpoint(
        symbol_name="tau_2",
        kind=CutpointKind.DERIVED,
        expression_text="tau_1 + delta_2",
        source_parameter_names=source_parameter_names,
    )

    assert cutpoint.symbol_name == "tau_2"
    assert cutpoint.kind is CutpointKind.DERIVED
    assert cutpoint.expression_text == "tau_1 + delta_2"
    assert cutpoint.source_parameter_names is source_parameter_names


def test_resolved_cutpoint_is_frozen_and_slotted() -> None:
    cutpoint = ResolvedCutpoint(
        symbol_name="tau_1",
        kind=CutpointKind.FREE,
        expression_text="0.0",
        source_parameter_names=[],
    )

    assert hasattr(ResolvedCutpoint, "__slots__")
    assert "__dict__" not in dir(cutpoint)

    with pytest.raises(FrozenInstanceError):
        cutpoint.symbol_name = "tau_2"  # type: ignore[misc]


def test_resolved_threshold_system_fields_preserved() -> None:
    cutpoints = [
        ResolvedCutpoint(
            symbol_name="tau_1",
            kind=CutpointKind.FREE,
            expression_text="0.0",
            source_parameter_names=[],
        )
    ]
    used_by = ["ind1", "ind2"]
    notes = ["normalized around zero"]

    system = ResolvedThresholdSystem(
        type_name="likert5",
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[3],
        construction_kind=ThresholdConstructionKind.SYMMETRIC,
        cutpoints=cutpoints,
        used_by_indicators=used_by,
        normalization_notes=notes,
    )

    assert system.type_name == "likert5"
    assert system.symmetric is True
    assert system.categories == [1, 2, 3, 4, 5]
    assert system.neutral_labels == [3]
    assert system.construction_kind is ThresholdConstructionKind.SYMMETRIC
    assert system.cutpoints is cutpoints
    assert system.used_by_indicators is used_by
    assert system.normalization_notes is notes


def test_resolved_threshold_system_is_frozen_and_slotted() -> None:
    system = ResolvedThresholdSystem(
        type_name="scale",
        symmetric=False,
        categories=[1, 2],
        neutral_labels=[],
        construction_kind=ThresholdConstructionKind.MONOTONE,
        cutpoints=[],
        used_by_indicators=[],
        normalization_notes=[],
    )

    assert hasattr(ResolvedThresholdSystem, "__slots__")
    assert "__dict__" not in dir(system)

    with pytest.raises(FrozenInstanceError):
        system.type_name = "other"  # type: ignore[misc]


def test_resolved_measurement_equation_fields_preserved() -> None:
    systematic_part = ResolvedLinearCombination(
        intercept=ResolvedParameterRef(final_name="alpha"),
        terms=[
            ResolvedLinearTerm(
                coefficient=ResolvedParameterRef(final_name="lambda_lv"),
                variable_name="LV1",
            )
        ],
    )
    sigma = ResolvedParameterRef(final_name="sigma_meas")
    notes = ["loading fixed for identification"]

    equation = ResolvedMeasurementEquation(
        indicator_name="ind1",
        statement="I like trains",
        type_name="likert5",
        measurement_model=MeasurementModel.ORDERED_PROBIT,
        systematic_part=systematic_part,
        sigma=sigma,
        observed_variable_name="obs_ind1",
        threshold_system_name="likert5",
        error_distribution=MeasurementErrorDistribution.GAUSSIAN,
        normalization_notes=notes,
    )

    assert equation.indicator_name == "ind1"
    assert equation.statement == "I like trains"
    assert equation.type_name == "likert5"
    assert equation.measurement_model is MeasurementModel.ORDERED_PROBIT
    assert equation.systematic_part is systematic_part
    assert equation.sigma is sigma
    assert equation.observed_variable_name == "obs_ind1"
    assert equation.threshold_system_name == "likert5"
    assert equation.error_distribution is MeasurementErrorDistribution.GAUSSIAN
    assert equation.normalization_notes is notes


def test_resolved_measurement_equation_accepts_none_sigma_and_threshold_name() -> None:
    equation = ResolvedMeasurementEquation(
        indicator_name="ind2",
        statement="Statement",
        type_name="continuous",
        measurement_model=MeasurementModel.GAUSSIAN,
        systematic_part=ResolvedLinearCombination(intercept=None, terms=[]),
        sigma=None,
        observed_variable_name="obs_ind2",
        threshold_system_name=None,
        error_distribution=MeasurementErrorDistribution.LOGISTIC,
        normalization_notes=[],
    )

    assert equation.sigma is None
    assert equation.threshold_system_name is None
    assert equation.error_distribution is MeasurementErrorDistribution.LOGISTIC


def test_resolved_measurement_equation_is_frozen_and_slotted() -> None:
    equation = ResolvedMeasurementEquation(
        indicator_name="ind",
        statement="s",
        type_name="t",
        measurement_model=MeasurementModel.ORDERED_LOGIT,
        systematic_part=ResolvedLinearCombination(intercept=None, terms=[]),
        sigma=None,
        observed_variable_name="obs",
        threshold_system_name="thr",
        error_distribution=MeasurementErrorDistribution.GAUSSIAN,
        normalization_notes=[],
    )

    assert hasattr(ResolvedMeasurementEquation, "__slots__")
    assert "__dict__" not in dir(equation)

    with pytest.raises(FrozenInstanceError):
        equation.type_name = "other"  # type: ignore[misc]


def test_resolved_normalization_rule_fields_preserved() -> None:
    rule = ResolvedNormalizationRule(
        scope="measurement",
        target_name="lambda_ind1",
        value="fixed_to_one",
        reason="reference indicator",
    )

    assert rule.scope == "measurement"
    assert rule.target_name == "lambda_ind1"
    assert rule.value == "fixed_to_one"
    assert rule.reason == "reference indicator"


def test_resolved_normalization_rule_accepts_float_value() -> None:
    rule = ResolvedNormalizationRule(
        scope="structural",
        target_name="sigma_lv",
        value=1.0,
        reason="scale normalization",
    )

    assert rule.value == 1.0


def test_resolved_normalization_rule_is_frozen_and_slotted() -> None:
    rule = ResolvedNormalizationRule(
        scope="s",
        target_name="t",
        value=0.0,
        reason="r",
    )

    assert hasattr(ResolvedNormalizationRule, "__slots__")
    assert "__dict__" not in dir(rule)

    with pytest.raises(FrozenInstanceError):
        rule.reason = "other"  # type: ignore[misc]


def test_resolved_normalization_summary_fields_preserved() -> None:
    rules = [
        ResolvedNormalizationRule(
            scope="measurement",
            target_name="alpha",
            value=0.0,
            reason="centering",
        )
    ]
    warnings = ["Potential over-identification"]

    summary = ResolvedNormalizationSummary(
        rules=rules,
        warnings=warnings,
        disclaimer="Automatically generated",
    )

    assert summary.rules is rules
    assert summary.warnings is warnings
    assert summary.disclaimer == "Automatically generated"


def test_resolved_normalization_summary_is_frozen_and_slotted() -> None:
    summary = ResolvedNormalizationSummary(
        rules=[],
        warnings=[],
        disclaimer="none",
    )

    assert hasattr(ResolvedNormalizationSummary, "__slots__")
    assert "__dict__" not in dir(summary)

    with pytest.raises(FrozenInstanceError):
        summary.disclaimer = "changed"  # type: ignore[misc]


def test_resolved_latent_variable_fields_preserved() -> None:
    structural_equation = ResolvedStructuralEquation(
        latent_name="LV1",
        expression_name="expr_lv1",
        terms=[],
        sigma=ResolvedParameterRef(final_name="sigma_lv1"),
        draw_name="omega",
        draw_type="NORMAL",
        error_distribution="gaussian",
    )
    indicator_names = ["ind1", "ind2"]
    notes = ["reference indicator: ind1"]

    latent = ResolvedLatentVariable(
        name="LV1",
        structural_equation=structural_equation,
        indicator_names=indicator_names,
        reference_indicator="ind1",
        normalization_notes=notes,
    )

    assert latent.name == "LV1"
    assert latent.structural_equation is structural_equation
    assert latent.indicator_names is indicator_names
    assert latent.reference_indicator == "ind1"
    assert latent.normalization_notes is notes


def test_resolved_latent_variable_accepts_none_reference_indicator() -> None:
    latent = ResolvedLatentVariable(
        name="LV2",
        structural_equation=ResolvedStructuralEquation(
            latent_name="LV2",
            expression_name="expr_lv2",
            terms=[],
            sigma=None,
            draw_name="omega2",
            draw_type="NORMAL",
            error_distribution="logistic",
        ),
        indicator_names=[],
        reference_indicator=None,
        normalization_notes=[],
    )

    assert latent.reference_indicator is None


def test_resolved_latent_variable_is_frozen_and_slotted() -> None:
    latent = ResolvedLatentVariable(
        name="LV",
        structural_equation=ResolvedStructuralEquation(
            latent_name="LV",
            expression_name="expr",
            terms=[],
            sigma=None,
            draw_name="omega",
            draw_type="NORMAL",
            error_distribution="gaussian",
        ),
        indicator_names=[],
        reference_indicator=None,
        normalization_notes=[],
    )

    assert hasattr(ResolvedLatentVariable, "__slots__")
    assert "__dict__" not in dir(latent)

    with pytest.raises(FrozenInstanceError):
        latent.name = "other"  # type: ignore[misc]


def test_resolved_model_metadata_fields_preserved() -> None:
    models_present = [MeasurementModel.GAUSSIAN, MeasurementModel.ORDERED_LOGIT]

    metadata = ResolvedModelMetadata(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        measurement_models_present=models_present,
        has_gaussian=True,
        has_ordered_probit=False,
        has_ordered_logit=True,
        has_ordinal=True,
        n_latent_variables=2,
        n_indicators=5,
        n_threshold_systems=1,
    )

    assert metadata.estimation_mode is EstimationMode.MAXIMUM_LIKELIHOOD
    assert metadata.measurement_models_present is models_present
    assert metadata.has_gaussian is True
    assert metadata.has_ordered_probit is False
    assert metadata.has_ordered_logit is True
    assert metadata.has_ordinal is True
    assert metadata.n_latent_variables == 2
    assert metadata.n_indicators == 5
    assert metadata.n_threshold_systems == 1


def test_resolved_model_metadata_is_frozen_and_slotted() -> None:
    metadata = ResolvedModelMetadata(
        estimation_mode=EstimationMode.BAYESIAN,
        measurement_models_present=[],
        has_gaussian=False,
        has_ordered_probit=False,
        has_ordered_logit=False,
        has_ordinal=False,
        n_latent_variables=0,
        n_indicators=0,
        n_threshold_systems=0,
    )

    assert hasattr(ResolvedModelMetadata, "__slots__")
    assert "__dict__" not in dir(metadata)

    with pytest.raises(FrozenInstanceError):
        metadata.n_indicators = 1  # type: ignore[misc]


def test_resolved_model_fields_preserved() -> None:
    metadata = ResolvedModelMetadata(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        measurement_models_present=[MeasurementModel.GAUSSIAN],
        has_gaussian=True,
        has_ordered_probit=False,
        has_ordered_logit=False,
        has_ordinal=False,
        n_latent_variables=1,
        n_indicators=1,
        n_threshold_systems=0,
    )
    latent_variables = {
        "LV1": ResolvedLatentVariable(
            name="LV1",
            structural_equation=ResolvedStructuralEquation(
                latent_name="LV1",
                expression_name="expr_lv1",
                terms=[],
                sigma=None,
                draw_name="omega",
                draw_type="NORMAL",
                error_distribution="gaussian",
            ),
            indicator_names=["ind1"],
            reference_indicator=None,
            normalization_notes=[],
        )
    }
    measurement_equations = {
        "ind1": ResolvedMeasurementEquation(
            indicator_name="ind1",
            statement="statement",
            type_name="continuous",
            measurement_model=MeasurementModel.GAUSSIAN,
            systematic_part=ResolvedLinearCombination(intercept=None, terms=[]),
            sigma=None,
            observed_variable_name="obs_ind1",
            threshold_system_name=None,
            error_distribution=MeasurementErrorDistribution.GAUSSIAN,
            normalization_notes=[],
        )
    }
    threshold_systems: dict[str, ResolvedThresholdSystem] = {}
    parameters = {
        "alpha": ResolvedParameter(
            semantic_ref=None,
            final_name="alpha",
            role=ParameterRole.MEASUREMENT_INTERCEPT,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=None,
            creation_kind=ParameterCreationKind.FREE_BETA,
        )
    }
    normalization = ResolvedNormalizationSummary(
        rules=[],
        warnings=[],
        disclaimer="none",
    )

    model = ResolvedModel(
        metadata=metadata,
        latent_variables=latent_variables,
        measurement_equations=measurement_equations,
        threshold_systems=threshold_systems,
        parameters=parameters,
        normalization=normalization,
    )

    assert model.metadata is metadata
    assert model.latent_variables is latent_variables
    assert model.measurement_equations is measurement_equations
    assert model.threshold_systems is threshold_systems
    assert model.parameters is parameters
    assert model.normalization is normalization


def test_resolved_model_is_frozen_and_slotted() -> None:
    model = ResolvedModel(
        metadata=ResolvedModelMetadata(
            estimation_mode=EstimationMode.BAYESIAN,
            measurement_models_present=[],
            has_gaussian=False,
            has_ordered_probit=False,
            has_ordered_logit=False,
            has_ordinal=False,
            n_latent_variables=0,
            n_indicators=0,
            n_threshold_systems=0,
        ),
        latent_variables={},
        measurement_equations={},
        threshold_systems={},
        parameters={},
        normalization=ResolvedNormalizationSummary(
            rules=[],
            warnings=[],
            disclaimer="d",
        ),
    )

    assert hasattr(ResolvedModel, "__slots__")
    assert "__dict__" not in dir(model)

    with pytest.raises(FrozenInstanceError):
        model.metadata = None  # type: ignore[misc]


def test_dataclass_equality_smoke_test_for_resolved_types() -> None:
    assert ResolvedConstant(1.0) == ResolvedConstant(1.0)
    assert ResolvedParameterRef("alpha") == ResolvedParameterRef("alpha")
    assert ResolvedLinearTerm(
        coefficient=ResolvedConstant(2.0),
        variable_name="x",
    ) == ResolvedLinearTerm(
        coefficient=ResolvedConstant(2.0),
        variable_name="x",
    )
    assert ResolvedCutpoint(
        symbol_name="tau_1",
        kind=CutpointKind.FREE,
        expression_text="0",
        source_parameter_names=[],
    ) == ResolvedCutpoint(
        symbol_name="tau_1",
        kind=CutpointKind.FREE,
        expression_text="0",
        source_parameter_names=[],
    )
