from __future__ import annotations

"""Resolution of a pure specification into a resolved semantic model."""

from dataclasses import dataclass

import numpy as np

from .context import BuildContext, PositivityMode
from .model_spec import (
    IndicatorMeasurementSpec,
    LatentVariable,
    LikertIndicator,
    LikertType,
    MeasurementConfiguration,
    MeasurementModel,
    PositiveParameterSpec,
)


def _positive_parameter_initial_value(
    spec: PositiveParameterSpec | None,
    *,
    default_start: float,
    context: BuildContext,
) -> float:
    """Return the initial value for a positive parameter.

    The specification expresses the start on the natural scale. When the
    positivity mode is `LOG_EXP`, the initial value returned here is the log of
    that natural-scale start so it can be used directly as the initial value of
    the unconstrained Biogeme parameter.

    :param spec: Optional positive-parameter specification.
    :param default_start: Default natural-scale start when no explicit start is
        provided.
    :param context: Build context determining the positivity mode.
    :return: Initial value to store in the resolved parameter.
    :raises ValueError: If the natural-scale start is not strictly positive.
    """
    natural_start = default_start if spec is None or spec.start is None else spec.start
    if natural_start <= 0:
        raise ValueError(
            f'Positive parameter starts must be strictly positive. Got {natural_start}.'
        )
    if context.positivity_mode == PositivityMode.LOG_EXP:
        return float(np.log(natural_start))
    return float(natural_start)


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
from .resolved import (
    CutpointKind,
    MeasurementErrorDistribution,
    ParameterCreationKind,
    ParameterRole,
    ParameterStatus,
    PositivityStrategy,
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
from .validation import validate_normalization_plan, validate_specification

_SMALL_POSITIVE = 1e-15


@dataclass(frozen=True, slots=True)
class _Prepared:
    latent_variables: list[LatentVariable]
    indicators: list[LikertIndicator]
    types: list[LikertType]
    indicator_by_name: dict[str, LikertIndicator]
    type_by_name: dict[str, LikertType]
    measurement_spec_by_indicator: dict[str, IndicatorMeasurementSpec]
    indicator_to_latents: dict[str, list[str]]
    ordinal_type_names: list[str]


def _prepare(
    *,
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
    measurement_configuration: MeasurementConfiguration,
    normalization_plan: NormalizationPlan | None,
) -> _Prepared:
    spec_validation = validate_specification(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
    )
    spec_validation.raise_for_errors()
    plan_validation = validate_normalization_plan(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        normalization_plan=normalization_plan,
    )
    plan_validation.raise_for_errors()

    indicator_by_name = {ind.name: ind for ind in likert_indicators}
    type_by_name = {lt.type_name: lt for lt in likert_types}
    measurement_spec_by_indicator = {
        spec.indicator_name: spec for spec in measurement_configuration.specifications
    }
    used_indicator_names = {
        indicator_name for lv in latent_variables for indicator_name in lv.indicators
    }
    missing_measurement_specs = sorted(
        name
        for name in used_indicator_names
        if name not in measurement_spec_by_indicator
    )
    if missing_measurement_specs:
        raise ValueError(
            'Missing measurement specification for indicator(s): '
            + ', '.join(missing_measurement_specs)
        )
    unknown_measurement_specs = sorted(
        name for name in measurement_spec_by_indicator if name not in indicator_by_name
    )
    if unknown_measurement_specs:
        raise ValueError(
            'Measurement specification refers to unknown indicator(s): '
            + ', '.join(unknown_measurement_specs)
        )
    used_indicators = [
        ind for ind in likert_indicators if ind.name in used_indicator_names
    ]
    indicator_to_latents: dict[str, list[str]] = {
        ind.name: [] for ind in used_indicators
    }
    for lv in latent_variables:
        for indicator_name in lv.indicators:
            indicator_to_latents[indicator_name].append(lv.name)
    ordinal_type_names = sorted(
        {
            ind.type_name
            for ind in used_indicators
            if measurement_spec_by_indicator[ind.name].measurement_model
            in {MeasurementModel.ORDERED_PROBIT, MeasurementModel.ORDERED_LOGIT}
        }
    )
    return _Prepared(
        latent_variables=list(latent_variables),
        indicators=used_indicators,
        types=list(likert_types),
        indicator_by_name=indicator_by_name,
        type_by_name=type_by_name,
        measurement_spec_by_indicator=measurement_spec_by_indicator,
        indicator_to_latents=indicator_to_latents,
        ordinal_type_names=ordinal_type_names,
    )


def _positivity_strategy(context: BuildContext) -> PositivityStrategy:
    return (
        PositivityStrategy.LOG_EXP
        if context.positivity_mode == PositivityMode.LOG_EXP
        else PositivityStrategy.LOWER_BOUND
    )


def _resolve_parameter(
    *,
    key: str,
    semantic_ref,
    final_name: str,
    role: ParameterRole,
    plan: NormalizationPlan | None,
    positivity: bool,
    context: BuildContext,
    initial_value: float,
    notes: list[str],
) -> ResolvedParameter:
    fixed_value = (
        plan.get(semantic_ref)
        if (plan is not None and semantic_ref is not None)
        else None
    )
    if fixed_value is not None:
        creation_kind = ParameterCreationKind.NUMERIC_CONSTANT
        return ResolvedParameter(
            semantic_ref=semantic_ref,
            final_name=final_name,
            role=role,
            status=ParameterStatus.FIXED,
            fixed_value=float(fixed_value),
            initial_value=float(fixed_value),
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=None,
            creation_kind=creation_kind,
            notes=notes,
        )
    if positivity:
        strategy = _positivity_strategy(context)
        if strategy == PositivityStrategy.LOG_EXP:
            creation_kind = ParameterCreationKind.LOG_EXP_BETA
            return ResolvedParameter(
                semantic_ref=semantic_ref,
                final_name=final_name,
                role=role,
                status=ParameterStatus.FREE,
                fixed_value=None,
                initial_value=initial_value,
                lower_bound=None,
                upper_bound=None,
                positivity_strategy=strategy,
                creation_kind=creation_kind,
                notes=notes,
            )
        creation_kind = ParameterCreationKind.BOUNDED_BETA
        return ResolvedParameter(
            semantic_ref=semantic_ref,
            final_name=final_name,
            role=role,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=max(initial_value, 1.0),
            lower_bound=_SMALL_POSITIVE,
            upper_bound=None,
            positivity_strategy=strategy,
            creation_kind=creation_kind,
            notes=notes,
        )
    creation_kind = ParameterCreationKind.FREE_BETA
    return ResolvedParameter(
        semantic_ref=semantic_ref,
        final_name=final_name,
        role=role,
        status=ParameterStatus.FREE,
        fixed_value=None,
        initial_value=initial_value,
        lower_bound=None,
        upper_bound=None,
        positivity_strategy=PositivityStrategy.NONE,
        creation_kind=creation_kind,
        notes=notes,
    )


def _parameter_ref(param: ResolvedParameter) -> ResolvedParameterRef:
    return ResolvedParameterRef(
        final_name=param.final_name, semantic_ref=param.semantic_ref
    )


def _resolve_structural_parameters(
    prepared: _Prepared, context: BuildContext, plan: NormalizationPlan | None
) -> dict[str, ResolvedParameter]:
    params: dict[str, ResolvedParameter] = {}
    for lv in prepared.latent_variables:
        if lv.structural_equation.intercept:
            intercept_ref = StructuralIntercept(lv.name)
            intercept_name = context.naming.structural_intercept_name(lv.name)
            params[intercept_name] = _resolve_parameter(
                key=intercept_name,
                semantic_ref=intercept_ref,
                final_name=intercept_name,
                role=ParameterRole.STRUCTURAL_INTERCEPT,
                plan=plan,
                positivity=False,
                context=context,
                initial_value=0.0,
                notes=[f"Structural intercept for latent '{lv.name}'."],
            )
        for variable_name in lv.structural_equation.explanatory_variables:
            ref = StructuralCoefficient(lv.name, variable_name)
            final_name = context.naming.structural_beta_name(lv.name, variable_name)
            params[final_name] = _resolve_parameter(
                key=final_name,
                semantic_ref=ref,
                final_name=final_name,
                role=ParameterRole.STRUCTURAL_COEFFICIENT,
                plan=plan,
                positivity=False,
                context=context,
                initial_value=0.0,
                notes=[
                    f"Structural coefficient for latent '{lv.name}' and variable '{variable_name}'."
                ],
            )
        sigma_ref = StructuralSigma(lv.name)
        sigma_name = context.naming.structural_sigma_name(lv.name)
        params[sigma_name] = _resolve_parameter(
            key=sigma_name,
            semantic_ref=sigma_ref,
            final_name=sigma_name,
            role=ParameterRole.STRUCTURAL_SIGMA,
            plan=plan,
            positivity=True,
            context=context,
            initial_value=_positive_parameter_initial_value(
                lv.structural_sigma,
                default_start=10.0,
                context=context,
            ),
            notes=[f"Structural sigma for latent '{lv.name}'."],
        )
    return params


def _resolve_measurement_parameters(
    prepared: _Prepared, context: BuildContext, plan: NormalizationPlan | None
) -> dict[str, ResolvedParameter]:
    params: dict[str, ResolvedParameter] = {}
    for ind in prepared.indicators:
        intercept_ref = MeasurementIntercept(ind.name)
        intercept_name = context.naming.measurement_intercept_name(ind.name)
        params[intercept_name] = _resolve_parameter(
            key=intercept_name,
            semantic_ref=intercept_ref,
            final_name=intercept_name,
            role=ParameterRole.MEASUREMENT_INTERCEPT,
            plan=plan,
            positivity=False,
            context=context,
            initial_value=0.0,
            notes=[f"Measurement intercept for indicator '{ind.name}'."],
        )
        sigma_ref = MeasurementSigma(ind.name)
        sigma_name = context.naming.measurement_sigma_name(ind.name)
        measurement_spec = prepared.measurement_spec_by_indicator[ind.name]
        params[sigma_name] = _resolve_parameter(
            key=sigma_name,
            semantic_ref=sigma_ref,
            final_name=sigma_name,
            role=ParameterRole.MEASUREMENT_SIGMA,
            plan=plan,
            positivity=True,
            context=context,
            initial_value=_positive_parameter_initial_value(
                measurement_spec.measurement_sigma,
                default_start=10.0,
                context=context,
            ),
            notes=[f"Measurement sigma for indicator '{ind.name}'."],
        )
        for latent_name in prepared.indicator_to_latents[ind.name]:
            loading_ref = MeasurementLoading(latent_name, ind.name)
            loading_name = context.naming.measurement_loading_name(
                latent_name, ind.name
            )
            params[loading_name] = _resolve_parameter(
                key=loading_name,
                semantic_ref=loading_ref,
                final_name=loading_name,
                role=ParameterRole.MEASUREMENT_LOADING,
                plan=plan,
                positivity=False,
                context=context,
                initial_value=0.0,
                notes=[
                    f"Measurement loading linking latent '{latent_name}' to indicator '{ind.name}'."
                ],
            )
    return params


def _resolve_threshold_parameters(
    prepared: _Prepared, context: BuildContext, plan: NormalizationPlan | None
) -> dict[str, ResolvedParameter]:
    params: dict[str, ResolvedParameter] = {}
    for type_name in prepared.ordinal_type_names:
        lt = prepared.type_by_name[type_name]
        n_tau = len(lt.categories) - 1
        if lt.symmetric:
            n_deltas = n_tau // 2
            for index in range(n_deltas):
                ref = ThresholdDelta(type_name, index)
                name = context.naming.threshold_delta_name(type_name, index)
                params[name] = _resolve_parameter(
                    key=name,
                    semantic_ref=ref,
                    final_name=name,
                    role=ParameterRole.THRESHOLD_DELTA,
                    plan=plan,
                    positivity=True,
                    context=context,
                    initial_value=(
                        -0.86 + 0.43 * index
                        if context.positivity_mode == PositivityMode.LOG_EXP
                        else 0.5
                    ),
                    notes=[
                        f"Symmetric threshold delta {index} for type '{type_name}'."
                    ],
                )
        else:
            tau1_ref = ThresholdFirst(type_name)
            tau1_name = context.naming.threshold_tau1_name(type_name)
            params[tau1_name] = _resolve_parameter(
                key=tau1_name,
                semantic_ref=tau1_ref,
                final_name=tau1_name,
                role=ParameterRole.THRESHOLD_FIRST,
                plan=plan,
                positivity=False,
                context=context,
                initial_value=0.0,
                notes=[f"First threshold for type '{type_name}'."],
            )
            for index in range(1, n_tau):
                ref = ThresholdDelta(type_name, index)
                name = context.naming.threshold_delta_name(type_name, index)
                params[name] = _resolve_parameter(
                    key=name,
                    semantic_ref=ref,
                    final_name=name,
                    role=ParameterRole.THRESHOLD_DELTA,
                    plan=plan,
                    positivity=True,
                    context=context,
                    initial_value=(
                        0.3 + 0.5 * (index - 1)
                        if context.positivity_mode == PositivityMode.LOG_EXP
                        else 1.0
                    ),
                    notes=[f"Monotone threshold delta {index} for type '{type_name}'."],
                )
    return params


def _resolve_threshold_systems(
    prepared: _Prepared, context: BuildContext, params: dict[str, ResolvedParameter]
) -> dict[str, ResolvedThresholdSystem]:
    systems: dict[str, ResolvedThresholdSystem] = {}
    for type_name in prepared.ordinal_type_names:
        lt = prepared.type_by_name[type_name]
        n_tau = len(lt.categories) - 1
        used_by = sorted(
            ind.name
            for ind in prepared.indicators
            if ind.type_name == type_name
            and prepared.measurement_spec_by_indicator[ind.name].measurement_model
            in {MeasurementModel.ORDERED_PROBIT, MeasurementModel.ORDERED_LOGIT}
        )
        cutpoints: list[ResolvedCutpoint] = []
        notes: list[str] = []
        if lt.symmetric:
            construction = ThresholdConstructionKind.SYMMETRIC
            n_deltas = n_tau // 2
            delta_names = [
                context.naming.threshold_delta_name(type_name, i)
                for i in range(n_deltas)
            ]
            center_index = n_tau // 2
            for idx in range(n_tau):
                if n_tau % 2 == 1 and idx == center_index:
                    cutpoints.append(
                        ResolvedCutpoint(
                            f'tau_{idx + 1}', CutpointKind.DERIVED, '0.0', []
                        )
                    )
                elif idx < center_index:
                    involved = delta_names[: center_index - idx]
                    expr = ' + '.join(involved)
                    cutpoints.append(
                        ResolvedCutpoint(
                            f'tau_{idx + 1}',
                            CutpointKind.DERIVED,
                            f'-({expr})' if len(involved) > 1 else f'-{expr}',
                            involved,
                        )
                    )
                else:
                    positive_index = idx - center_index
                    if n_tau % 2 == 1:
                        positive_index -= 1
                    involved = delta_names[: positive_index + 1]
                    expr = ' + '.join(involved)
                    cutpoints.append(
                        ResolvedCutpoint(
                            f'tau_{idx + 1}', CutpointKind.DERIVED, expr, involved
                        )
                    )
            notes.append(f"Symmetric threshold construction for type '{type_name}'.")
        else:
            construction = ThresholdConstructionKind.MONOTONE
            tau1_name = context.naming.threshold_tau1_name(type_name)
            tau1_param = params[tau1_name]
            cutpoints.append(
                ResolvedCutpoint(
                    'tau_1',
                    (
                        CutpointKind.FIXED
                        if tau1_param.status == ParameterStatus.FIXED
                        else CutpointKind.FREE
                    ),
                    (
                        tau1_name
                        if tau1_param.status == ParameterStatus.FREE
                        else str(tau1_param.fixed_value)
                    ),
                    [tau1_name] if tau1_param.status == ParameterStatus.FREE else [],
                )
            )
            previous = 'tau_1'
            for index in range(1, n_tau):
                delta_name = context.naming.threshold_delta_name(type_name, index)
                symbol_name = f'tau_{index + 1}'
                cutpoints.append(
                    ResolvedCutpoint(
                        symbol_name,
                        CutpointKind.DERIVED,
                        f'{previous} + {delta_name}',
                        [previous, delta_name],
                    )
                )
                previous = symbol_name
            notes.append(f"Monotone threshold construction for type '{type_name}'.")
        systems[type_name] = ResolvedThresholdSystem(
            type_name=type_name,
            symmetric=lt.symmetric,
            categories=list(lt.categories),
            neutral_labels=list(lt.neutral_labels),
            construction_kind=construction,
            cutpoints=cutpoints,
            used_by_indicators=used_by,
            normalization_notes=notes,
        )
    return systems


def _resolve_structural_equations(
    prepared: _Prepared, context: BuildContext, params: dict[str, ResolvedParameter]
) -> dict[str, ResolvedStructuralEquation]:
    equations: dict[str, ResolvedStructuralEquation] = {}
    for lv in prepared.latent_variables:
        terms: list[ResolvedLinearTerm] = []
        for variable_name in lv.structural_equation.explanatory_variables:
            name = context.naming.structural_beta_name(lv.name, variable_name)
            terms.append(
                ResolvedLinearTerm(_parameter_ref(params[name]), variable_name)
            )
        intercept = None
        if lv.structural_equation.intercept:
            intercept_name = context.naming.structural_intercept_name(lv.name)
            intercept = _parameter_ref(params[intercept_name])
        sigma_name = context.naming.structural_sigma_name(lv.name)
        equations[lv.name] = ResolvedStructuralEquation(
            latent_name=lv.name,
            expression_name=lv.name,
            systematic_part=ResolvedLinearCombination(intercept, terms),
            sigma=_parameter_ref(params[sigma_name]),
            draw_name=context.naming.structural_draw_name(lv.name),
            draw_type=context.draw_type,
            error_distribution='normal',
        )
    return equations


def _resolve_measurement_equations(
    prepared: _Prepared, context: BuildContext, params: dict[str, ResolvedParameter]
) -> dict[str, ResolvedMeasurementEquation]:
    equations: dict[str, ResolvedMeasurementEquation] = {}
    for ind in prepared.indicators:
        intercept_name = context.naming.measurement_intercept_name(ind.name)
        intercept_param = params[intercept_name]
        terms: list[ResolvedLinearTerm] = []
        for latent_name in prepared.indicator_to_latents[ind.name]:
            loading_name = context.naming.measurement_loading_name(
                latent_name, ind.name
            )
            terms.append(
                ResolvedLinearTerm(_parameter_ref(params[loading_name]), latent_name)
            )
        sigma_name = context.naming.measurement_sigma_name(ind.name)
        measurement_spec = prepared.measurement_spec_by_indicator[ind.name]
        model = measurement_spec.measurement_model
        if model == MeasurementModel.GAUSSIAN:
            distribution = MeasurementErrorDistribution.GAUSSIAN
            threshold_system_name = None
        elif model == MeasurementModel.ORDERED_PROBIT:
            distribution = MeasurementErrorDistribution.GAUSSIAN
            threshold_system_name = ind.type_name
        else:
            distribution = MeasurementErrorDistribution.LOGISTIC
            threshold_system_name = ind.type_name
        equations[ind.name] = ResolvedMeasurementEquation(
            indicator_name=ind.name,
            statement=ind.statement,
            type_name=ind.type_name,
            measurement_model=model,
            systematic_part=ResolvedLinearCombination(
                _parameter_ref(intercept_param), terms
            ),
            sigma=_parameter_ref(params[sigma_name]),
            observed_variable_name=ind.name,
            threshold_system_name=threshold_system_name,
            error_distribution=distribution,
            normalization_notes=[],
        )
    return equations


def _resolve_normalization_summary(
    prepared: _Prepared,
    plan: NormalizationPlan | None,
    latent_variables: dict[str, ResolvedLatentVariable],
) -> ResolvedNormalizationSummary:
    rules: list[ResolvedNormalizationRule] = []
    warnings: list[str] = []
    if plan is not None:
        for fixing in plan:
            rules.append(
                ResolvedNormalizationRule(
                    scope=fixing.target.__class__.__name__,
                    target_name=repr(fixing.target),
                    value=fixing.value,
                    reason=fixing.note or 'Explicit normalization fixing.',
                )
            )
    for lv in prepared.latent_variables:
        if latent_variables[lv.name].reference_indicator is None:
            warnings.append(
                f"No obvious reference indicator could be inferred for latent variable '{lv.name}'."
            )
    disclaimer = (
        'The suggested or explicit normalizations are provided as modeling guidance. '
        'Every specific model may require additional reasoning about identification.'
    )
    return ResolvedNormalizationSummary(
        rules=rules, warnings=warnings, disclaimer=disclaimer
    )


def resolve_model(
    *,
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
    measurement_configuration: MeasurementConfiguration,
    context: BuildContext,
    normalization_plan: NormalizationPlan | None = None,
) -> ResolvedModel:
    prepared = _prepare(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        measurement_configuration=measurement_configuration,
        normalization_plan=normalization_plan,
    )
    parameters: dict[str, ResolvedParameter] = {}
    parameters.update(
        _resolve_structural_parameters(prepared, context, normalization_plan)
    )
    parameters.update(
        _resolve_measurement_parameters(prepared, context, normalization_plan)
    )
    parameters.update(
        _resolve_threshold_parameters(prepared, context, normalization_plan)
    )
    threshold_systems = _resolve_threshold_systems(prepared, context, parameters)
    structural_equations = _resolve_structural_equations(prepared, context, parameters)
    measurement_equations = _resolve_measurement_equations(
        prepared, context, parameters
    )

    resolved_latents: dict[str, ResolvedLatentVariable] = {}
    for lv in prepared.latent_variables:
        reference_indicator = None
        for indicator_name in lv.indicators:
            intercept_fixed = (
                normalization_plan.is_fixed(MeasurementIntercept(indicator_name))
                if normalization_plan is not None
                else False
            )
            loading_fixed = (
                normalization_plan.is_fixed(MeasurementLoading(lv.name, indicator_name))
                if normalization_plan is not None
                else False
            )
            if intercept_fixed and loading_fixed:
                reference_indicator = indicator_name
                break
        notes = []
        if reference_indicator is not None:
            notes.append(
                f"Reference indicator inferred from normalization plan: '{reference_indicator}'."
            )
        resolved_latents[lv.name] = ResolvedLatentVariable(
            name=lv.name,
            structural_equation=structural_equations[lv.name],
            indicator_names=sorted(
                indicator_name
                for indicator_name in lv.indicators
                if indicator_name in prepared.indicator_to_latents
            ),
            reference_indicator=reference_indicator,
            normalization_notes=notes,
        )

    normalization = _resolve_normalization_summary(
        prepared, normalization_plan, resolved_latents
    )
    measurement_models_present = sorted(
        {
            prepared.measurement_spec_by_indicator[indicator_name].measurement_model
            for indicator_name in measurement_equations
        },
        key=lambda x: x.value,
    )
    metadata = ResolvedModelMetadata(
        estimation_mode=context.estimation_mode,
        measurement_models_present=measurement_models_present,
        has_gaussian=MeasurementModel.GAUSSIAN in measurement_models_present,
        has_ordered_probit=MeasurementModel.ORDERED_PROBIT
        in measurement_models_present,
        has_ordered_logit=MeasurementModel.ORDERED_LOGIT in measurement_models_present,
        has_ordinal=any(
            m in {MeasurementModel.ORDERED_PROBIT, MeasurementModel.ORDERED_LOGIT}
            for m in measurement_models_present
        ),
        n_latent_variables=len(resolved_latents),
        n_indicators=len(measurement_equations),
        n_threshold_systems=len(threshold_systems),
    )
    return ResolvedModel(
        metadata=metadata,
        latent_variables=resolved_latents,
        measurement_equations=measurement_equations,
        threshold_systems=threshold_systems,
        parameters=parameters,
        normalization=normalization,
    )
