from __future__ import annotations

import math

import pytest

from biogeme.latent_variables import resolver
from biogeme.latent_variables.context import (
    BuildContext,
    EstimationMode,
    PositivityMode,
)
from biogeme.latent_variables.model_spec import (
    IndicatorMeasurementSpec,
    LatentVariable,
    LikertIndicator,
    LikertType,
    MeasurementConfiguration,
    MeasurementModel,
    PositiveParameterSpec,
    StructuralEquation,
)
from biogeme.latent_variables.naming import DefaultNamingPolicy
from biogeme.latent_variables.normalization_plan import Fixing, NormalizationPlan
from biogeme.latent_variables.normalization_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
    StructuralCoefficient,
    StructuralIntercept,
    StructuralSigma,
    ThresholdDelta,
    ThresholdFirst,
)
from biogeme.latent_variables.resolved import (
    CutpointKind,
    MeasurementErrorDistribution,
    ParameterCreationKind,
    ParameterRole,
    ParameterStatus,
    PositivityStrategy,
    ThresholdConstructionKind,
)


def make_context(
    *,
    estimation_mode: EstimationMode = EstimationMode.MAXIMUM_LIKELIHOOD,
    positivity_mode: PositivityMode = PositivityMode.LOG_EXP,
    draw_type: str = 'NORMAL_TEST',
) -> BuildContext:
    return BuildContext(
        estimation_mode=estimation_mode,
        draw_type=draw_type,
        positivity_mode=positivity_mode,
        naming=DefaultNamingPolicy(),
    )


def make_latent(
    name: str,
    variables: list[str],
    indicators: list[str],
    structural_sigma: PositiveParameterSpec | None = None,
) -> LatentVariable:
    return LatentVariable(
        name=name,
        structural_equation=StructuralEquation(
            name=name,
            explanatory_variables=variables,
        ),
        indicators=indicators,
        structural_sigma=structural_sigma,
    )


def make_indicator(name: str, type_name: str) -> LikertIndicator:
    return LikertIndicator(
        name=name,
        statement=f'Statement for {name}',
        type_name=type_name,
    )


def make_measurement_configuration(
    *specs: IndicatorMeasurementSpec,
) -> MeasurementConfiguration:
    return MeasurementConfiguration(specifications=list(specs))


class _ValidationResult:
    def __init__(self) -> None:
        self.called = False

    def raise_for_errors(self) -> None:
        self.called = True


def test_positive_parameter_initial_value_with_spec_none_and_log_exp() -> None:
    context = make_context(positivity_mode=PositivityMode.LOG_EXP)

    result = resolver._positive_parameter_initial_value(
        None,
        default_start=10.0,
        context=context,
    )

    assert result == pytest.approx(math.log(10.0))


def test_positive_parameter_initial_value_with_spec_start_none_and_lower_bound() -> (
    None
):
    context = make_context(positivity_mode=PositivityMode.LOWER_BOUND)
    spec = PositiveParameterSpec(start=None, lower_bound=0.2)

    result = resolver._positive_parameter_initial_value(
        spec,
        default_start=7.5,
        context=context,
    )

    assert result == 7.5


def test_positive_parameter_initial_value_with_explicit_start_and_log_exp() -> None:
    context = make_context(positivity_mode=PositivityMode.LOG_EXP)
    spec = PositiveParameterSpec(start=2.5, lower_bound=0.1)

    result = resolver._positive_parameter_initial_value(
        spec,
        default_start=10.0,
        context=context,
    )

    assert result == pytest.approx(math.log(2.5))


@pytest.mark.parametrize('bad_start', [0.0, -1.0])
def test_positive_parameter_initial_value_raises_for_nonpositive_start(
    bad_start: float,
) -> None:
    context = make_context()
    spec = PositiveParameterSpec(start=bad_start, lower_bound=0.0)

    with pytest.raises(
        ValueError,
        match=r'Positive parameter starts must be strictly positive\.',
    ):
        resolver._positive_parameter_initial_value(
            spec,
            default_start=10.0,
            context=context,
        )


def test_prepare_calls_validators_and_builds_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_result = _ValidationResult()
    plan_result = _ValidationResult()

    def fake_validate_specification(**kwargs):
        assert [lv.name for lv in kwargs['latent_variables']] == ['LV1', 'LV2']
        return spec_result

    def fake_validate_normalization_plan(**kwargs):
        assert kwargs['normalization_plan'] is None
        return plan_result

    monkeypatch.setattr(resolver, 'validate_specification', fake_validate_specification)
    monkeypatch.setattr(
        resolver,
        'validate_normalization_plan',
        fake_validate_normalization_plan,
    )

    latent_variables = [
        make_latent('LV1', ['x1'], ['ind_g', 'ind_o']),
        make_latent('LV2', ['x2'], ['ind_o']),
    ]
    likert_indicators = [
        make_indicator('ind_g', 'continuous'),
        make_indicator('ind_o', 'ord5'),
    ]
    likert_types = [
        LikertType(
            type_name='continuous',
            symmetric=False,
            categories=[0, 1],
            neutral_labels=[],
        ),
        LikertType(
            type_name='ord5',
            symmetric=True,
            categories=[1, 2, 3, 4, 5],
            neutral_labels=[3],
        ),
    ]
    measurement_configuration = make_measurement_configuration(
        IndicatorMeasurementSpec(
            indicator_name='ind_g',
            measurement_model=MeasurementModel.GAUSSIAN,
        ),
        IndicatorMeasurementSpec(
            indicator_name='ind_o',
            measurement_model=MeasurementModel.ORDERED_PROBIT,
        ),
    )

    prepared = resolver._prepare(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        measurement_configuration=measurement_configuration,
        normalization_plan=None,
    )

    assert spec_result.called is True
    assert plan_result.called is True
    assert [lv.name for lv in prepared.latent_variables] == ['LV1', 'LV2']
    assert [ind.name for ind in prepared.indicators] == ['ind_g', 'ind_o']
    assert [lt.type_name for lt in prepared.types] == ['continuous', 'ord5']
    assert prepared.indicator_by_name['ind_g'].name == 'ind_g'
    assert prepared.type_by_name['ord5'].type_name == 'ord5'
    assert (
        prepared.measurement_spec_by_indicator['ind_o'].measurement_model
        is MeasurementModel.ORDERED_PROBIT
    )
    assert prepared.indicator_to_latents == {
        'ind_g': ['LV1'],
        'ind_o': ['LV1', 'LV2'],
    }
    assert prepared.ordinal_type_names == ['ord5']


def test_prepare_raises_for_missing_measurement_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolver,
        'validate_specification',
        lambda **kwargs: _ValidationResult(),
    )
    monkeypatch.setattr(
        resolver,
        'validate_normalization_plan',
        lambda **kwargs: _ValidationResult(),
    )

    with pytest.raises(
        ValueError,
        match=r'Missing measurement specification for indicator\(s\): ind2',
    ):
        resolver._prepare(
            latent_variables=[make_latent('LV1', [], ['ind1', 'ind2'])],
            likert_indicators=[
                make_indicator('ind1', 't'),
                make_indicator('ind2', 't'),
            ],
            likert_types=[
                LikertType(
                    type_name='t',
                    symmetric=False,
                    categories=[1, 2],
                    neutral_labels=[],
                )
            ],
            measurement_configuration=make_measurement_configuration(
                IndicatorMeasurementSpec(
                    indicator_name='ind1',
                    measurement_model=MeasurementModel.GAUSSIAN,
                )
            ),
            normalization_plan=None,
        )


def test_prepare_raises_for_unknown_measurement_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolver,
        'validate_specification',
        lambda **kwargs: _ValidationResult(),
    )
    monkeypatch.setattr(
        resolver,
        'validate_normalization_plan',
        lambda **kwargs: _ValidationResult(),
    )

    with pytest.raises(
        ValueError,
        match=r'Measurement specification refers to unknown indicator\(s\): ghost',
    ):
        resolver._prepare(
            latent_variables=[make_latent('LV1', [], ['ind1'])],
            likert_indicators=[make_indicator('ind1', 't')],
            likert_types=[
                LikertType(
                    type_name='t',
                    symmetric=False,
                    categories=[1, 2],
                    neutral_labels=[],
                )
            ],
            measurement_configuration=make_measurement_configuration(
                IndicatorMeasurementSpec(
                    indicator_name='ind1',
                    measurement_model=MeasurementModel.GAUSSIAN,
                ),
                IndicatorMeasurementSpec(
                    indicator_name='ghost',
                    measurement_model=MeasurementModel.GAUSSIAN,
                ),
            ),
            normalization_plan=None,
        )


def test_positivity_strategy_both_modes() -> None:
    assert (
        resolver._positivity_strategy(
            make_context(positivity_mode=PositivityMode.LOG_EXP)
        )
        is PositivityStrategy.LOG_EXP
    )
    assert (
        resolver._positivity_strategy(
            make_context(positivity_mode=PositivityMode.LOWER_BOUND)
        )
        is PositivityStrategy.LOWER_BOUND
    )


def test_resolve_parameter_fixed_value_from_plan() -> None:
    context = make_context()
    plan = NormalizationPlan([Fixing(target=MeasurementIntercept('ind1'), value=2.0)])

    result = resolver._resolve_parameter(
        key='unused',
        semantic_ref=MeasurementIntercept('ind1'),
        final_name='alpha_ind1',
        role=ParameterRole.MEASUREMENT_INTERCEPT,
        plan=plan,
        positivity=False,
        context=context,
        initial_value=99.0,
        notes=['fixed note'],
    )

    assert result.status is ParameterStatus.FIXED
    assert result.fixed_value == 2.0
    assert result.initial_value == 2.0
    assert result.lower_bound is None
    assert result.upper_bound is None
    assert result.positivity_strategy is None
    assert result.creation_kind is ParameterCreationKind.NUMERIC_CONSTANT
    assert result.notes == ['fixed note']


def test_resolve_parameter_positive_log_exp_branch() -> None:
    context = make_context(positivity_mode=PositivityMode.LOG_EXP)

    result = resolver._resolve_parameter(
        key='unused',
        semantic_ref=StructuralSigma('LV1'),
        final_name='sigma_lv1',
        role=ParameterRole.STRUCTURAL_SIGMA,
        plan=None,
        positivity=True,
        context=context,
        initial_value=1.7,
        notes=['positive'],
    )

    assert result.status is ParameterStatus.FREE
    assert result.fixed_value is None
    assert result.initial_value == 1.7
    assert result.lower_bound is None
    assert result.upper_bound is None
    assert result.positivity_strategy is PositivityStrategy.LOG_EXP
    assert result.creation_kind is ParameterCreationKind.LOG_EXP_BETA


def test_resolve_parameter_positive_lower_bound_branch_uses_small_positive_floor() -> (
    None
):
    context = make_context(positivity_mode=PositivityMode.LOWER_BOUND)

    result = resolver._resolve_parameter(
        key='unused',
        semantic_ref=StructuralSigma('LV1'),
        final_name='sigma_lv1',
        role=ParameterRole.STRUCTURAL_SIGMA,
        plan=None,
        positivity=True,
        context=context,
        initial_value=0.3,
        notes=['positive'],
    )

    assert result.status is ParameterStatus.FREE
    assert result.initial_value == 1.0
    assert result.lower_bound == resolver._SMALL_POSITIVE
    assert result.upper_bound is None
    assert result.positivity_strategy is PositivityStrategy.LOWER_BOUND
    assert result.creation_kind is ParameterCreationKind.BOUNDED_BETA


def test_resolve_parameter_free_nonpositive_branch() -> None:
    context = make_context()

    result = resolver._resolve_parameter(
        key='unused',
        semantic_ref=StructuralCoefficient('LV1', 'x1'),
        final_name='beta_lv1_x1',
        role=ParameterRole.STRUCTURAL_COEFFICIENT,
        plan=None,
        positivity=False,
        context=context,
        initial_value=-3.0,
        notes=['free'],
    )

    assert result.status is ParameterStatus.FREE
    assert result.fixed_value is None
    assert result.initial_value == -3.0
    assert result.lower_bound is None
    assert result.upper_bound is None
    assert result.positivity_strategy is PositivityStrategy.NONE
    assert result.creation_kind is ParameterCreationKind.FREE_BETA


def test_parameter_ref_wraps_resolved_parameter() -> None:
    parameter = resolver.ResolvedParameter(
        semantic_ref=MeasurementSigma('ind1'),
        final_name='sigma_ind1',
        role=ParameterRole.MEASUREMENT_SIGMA,
        status=ParameterStatus.FREE,
        fixed_value=None,
        initial_value=0.0,
        lower_bound=None,
        upper_bound=None,
        positivity_strategy=PositivityStrategy.NONE,
        creation_kind=ParameterCreationKind.FREE_BETA,
        notes=[],
    )

    ref = resolver._parameter_ref(parameter)

    assert ref.final_name == 'sigma_ind1'
    assert ref.semantic_ref == MeasurementSigma('ind1')


def test_resolve_structural_parameters_builds_intercepts_coefficients_and_sigmas() -> (
    None
):
    prepared = resolver._Prepared(
        latent_variables=[
            make_latent('LV1', ['x1', 'x2'], ['ind1']),
            make_latent(
                'LV2', [], ['ind2'], PositiveParameterSpec(start=2.0, lower_bound=0.1)
            ),
        ],
        indicators=[],
        types=[],
        indicator_by_name={},
        type_by_name={},
        measurement_spec_by_indicator={},
        indicator_to_latents={},
        ordinal_type_names=[],
    )
    context = make_context(positivity_mode=PositivityMode.LOG_EXP)

    params = resolver._resolve_structural_parameters(prepared, context, None)

    assert set(params) == {
        'struct_LV1_intercept',
        'struct_LV1_x1',
        'struct_LV1_x2',
        'struct_LV1_sigma',
        'struct_LV2_intercept',
        'struct_LV2_sigma',
    }
    assert params['struct_LV1_intercept'].role is ParameterRole.STRUCTURAL_INTERCEPT
    assert params['struct_LV1_intercept'].semantic_ref == StructuralIntercept('LV1')
    assert (
        params['struct_LV1_intercept'].creation_kind is ParameterCreationKind.FREE_BETA
    )
    assert params['struct_LV1_intercept'].initial_value == 0.0
    assert params['struct_LV2_intercept'].role is ParameterRole.STRUCTURAL_INTERCEPT
    assert params['struct_LV2_intercept'].semantic_ref == StructuralIntercept('LV2')
    assert params['struct_LV1_x1'].role is ParameterRole.STRUCTURAL_COEFFICIENT
    assert params['struct_LV1_x1'].creation_kind is ParameterCreationKind.FREE_BETA
    assert params['struct_LV1_sigma'].role is ParameterRole.STRUCTURAL_SIGMA
    assert (
        params['struct_LV1_sigma'].creation_kind is ParameterCreationKind.LOG_EXP_BETA
    )
    assert params['struct_LV1_sigma'].initial_value == pytest.approx(math.log(10.0))
    assert params['struct_LV2_sigma'].initial_value == pytest.approx(math.log(2.0))


def test_resolve_measurement_parameters_builds_intercepts_sigmas_and_loadings() -> None:
    prepared = resolver._Prepared(
        latent_variables=[],
        indicators=[make_indicator('ind1', 't'), make_indicator('ind2', 't')],
        types=[],
        indicator_by_name={},
        type_by_name={},
        measurement_spec_by_indicator={
            'ind1': IndicatorMeasurementSpec(
                indicator_name='ind1',
                measurement_model=MeasurementModel.GAUSSIAN,
                measurement_sigma=PositiveParameterSpec(start=3.0, lower_bound=0.1),
            ),
            'ind2': IndicatorMeasurementSpec(
                indicator_name='ind2',
                measurement_model=MeasurementModel.ORDERED_LOGIT,
            ),
        },
        indicator_to_latents={'ind1': ['LV1', 'LV2'], 'ind2': []},
        ordinal_type_names=[],
    )
    context = make_context(positivity_mode=PositivityMode.LOWER_BOUND)

    params = resolver._resolve_measurement_parameters(prepared, context, None)

    assert set(params) == {
        'measurement_intercept_ind1',
        'measurement_ind1_sigma',
        'measurement_coefficient_LV1_ind1',
        'measurement_coefficient_LV2_ind1',
        'measurement_intercept_ind2',
        'measurement_ind2_sigma',
    }
    assert (
        params['measurement_intercept_ind1'].role is ParameterRole.MEASUREMENT_INTERCEPT
    )
    assert params['measurement_ind1_sigma'].role is ParameterRole.MEASUREMENT_SIGMA
    assert (
        params['measurement_ind1_sigma'].creation_kind
        is ParameterCreationKind.BOUNDED_BETA
    )
    assert params['measurement_ind1_sigma'].initial_value == 3.0
    assert (
        params['measurement_coefficient_LV1_ind1'].role
        is ParameterRole.MEASUREMENT_LOADING
    )


def test_resolve_threshold_parameters_symmetric_log_exp() -> None:
    lt = LikertType(
        type_name='sym5',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[3],
    )
    prepared = resolver._Prepared(
        latent_variables=[],
        indicators=[],
        types=[lt],
        indicator_by_name={},
        type_by_name={'sym5': lt},
        measurement_spec_by_indicator={},
        indicator_to_latents={},
        ordinal_type_names=['sym5'],
    )
    context = make_context(positivity_mode=PositivityMode.LOG_EXP)

    params = resolver._resolve_threshold_parameters(prepared, context, None)

    assert set(params) == {'sym5_delta_0', 'sym5_delta_1'}
    assert params['sym5_delta_0'].role is ParameterRole.THRESHOLD_DELTA
    assert params['sym5_delta_0'].creation_kind is ParameterCreationKind.LOG_EXP_BETA
    assert params['sym5_delta_0'].initial_value == -0.86
    assert params['sym5_delta_1'].initial_value == -0.43


def test_resolve_threshold_parameters_monotone_lower_bound() -> None:
    lt = LikertType(
        type_name='mono4',
        symmetric=False,
        categories=[1, 2, 3, 4],
        neutral_labels=[],
    )
    prepared = resolver._Prepared(
        latent_variables=[],
        indicators=[],
        types=[lt],
        indicator_by_name={},
        type_by_name={'mono4': lt},
        measurement_spec_by_indicator={},
        indicator_to_latents={},
        ordinal_type_names=['mono4'],
    )
    context = make_context(positivity_mode=PositivityMode.LOWER_BOUND)

    params = resolver._resolve_threshold_parameters(prepared, context, None)

    assert set(params) == {'mono4_tau_1', 'mono4_delta_1', 'mono4_delta_2'}
    assert params['mono4_tau_1'].role is ParameterRole.THRESHOLD_FIRST
    assert params['mono4_tau_1'].creation_kind is ParameterCreationKind.FREE_BETA
    assert params['mono4_delta_1'].initial_value == 1.0
    assert params['mono4_delta_1'].creation_kind is ParameterCreationKind.BOUNDED_BETA
    assert params['mono4_delta_2'].initial_value == 1.0


def test_resolve_threshold_systems_symmetric_covers_left_middle_and_right() -> None:
    lt = LikertType(
        type_name='sym5',
        symmetric=True,
        categories=[1, 2, 3, 4, 5],
        neutral_labels=[3],
    )
    prepared = resolver._Prepared(
        latent_variables=[],
        indicators=[make_indicator('ind1', 'sym5'), make_indicator('ind2', 'sym5')],
        types=[lt],
        indicator_by_name={},
        type_by_name={'sym5': lt},
        measurement_spec_by_indicator={
            'ind1': IndicatorMeasurementSpec(
                indicator_name='ind1',
                measurement_model=MeasurementModel.ORDERED_PROBIT,
            ),
            'ind2': IndicatorMeasurementSpec(
                indicator_name='ind2',
                measurement_model=MeasurementModel.ORDERED_LOGIT,
            ),
        },
        indicator_to_latents={},
        ordinal_type_names=['sym5'],
    )
    context = make_context()
    params = {
        'sym5_delta_0': resolver.ResolvedParameter(
            semantic_ref=ThresholdDelta('sym5', 0),
            final_name='sym5_delta_0',
            role=ParameterRole.THRESHOLD_DELTA,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.LOG_EXP,
            creation_kind=ParameterCreationKind.LOG_EXP_BETA,
            notes=[],
        ),
        'sym5_delta_1': resolver.ResolvedParameter(
            semantic_ref=ThresholdDelta('sym5', 1),
            final_name='sym5_delta_1',
            role=ParameterRole.THRESHOLD_DELTA,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.LOG_EXP,
            creation_kind=ParameterCreationKind.LOG_EXP_BETA,
            notes=[],
        ),
    }

    systems = resolver._resolve_threshold_systems(prepared, context, params)
    system = systems['sym5']

    assert system.construction_kind is ThresholdConstructionKind.SYMMETRIC
    assert system.used_by_indicators == ['ind1', 'ind2']
    assert system.normalization_notes == [
        "Symmetric threshold construction for type 'sym5'."
    ]
    assert [cp.symbol_name for cp in system.cutpoints] == [
        'tau_1',
        'tau_2',
        'tau_3',
        'tau_4',
    ]
    assert system.cutpoints[0].kind is CutpointKind.DERIVED
    assert system.cutpoints[0].expression_text == '-(sym5_delta_0 - sym5_delta_1)'
    assert system.cutpoints[0].source_parameter_names == [
        'sym5_delta_0',
        'sym5_delta_1',
    ]
    assert system.cutpoints[1].expression_text == '-sym5_delta_1'
    assert system.cutpoints[2].expression_text == 'sym5_delta_0'
    assert system.cutpoints[3].expression_text == 'sym5_delta_0 + sym5_delta_1'


def test_resolve_threshold_systems_monotone_covers_fixed_and_free_tau1() -> None:
    lt = LikertType(
        type_name='mono4',
        symmetric=False,
        categories=[1, 2, 3, 4],
        neutral_labels=[],
    )
    prepared = resolver._Prepared(
        latent_variables=[],
        indicators=[make_indicator('ind1', 'mono4'), make_indicator('ind2', 'mono4')],
        types=[lt],
        indicator_by_name={},
        type_by_name={'mono4': lt},
        measurement_spec_by_indicator={
            'ind1': IndicatorMeasurementSpec(
                indicator_name='ind1',
                measurement_model=MeasurementModel.ORDERED_LOGIT,
            ),
            'ind2': IndicatorMeasurementSpec(
                indicator_name='ind2',
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
        },
        indicator_to_latents={},
        ordinal_type_names=['mono4'],
    )
    context = make_context()

    free_params = {
        'mono4_tau_1': resolver.ResolvedParameter(
            semantic_ref=ThresholdFirst('mono4'),
            final_name='mono4_tau_1',
            role=ParameterRole.THRESHOLD_FIRST,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        )
    }
    free_params['mono4_delta_1'] = resolver.ResolvedParameter(
        semantic_ref=ThresholdDelta('mono4', 1),
        final_name='mono4_delta_1',
        role=ParameterRole.THRESHOLD_DELTA,
        status=ParameterStatus.FREE,
        fixed_value=None,
        initial_value=0.0,
        lower_bound=None,
        upper_bound=None,
        positivity_strategy=PositivityStrategy.LOWER_BOUND,
        creation_kind=ParameterCreationKind.BOUNDED_BETA,
        notes=[],
    )
    free_params['mono4_delta_2'] = resolver.ResolvedParameter(
        semantic_ref=ThresholdDelta('mono4', 2),
        final_name='mono4_delta_2',
        role=ParameterRole.THRESHOLD_DELTA,
        status=ParameterStatus.FREE,
        fixed_value=None,
        initial_value=0.0,
        lower_bound=None,
        upper_bound=None,
        positivity_strategy=PositivityStrategy.LOWER_BOUND,
        creation_kind=ParameterCreationKind.BOUNDED_BETA,
        notes=[],
    )

    systems_free = resolver._resolve_threshold_systems(prepared, context, free_params)
    mono_free = systems_free['mono4']

    assert mono_free.construction_kind is ThresholdConstructionKind.MONOTONE
    assert mono_free.used_by_indicators == ['ind1']
    assert mono_free.cutpoints[0].kind is CutpointKind.FREE
    assert mono_free.cutpoints[0].expression_text == 'mono4_tau_1'
    assert mono_free.cutpoints[0].source_parameter_names == ['mono4_tau_1']
    assert mono_free.cutpoints[1].expression_text == 'tau_1 + mono4_delta_1'
    assert mono_free.cutpoints[1].source_parameter_names == ['tau_1', 'mono4_delta_1']
    assert mono_free.cutpoints[2].expression_text == 'tau_2 + mono4_delta_2'

    fixed_params = dict(free_params)
    fixed_params['mono4_tau_1'] = resolver.ResolvedParameter(
        semantic_ref=ThresholdFirst('mono4'),
        final_name='mono4_tau_1',
        role=ParameterRole.THRESHOLD_FIRST,
        status=ParameterStatus.FIXED,
        fixed_value=-1.5,
        initial_value=-1.5,
        lower_bound=None,
        upper_bound=None,
        positivity_strategy=None,
        creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
        notes=[],
    )

    systems_fixed = resolver._resolve_threshold_systems(prepared, context, fixed_params)
    mono_fixed = systems_fixed['mono4']

    assert mono_fixed.cutpoints[0].kind is CutpointKind.FIXED
    assert mono_fixed.cutpoints[0].expression_text == '-1.5'
    assert mono_fixed.cutpoints[0].source_parameter_names == []
    assert mono_fixed.normalization_notes == [
        "Monotone threshold construction for type 'mono4'."
    ]


def test_resolve_structural_equations_uses_draw_type_and_sigma_refs() -> None:
    prepared = resolver._Prepared(
        latent_variables=[
            make_latent('LV1', ['x1', 'x2'], ['ind1']),
        ],
        indicators=[],
        types=[],
        indicator_by_name={},
        type_by_name={},
        measurement_spec_by_indicator={},
        indicator_to_latents={},
        ordinal_type_names=[],
    )
    context = make_context(draw_type='MLHS_DRAW')
    params = {
        'struct_LV1_intercept': resolver.ResolvedParameter(
            semantic_ref=StructuralIntercept('LV1'),
            final_name='struct_LV1_intercept',
            role=ParameterRole.STRUCTURAL_INTERCEPT,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'struct_LV1_x1': resolver.ResolvedParameter(
            semantic_ref=StructuralCoefficient('LV1', 'x1'),
            final_name='struct_LV1_x1',
            role=ParameterRole.STRUCTURAL_COEFFICIENT,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'struct_LV1_x2': resolver.ResolvedParameter(
            semantic_ref=StructuralCoefficient('LV1', 'x2'),
            final_name='struct_LV1_x2',
            role=ParameterRole.STRUCTURAL_COEFFICIENT,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'struct_LV1_sigma': resolver.ResolvedParameter(
            semantic_ref=StructuralSigma('LV1'),
            final_name='struct_LV1_sigma',
            role=ParameterRole.STRUCTURAL_SIGMA,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.LOG_EXP,
            creation_kind=ParameterCreationKind.LOG_EXP_BETA,
            notes=[],
        ),
    }

    equations = resolver._resolve_structural_equations(prepared, context, params)
    eq = equations['LV1']

    assert eq.latent_name == 'LV1'
    assert eq.expression_name == 'LV1'
    assert eq.systematic_part.intercept.final_name == 'struct_LV1_intercept'
    assert [term.variable_name for term in eq.systematic_part.terms] == ['x1', 'x2']
    assert eq.systematic_part.terms[0].coefficient.final_name == 'struct_LV1_x1'
    assert eq.sigma.final_name == 'struct_LV1_sigma'
    assert eq.draw_name == 'struct_LV1_draws'
    assert eq.draw_type == 'MLHS_DRAW'
    assert eq.error_distribution == 'normal'


def test_resolve_measurement_equations_covers_all_model_branches() -> None:
    prepared = resolver._Prepared(
        latent_variables=[],
        indicators=[
            make_indicator('ind_g', 'cont'),
            make_indicator('ind_p', 'ord'),
            make_indicator('ind_l', 'ord'),
        ],
        types=[],
        indicator_by_name={},
        type_by_name={},
        measurement_spec_by_indicator={
            'ind_g': IndicatorMeasurementSpec(
                indicator_name='ind_g',
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
            'ind_p': IndicatorMeasurementSpec(
                indicator_name='ind_p',
                measurement_model=MeasurementModel.ORDERED_PROBIT,
            ),
            'ind_l': IndicatorMeasurementSpec(
                indicator_name='ind_l',
                measurement_model=MeasurementModel.ORDERED_LOGIT,
            ),
        },
        indicator_to_latents={
            'ind_g': ['LV1'],
            'ind_p': ['LV1', 'LV2'],
            'ind_l': [],
        },
        ordinal_type_names=['ord'],
    )
    context = make_context()
    params = {
        'measurement_intercept_ind_g': resolver.ResolvedParameter(
            semantic_ref=MeasurementIntercept('ind_g'),
            final_name='measurement_intercept_ind_g',
            role=ParameterRole.MEASUREMENT_INTERCEPT,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'measurement_ind_g_sigma': resolver.ResolvedParameter(
            semantic_ref=MeasurementSigma('ind_g'),
            final_name='measurement_ind_g_sigma',
            role=ParameterRole.MEASUREMENT_SIGMA,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.LOWER_BOUND,
            creation_kind=ParameterCreationKind.BOUNDED_BETA,
            notes=[],
        ),
        'measurement_coefficient_LV1_ind_g': resolver.ResolvedParameter(
            semantic_ref=MeasurementLoading('LV1', 'ind_g'),
            final_name='measurement_coefficient_LV1_ind_g',
            role=ParameterRole.MEASUREMENT_LOADING,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'measurement_intercept_ind_p': resolver.ResolvedParameter(
            semantic_ref=MeasurementIntercept('ind_p'),
            final_name='measurement_intercept_ind_p',
            role=ParameterRole.MEASUREMENT_INTERCEPT,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'measurement_ind_p_sigma': resolver.ResolvedParameter(
            semantic_ref=MeasurementSigma('ind_p'),
            final_name='measurement_ind_p_sigma',
            role=ParameterRole.MEASUREMENT_SIGMA,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.LOWER_BOUND,
            creation_kind=ParameterCreationKind.BOUNDED_BETA,
            notes=[],
        ),
        'measurement_coefficient_LV1_ind_p': resolver.ResolvedParameter(
            semantic_ref=MeasurementLoading('LV1', 'ind_p'),
            final_name='measurement_coefficient_LV1_ind_p',
            role=ParameterRole.MEASUREMENT_LOADING,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'measurement_coefficient_LV2_ind_p': resolver.ResolvedParameter(
            semantic_ref=MeasurementLoading('LV2', 'ind_p'),
            final_name='measurement_coefficient_LV2_ind_p',
            role=ParameterRole.MEASUREMENT_LOADING,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'measurement_intercept_ind_l': resolver.ResolvedParameter(
            semantic_ref=MeasurementIntercept('ind_l'),
            final_name='measurement_intercept_ind_l',
            role=ParameterRole.MEASUREMENT_INTERCEPT,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=ParameterCreationKind.FREE_BETA,
            notes=[],
        ),
        'measurement_ind_l_sigma': resolver.ResolvedParameter(
            semantic_ref=MeasurementSigma('ind_l'),
            final_name='measurement_ind_l_sigma',
            role=ParameterRole.MEASUREMENT_SIGMA,
            status=ParameterStatus.FREE,
            fixed_value=None,
            initial_value=0.0,
            lower_bound=None,
            upper_bound=None,
            positivity_strategy=PositivityStrategy.LOWER_BOUND,
            creation_kind=ParameterCreationKind.BOUNDED_BETA,
            notes=[],
        ),
    }

    equations = resolver._resolve_measurement_equations(prepared, context, params)

    eq_g = equations['ind_g']
    assert eq_g.measurement_model is MeasurementModel.GAUSSIAN
    assert eq_g.threshold_system_name is None
    assert eq_g.error_distribution is MeasurementErrorDistribution.GAUSSIAN
    assert eq_g.systematic_part.intercept.final_name == 'measurement_intercept_ind_g'
    assert [t.variable_name for t in eq_g.systematic_part.terms] == ['LV1']
    assert eq_g.observed_variable_name == 'ind_g'

    eq_p = equations['ind_p']
    assert eq_p.threshold_system_name == 'ord'
    assert eq_p.error_distribution is MeasurementErrorDistribution.GAUSSIAN
    assert [t.variable_name for t in eq_p.systematic_part.terms] == ['LV1', 'LV2']

    eq_l = equations['ind_l']
    assert eq_l.threshold_system_name == 'ord'
    assert eq_l.error_distribution is MeasurementErrorDistribution.LOGISTIC
    assert eq_l.systematic_part.terms == []


def test_resolve_normalization_summary_with_plan_and_warning() -> None:
    prepared = resolver._Prepared(
        latent_variables=[
            make_latent('LV1', [], ['ind1']),
            make_latent('LV2', [], ['ind2']),
        ],
        indicators=[],
        types=[],
        indicator_by_name={},
        type_by_name={},
        measurement_spec_by_indicator={},
        indicator_to_latents={},
        ordinal_type_names=[],
    )
    plan = NormalizationPlan(
        [
            Fixing(
                target=MeasurementIntercept('ind1'),
                value=0.0,
                note='center indicator',
            ),
            Fixing(
                target=MeasurementLoading('LV1', 'ind1'),
                value=1.0,
                note=None,
            ),
        ]
    )
    latent_variables = {
        'LV1': resolver.ResolvedLatentVariable(
            name='LV1',
            structural_equation=resolver.ResolvedStructuralEquation(
                latent_name='LV1',
                expression_name='LV1',
                systematic_part=resolver.ResolvedLinearCombination(
                    intercept=resolver.ResolvedParameterRef(
                        final_name='struct_LV1_intercept'
                    ),
                    terms=[],
                ),
                sigma=resolver.ResolvedParameterRef(final_name='sigma1'),
                draw_name='d1',
                draw_type='NORMAL',
                error_distribution='normal',
            ),
            indicator_names=['ind1'],
            reference_indicator='ind1',
            normalization_notes=[],
        ),
        'LV2': resolver.ResolvedLatentVariable(
            name='LV2',
            structural_equation=resolver.ResolvedStructuralEquation(
                latent_name='LV2',
                expression_name='LV2',
                systematic_part=resolver.ResolvedLinearCombination(
                    intercept=resolver.ResolvedParameterRef(
                        final_name='struct_LV2_intercept'
                    ),
                    terms=[],
                ),
                sigma=resolver.ResolvedParameterRef(final_name='sigma2'),
                draw_name='d2',
                draw_type='NORMAL',
                error_distribution='normal',
            ),
            indicator_names=['ind2'],
            reference_indicator=None,
            normalization_notes=[],
        ),
    }

    summary = resolver._resolve_normalization_summary(
        prepared,
        plan,
        latent_variables,
    )

    assert len(summary.rules) == 2
    assert summary.rules[0].scope == 'MeasurementIntercept'
    assert summary.rules[0].target_name == repr(MeasurementIntercept('ind1'))
    assert summary.rules[0].value == 0.0
    assert summary.rules[0].reason == 'center indicator'
    assert summary.rules[1].reason == 'Explicit normalization fixing.'
    assert summary.warnings == [
        "No obvious reference indicator could be inferred for latent variable 'LV2'."
    ]
    assert 'modeling guidance' in summary.disclaimer
    assert 'identification' in summary.disclaimer


def test_resolve_model_end_to_end_infers_reference_indicator_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolver,
        'validate_specification',
        lambda **kwargs: _ValidationResult(),
    )
    monkeypatch.setattr(
        resolver,
        'validate_normalization_plan',
        lambda **kwargs: _ValidationResult(),
    )

    latent_variables = [
        make_latent(
            'LV1',
            ['income'],
            ['ind_ref', 'ind_ord'],
            structural_sigma=PositiveParameterSpec(start=4.0, lower_bound=0.5),
        ),
        make_latent(
            'LV2',
            ['age'],
            ['ind_ord', 'ind_log'],
            structural_sigma=None,
        ),
    ]
    likert_indicators = [
        make_indicator('ind_ref', 'cont'),
        make_indicator('ind_ord', 'ord5sym'),
        make_indicator('ind_log', 'ord4mono'),
    ]
    likert_types = [
        LikertType(
            type_name='cont',
            symmetric=False,
            categories=[0, 1],
            neutral_labels=[],
        ),
        LikertType(
            type_name='ord5sym',
            symmetric=True,
            categories=[1, 2, 3, 4, 5],
            neutral_labels=[3],
        ),
        LikertType(
            type_name='ord4mono',
            symmetric=False,
            categories=[1, 2, 3, 4],
            neutral_labels=[],
        ),
    ]
    measurement_configuration = make_measurement_configuration(
        IndicatorMeasurementSpec(
            indicator_name='ind_ref',
            measurement_model=MeasurementModel.GAUSSIAN,
            measurement_sigma=PositiveParameterSpec(start=2.0, lower_bound=0.1),
        ),
        IndicatorMeasurementSpec(
            indicator_name='ind_ord',
            measurement_model=MeasurementModel.ORDERED_PROBIT,
        ),
        IndicatorMeasurementSpec(
            indicator_name='ind_log',
            measurement_model=MeasurementModel.ORDERED_LOGIT,
        ),
    )
    normalization_plan = NormalizationPlan(
        [
            Fixing(target=MeasurementIntercept('ind_ref'), value=0.0, note='anchor'),
            Fixing(
                target=MeasurementLoading('LV1', 'ind_ref'), value=1.0, note='scale'
            ),
            Fixing(
                target=ThresholdFirst('ord4mono'), value=-1.0, note='threshold anchor'
            ),
        ]
    )
    context = make_context(
        estimation_mode=EstimationMode.BAYESIAN,
        positivity_mode=PositivityMode.LOWER_BOUND,
        draw_type='DRAW_TEST',
    )

    model = resolver.resolve_model(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        measurement_configuration=measurement_configuration,
        context=context,
        normalization_plan=normalization_plan,
    )

    assert model.metadata.estimation_mode is EstimationMode.BAYESIAN
    assert model.metadata.measurement_models_present == [
        MeasurementModel.GAUSSIAN,
        MeasurementModel.ORDERED_LOGIT,
        MeasurementModel.ORDERED_PROBIT,
    ]
    assert model.metadata.has_gaussian is True
    assert model.metadata.has_ordered_probit is True
    assert model.metadata.has_ordered_logit is True
    assert model.metadata.has_ordinal is True
    assert model.metadata.n_latent_variables == 2
    assert model.metadata.n_indicators == 3
    assert model.metadata.n_threshold_systems == 2

    assert set(model.latent_variables) == {'LV1', 'LV2'}
    assert model.latent_variables['LV1'].indicator_names == ['ind_ord', 'ind_ref']
    assert model.latent_variables['LV1'].reference_indicator == 'ind_ref'
    assert model.latent_variables['LV1'].normalization_notes == [
        "Reference indicator inferred from normalization plan: 'ind_ref'."
    ]
    assert model.latent_variables['LV2'].reference_indicator is None
    assert model.latent_variables['LV2'].normalization_notes == []

    assert (
        model.latent_variables['LV1'].structural_equation.draw_name
        == 'struct_LV1_draws'
    )
    assert model.latent_variables['LV1'].structural_equation.draw_type == 'DRAW_TEST'
    assert (
        model.latent_variables['LV1'].structural_equation.error_distribution == 'normal'
    )

    assert model.measurement_equations['ind_ref'].threshold_system_name is None
    assert (
        model.measurement_equations['ind_ref'].error_distribution
        is MeasurementErrorDistribution.GAUSSIAN
    )
    assert model.measurement_equations['ind_ord'].threshold_system_name == 'ord5sym'
    assert (
        model.measurement_equations['ind_ord'].error_distribution
        is MeasurementErrorDistribution.GAUSSIAN
    )
    assert model.measurement_equations['ind_log'].threshold_system_name == 'ord4mono'
    assert (
        model.measurement_equations['ind_log'].error_distribution
        is MeasurementErrorDistribution.LOGISTIC
    )

    sym_system = model.threshold_systems['ord5sym']
    assert sym_system.construction_kind is ThresholdConstructionKind.SYMMETRIC
    assert sym_system.used_by_indicators == ['ind_ord']
    assert len(sym_system.cutpoints) == 4
    assert sym_system.cutpoints[2].expression_text == 'ord5sym_delta_0'

    mono_system = model.threshold_systems['ord4mono']
    assert mono_system.construction_kind is ThresholdConstructionKind.MONOTONE
    assert mono_system.used_by_indicators == ['ind_log']
    assert mono_system.cutpoints[0].kind is CutpointKind.FIXED
    assert mono_system.cutpoints[0].expression_text == '-1.0'

    assert 'struct_LV1_income' in model.parameters
    assert 'struct_LV2_age' in model.parameters
    assert 'measurement_intercept_ind_ref' in model.parameters
    assert 'measurement_coefficient_LV1_ind_ref' in model.parameters
    assert 'measurement_coefficient_LV2_ind_log' in model.parameters
    assert 'ord5sym_delta_0' in model.parameters
    assert 'ord5sym_delta_1' in model.parameters
    assert 'ord4mono_tau_1' in model.parameters
    assert 'ord4mono_delta_1' in model.parameters
    assert 'ord4mono_delta_2' in model.parameters

    assert (
        model.parameters['measurement_intercept_ind_ref'].status
        is ParameterStatus.FIXED
    )
    assert (
        model.parameters['measurement_coefficient_LV1_ind_ref'].status
        is ParameterStatus.FIXED
    )
    assert model.parameters['ord4mono_tau_1'].status is ParameterStatus.FIXED
    assert (
        model.parameters['measurement_ind_ref_sigma'].creation_kind
        is ParameterCreationKind.BOUNDED_BETA
    )
    assert model.parameters['measurement_ind_ref_sigma'].initial_value == 2.0
    assert (
        model.parameters['struct_LV1_sigma'].creation_kind
        is ParameterCreationKind.BOUNDED_BETA
    )
    assert model.parameters['struct_LV1_sigma'].initial_value == 4.0
    assert (
        model.parameters['ord5sym_delta_0'].creation_kind
        is ParameterCreationKind.BOUNDED_BETA
    )
    assert model.parameters['ord5sym_delta_0'].initial_value == 1.0

    assert len(model.normalization.rules) == 3
    assert model.normalization.warnings == [
        "No obvious reference indicator could be inferred for latent variable 'LV2'."
    ]
    assert 'identification' in model.normalization.disclaimer
