from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from biogeme.expressions import Beta, Numeric, Variable
from biogeme.latent_variables import EstimationMode, MeasurementModel
from biogeme.latent_variables.biogeme_builder import (
    BuiltBiogemeModel,
    _beta_or_numeric,
    _build_biogeme_model_bayesian,
    _build_biogeme_model_ml,
    _build_measurement_log_terms_bayesian,
    _build_measurement_terms_ml,
    _mask_neutral_labels,
    _render_linear_combination,
    build_biogeme_model,
)
from biogeme.latent_variables.resolved import (
    ParameterCreationKind,
    ParameterRole,
    ParameterStatus,
    PositivityStrategy,
    ResolvedConstant,
    ResolvedLinearCombination,
    ResolvedParameter,
)

# ------------------------------------------------------------------------------
# Small helper fixtures for duck-typed resolved model components
# ------------------------------------------------------------------------------


@dataclass
class FakeTerm:
    coefficient: object
    variable_name: str


@dataclass
class FakeStructuralEquation:
    terms: list[FakeTerm]
    sigma: object | None
    draw_name: str
    draw_type: str
    intercept: object | None = None

    @property
    def systematic_part(self) -> ResolvedLinearCombination:
        return ResolvedLinearCombination(self.intercept, self.terms)


@dataclass
class FakeLatentVariable:
    structural_equation: FakeStructuralEquation


@dataclass
class FakeMeasurementEquation:
    systematic_part: ResolvedLinearCombination
    observed_variable_name: str
    measurement_model: MeasurementModel
    sigma: object | None
    threshold_system_name: str | None = None
    type_name: str = 'likert'


@dataclass
class FakeCutpoint:
    symbol_name: str
    expression_text: str


@dataclass
class FakeThresholdSystem:
    cutpoints: list[FakeCutpoint]
    categories: list[int]
    neutral_labels: list[str]


@dataclass
class FakeMetadata:
    estimation_mode: EstimationMode


@dataclass
class FakeIndicatorType:
    neutral_labels: list[int]


@dataclass
class FakeResolvedModel:
    parameters: dict[str, object]
    latent_variables: dict[str, object]
    threshold_systems: dict[str, object]
    measurement_equations: dict[str, object]
    metadata: FakeMetadata
    indicator_types: dict[str, FakeIndicatorType] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


def expr_text(expr: object) -> str:
    """Robust textual rendering for Biogeme expressions."""
    return str(expr).replace('`', '')


def class_name(expr: object) -> str:
    return expr.__class__.__name__


def make_parameter(
    *,
    final_name: str,
    creation_kind: ParameterCreationKind,
    initial_value: float = 0.0,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    fixed_value: float | None = None,
    role: ParameterRole = ParameterRole.MEASUREMENT_SIGMA,
    status: ParameterStatus | None = None,
) -> ResolvedParameter:
    """
    Create a ResolvedParameter while being tolerant to constructor differences.
    """
    if status is None:
        status = (
            ParameterStatus.FIXED
            if creation_kind
            in {
                ParameterCreationKind.NUMERIC_CONSTANT,
                ParameterCreationKind.FIXED_BETA,
            }
            else ParameterStatus.FREE
        )
    try:
        return ResolvedParameter(
            semantic_ref=None,
            final_name=final_name,
            role=role,
            status=status,
            fixed_value=fixed_value,
            initial_value=initial_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            positivity_strategy=PositivityStrategy.NONE,
            creation_kind=creation_kind,
            notes=[],
        )
    except TypeError:
        # Fallback if the project dataclass/model has a different constructor.
        obj = SimpleNamespace(
            semantic_ref=None,
            role=role,
            status=status,
            final_name=final_name,
            creation_kind=creation_kind,
            initial_value=initial_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            positivity_strategy=PositivityStrategy.NONE,
            fixed_value=fixed_value,
            notes=[],
        )
        return obj  # type: ignore[return-value]


def ml_mode() -> EstimationMode:
    """Return the maximum-likelihood estimation mode used by the current Biogeme version."""
    for candidate in ('ML', 'MAXIMUM_LIKELIHOOD'):
        if hasattr(EstimationMode, candidate):
            return getattr(EstimationMode, candidate)
    raise AttributeError("EstimationMode has neither 'ML' nor 'MAXIMUM_LIKELIHOOD'.")


def bayesian_mode() -> EstimationMode:
    """Return the Bayesian estimation mode used by the current Biogeme version."""
    for candidate in ('BAYESIAN', 'BAYES'):
        if hasattr(EstimationMode, candidate):
            return getattr(EstimationMode, candidate)
    raise AttributeError("EstimationMode has neither 'BAYESIAN' nor 'BAYES'.")


# ------------------------------------------------------------------------------
# Tests for _beta_or_numeric
# ------------------------------------------------------------------------------


def test_beta_or_numeric_numeric_constant():
    param = make_parameter(
        final_name='c1',
        creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
        fixed_value=3.5,
    )

    result = _beta_or_numeric(param)

    assert isinstance(result, Numeric)
    assert expr_text(result) == '3.5'


def test_beta_or_numeric_log_exp_beta():
    param = make_parameter(
        final_name='sigma',
        creation_kind=ParameterCreationKind.LOG_EXP_BETA,
        initial_value=0.1,
    )

    result = _beta_or_numeric(param)

    text = expr_text(result)
    assert 'sigma_log' in text
    assert 'exp(' in text or class_name(result).lower().startswith('exp')


@pytest.mark.parametrize(
    ('kind', 'expected_status'),
    [
        (ParameterCreationKind.BOUNDED_BETA, '0'),
        (ParameterCreationKind.FREE_BETA, '0'),
        (ParameterCreationKind.FIXED_BETA, '1'),
    ],
)
def test_beta_or_numeric_beta_variants(kind, expected_status):
    param = make_parameter(
        final_name='b_time',
        creation_kind=kind,
        initial_value=1.2,
        lower_bound=-10,
        upper_bound=10,
    )

    result = _beta_or_numeric(param)

    assert isinstance(result, Beta)
    text = expr_text(result)
    assert 'b_time' in text
    assert '1.2' in text
    assert expected_status in text


def test_beta_or_numeric_unsupported_creation_kind_raises():
    class Unsupported:
        pass

    param = SimpleNamespace(
        final_name='x',
        creation_kind=Unsupported(),
        initial_value=0,
        lower_bound=None,
        upper_bound=None,
        fixed_value=None,
    )

    with pytest.raises(ValueError, match='Unsupported parameter creation kind'):
        _beta_or_numeric(param)


# ------------------------------------------------------------------------------
# Tests for _render_linear_combination
# ------------------------------------------------------------------------------


def test_render_linear_combination_with_constant_intercept_and_constant_term():
    combo = ResolvedLinearCombination(
        intercept=ResolvedConstant(2.0),
        terms=[FakeTerm(ResolvedConstant(3.0), 'x1')],
    )

    result = _render_linear_combination(combo, parameters={})

    text = expr_text(result)
    assert '2.0' in text
    assert '3.0' in text
    assert 'x1' in text


def test_render_linear_combination_with_parameter_intercept_and_parameter_term():
    alpha = Beta('alpha', 0, None, None, 0)
    beta_x = Beta('beta_x', 1, None, None, 0)
    parameters = {'alpha': alpha, 'beta_x': beta_x}

    combo = ResolvedLinearCombination(
        intercept=SimpleNamespace(final_name='alpha'),
        terms=[FakeTerm(SimpleNamespace(final_name='beta_x'), 'x')],
    )

    result = _render_linear_combination(combo, parameters=parameters)

    text = expr_text(result)
    assert 'alpha' in text
    assert 'beta_x' in text
    assert 'x' in text


def test_render_linear_combination_uses_symbols_when_available():
    combo = ResolvedLinearCombination(
        intercept=None,
        terms=[FakeTerm(ResolvedConstant(1.0), 'LV1')],
    )

    latent_expr = Beta('latent_proxy', 0, None, None, 0)

    result = _render_linear_combination(
        combo,
        parameters={},
        symbols={'LV1': latent_expr},
    )

    text = expr_text(result)
    assert 'latent_proxy' in text
    assert 'LV1' not in text


def test_render_linear_combination_uses_variable_when_symbol_missing():
    combo = ResolvedLinearCombination(
        intercept=None,
        terms=[FakeTerm(ResolvedConstant(1.0), 'obs_income')],
    )

    result = _render_linear_combination(combo, parameters={}, symbols={})

    text = expr_text(result)
    assert 'obs_income' in text


# ------------------------------------------------------------------------------
# Tests for _build_measurement_terms_ml
# ------------------------------------------------------------------------------


def test_build_measurement_terms_ml_gaussian():
    sigma_param = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )
    parameters = {'sigma_ind': _beta_or_numeric(sigma_param)}

    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(
            intercept=ResolvedConstant(1.0),
            terms=[FakeTerm(ResolvedConstant(2.0), 'x')],
        ),
        observed_variable_name='y_obs',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=SimpleNamespace(final_name='sigma_ind'),
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={},
        measurement_equations={'y1': equation},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
        indicator_types={'likert': FakeIndicatorType(neutral_labels=[6, -1])},
    )

    result = _build_measurement_terms_ml(
        resolved=resolved,
        parameters=parameters,
        latent_expressions={},
        threshold_expressions={},
    )

    assert set(result) == {'y1'}
    text = expr_text(result['y1'])
    assert 'y_obs' in text
    assert 'sigma_ind' in text
    assert class_name(result['y1']) == 'Elem'
    assert '6' in text
    assert '-1' in text


@pytest.mark.parametrize(
    ('observed', 'neutral_value', 'expected'),
    [
        (6, 1.0, 1.0),
        (-1, 0.0, 0.0),
        (3, 1.0, 0.25),
    ],
)
def test_mask_neutral_labels(
    observed: int, neutral_value: float, expected: float
) -> None:
    result = _mask_neutral_labels(
        Numeric(0.25),
        Numeric(observed),
        [6, -1],
        neutral_value=neutral_value,
    )

    assert result.get_value() == pytest.approx(expected)


def test_build_measurement_terms_ml_gaussian_requires_sigma():
    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(
            intercept=None,
            terms=[],
        ),
        observed_variable_name='y_obs',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=None,
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={},
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    with pytest.raises(ValueError, match='requires a resolved sigma parameter'):
        _build_measurement_terms_ml(
            resolved=resolved,
            parameters={},
            latent_expressions={},
            threshold_expressions={},
        )


def test_build_measurement_terms_ml_ordered_probit():
    sigma_param = make_parameter(
        final_name='sigma_ord',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )
    parameters = {'sigma_ord': _beta_or_numeric(sigma_param)}

    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y_ord',
        measurement_model=MeasurementModel.ORDERED_PROBIT,
        sigma=SimpleNamespace(final_name='sigma_ord'),
        threshold_system_name='thr',
    )

    threshold_system = FakeThresholdSystem(
        cutpoints=[],
        categories=[1, 2, 3],
        neutral_labels=[],
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={'thr': threshold_system},
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    threshold_expressions = {'thr': [Numeric(-1), Numeric(1)]}

    result = _build_measurement_terms_ml(
        resolved=resolved,
        parameters=parameters,
        latent_expressions={},
        threshold_expressions=threshold_expressions,
    )

    term = result['ind']
    assert class_name(term) == 'OrderedProbit'
    text = expr_text(term)
    assert 'sigma_ord' in text
    assert 'OrderedProbit' in text


def test_build_measurement_terms_ml_ordered_logit():
    sigma_param = make_parameter(
        final_name='sigma_ord',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )
    parameters = {'sigma_ord': _beta_or_numeric(sigma_param)}

    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y_ord',
        measurement_model=MeasurementModel.ORDERED_LOGIT,
        sigma=SimpleNamespace(final_name='sigma_ord'),
        threshold_system_name='thr',
    )

    threshold_system = FakeThresholdSystem(
        cutpoints=[],
        categories=[1, 2, 3],
        neutral_labels=[],
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={'thr': threshold_system},
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    threshold_expressions = {'thr': [Numeric(-1), Numeric(1)]}

    result = _build_measurement_terms_ml(
        resolved=resolved,
        parameters=parameters,
        latent_expressions={},
        threshold_expressions=threshold_expressions,
    )

    term = result['ind']
    assert class_name(term) == 'OrderedLogit'
    text = expr_text(term)
    assert 'sigma_ord' in text
    assert 'OrderedLogit' in text


def test_build_measurement_terms_ml_non_gaussian_requires_sigma():
    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y_ord',
        measurement_model=MeasurementModel.ORDERED_LOGIT,
        sigma=None,
        threshold_system_name='thr',
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={
            'thr': FakeThresholdSystem(
                cutpoints=[],
                categories=[1, 2],
                neutral_labels=[],
            )
        },
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    with pytest.raises(ValueError, match='requires a resolved sigma parameter'):
        _build_measurement_terms_ml(
            resolved=resolved,
            parameters={},
            latent_expressions={},
            threshold_expressions={'thr': [Numeric(0)]},
        )


# ------------------------------------------------------------------------------
# Tests for _build_measurement_log_terms_bayesian
# ------------------------------------------------------------------------------


def test_build_measurement_log_terms_bayesian_gaussian():
    sigma_param = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )
    parameters = {'sigma_ind': _beta_or_numeric(sigma_param)}

    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(
            intercept=ResolvedConstant(0.5),
            terms=[FakeTerm(ResolvedConstant(1.5), 'x')],
        ),
        observed_variable_name='y_obs',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=SimpleNamespace(final_name='sigma_ind'),
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={},
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=bayesian_mode()),
        indicator_types={'likert': FakeIndicatorType(neutral_labels=[6, -1])},
    )

    result = _build_measurement_log_terms_bayesian(
        resolved=resolved,
        parameters=parameters,
        latent_expressions={},
        threshold_expressions={},
    )

    text = expr_text(result['ind'])
    assert 'y_obs' in text
    assert 'sigma_ind' in text
    assert class_name(result['ind']) == 'Elem'
    assert '6' in text
    assert '-1' in text


def test_build_measurement_log_terms_bayesian_ordered_probit():
    sigma_param = make_parameter(
        final_name='sigma_ord',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )
    parameters = {'sigma_ord': _beta_or_numeric(sigma_param)}

    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y_ord',
        measurement_model=MeasurementModel.ORDERED_PROBIT,
        sigma=SimpleNamespace(final_name='sigma_ord'),
        threshold_system_name='thr',
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={
            'thr': FakeThresholdSystem(
                cutpoints=[],
                categories=[1, 2, 3],
                neutral_labels=[],
            )
        },
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=bayesian_mode()),
    )

    result = _build_measurement_log_terms_bayesian(
        resolved=resolved,
        parameters=parameters,
        latent_expressions={},
        threshold_expressions={'thr': [Numeric(-1), Numeric(1)]},
    )

    assert class_name(result['ind']) == 'OrderedLogProbit'


def test_build_measurement_log_terms_bayesian_ordered_logit():
    sigma_param = make_parameter(
        final_name='sigma_ord',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )
    parameters = {'sigma_ord': _beta_or_numeric(sigma_param)}

    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y_ord',
        measurement_model=MeasurementModel.ORDERED_LOGIT,
        sigma=SimpleNamespace(final_name='sigma_ord'),
        threshold_system_name='thr',
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={
            'thr': FakeThresholdSystem(
                cutpoints=[],
                categories=[1, 2, 3],
                neutral_labels=[],
            )
        },
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=bayesian_mode()),
    )

    result = _build_measurement_log_terms_bayesian(
        resolved=resolved,
        parameters=parameters,
        latent_expressions={},
        threshold_expressions={'thr': [Numeric(-1), Numeric(1)]},
    )

    assert class_name(result['ind']) == 'OrderedLogLogit'


def test_build_measurement_log_terms_bayesian_requires_sigma():
    equation = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=None,
    )

    resolved = FakeResolvedModel(
        parameters={},
        latent_variables={},
        threshold_systems={},
        measurement_equations={'ind': equation},
        metadata=FakeMetadata(estimation_mode=bayesian_mode()),
    )

    with pytest.raises(ValueError, match='requires a resolved sigma parameter'):
        _build_measurement_log_terms_bayesian(
            resolved=resolved,
            parameters={},
            latent_expressions={},
            threshold_expressions={},
        )


# ------------------------------------------------------------------------------
# Tests for aggregate builders
# ------------------------------------------------------------------------------


def test_build_biogeme_model_ml_returns_expected_structure():
    measurement_terms = {
        'ind1': Variable('p1'),
        'ind2': Variable('p2'),
    }

    result = _build_biogeme_model_ml(
        parameters={'b': Beta('b', 0, None, None, 0)},
        estimated_parameter_names={'b': 'b'},
        parameter_groups={'Test group': ['b']},
        latent_expressions={'LV1': Variable('LV1')},
        threshold_expressions={'thr': [Numeric(-1), Numeric(1)]},
        measurement_terms=measurement_terms,
    )

    assert isinstance(result, BuiltBiogemeModel)
    assert result.estimated_parameter_names == {'b': 'b'}
    assert result.parameter_groups == {'Test group': ['b']}
    assert result.conditional_likelihood is not None
    assert result.integrated_likelihood is not None
    assert result.measurement_terms == measurement_terms
    assert 'p1' in expr_text(result.conditional_likelihood)
    assert 'p2' in expr_text(result.conditional_likelihood)
    assert 'log(' in expr_text(result.conditional_log_likelihood)


def test_build_biogeme_model_bayesian_returns_expected_structure():
    measurement_log_terms = {
        'ind1': Variable('ll1'),
        'ind2': Variable('ll2'),
    }

    result = _build_biogeme_model_bayesian(
        parameters={'b': Beta('b', 0, None, None, 0)},
        estimated_parameter_names={'b': 'b'},
        parameter_groups={'Test group': ['b']},
        latent_expressions={'LV1': Variable('LV1')},
        threshold_expressions={'thr': [Numeric(-1), Numeric(1)]},
        measurement_log_terms=measurement_log_terms,
    )

    assert isinstance(result, BuiltBiogemeModel)
    assert result.estimated_parameter_names == {'b': 'b'}
    assert result.parameter_groups == {'Test group': ['b']}
    assert result.conditional_likelihood is None
    assert result.integrated_likelihood is None
    assert result.measurement_terms == measurement_log_terms
    text = expr_text(result.conditional_log_likelihood)
    assert 'll1' in text
    assert 'll2' in text


# ------------------------------------------------------------------------------
# Tests for build_biogeme_model orchestration
# ------------------------------------------------------------------------------


def test_build_biogeme_model_ml_end_to_end():
    alpha_lv = make_parameter(
        final_name='alpha_lv',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=0.0,
        role=ParameterRole.STRUCTURAL_INTERCEPT,
    )
    beta_x = make_parameter(
        final_name='beta_x',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=0.5,
        role=ParameterRole.STRUCTURAL_COEFFICIENT,
    )
    sigma_lv = make_parameter(
        final_name='sigma_lv',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
        role=ParameterRole.STRUCTURAL_SIGMA,
    )
    sigma_ind = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
        role=ParameterRole.MEASUREMENT_SIGMA,
    )

    latent = FakeLatentVariable(
        structural_equation=FakeStructuralEquation(
            terms=[FakeTerm(SimpleNamespace(final_name='beta_x'), 'x')],
            sigma=SimpleNamespace(final_name='sigma_lv'),
            draw_name='omega',
            draw_type='NORMAL',
            intercept=SimpleNamespace(final_name='alpha_lv'),
        )
    )

    measurement_eq = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(
            intercept=ResolvedConstant(0.0),
            terms=[FakeTerm(ResolvedConstant(1.0), 'LV1')],
        ),
        observed_variable_name='indicator_1',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=SimpleNamespace(final_name='sigma_ind'),
    )

    resolved = FakeResolvedModel(
        parameters={
            'alpha_lv': alpha_lv,
            'beta_x': beta_x,
            'sigma_lv': sigma_lv,
            'sigma_ind': sigma_ind,
        },
        latent_variables={'LV1': latent},
        threshold_systems={},
        measurement_equations={'ind1': measurement_eq},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    result = build_biogeme_model(resolved)

    assert isinstance(result, BuiltBiogemeModel)
    assert result.conditional_likelihood is not None
    assert result.integrated_likelihood is not None
    assert 'LV1' in result.latent_expressions
    assert 'ind1' in result.measurement_terms

    latent_text = expr_text(result.latent_expressions['LV1'])
    assert 'alpha_lv' in latent_text
    assert 'beta_x' in latent_text
    assert 'omega' in latent_text
    assert 'sigma_lv' in latent_text

    measurement_text = expr_text(result.measurement_terms['ind1'])
    assert 'indicator_1' in measurement_text

    assert result.estimated_parameter_names == {
        'alpha_lv': 'alpha_lv',
        'beta_x': 'beta_x',
        'sigma_lv': 'sigma_lv',
        'sigma_ind': 'sigma_ind',
    }
    assert result.parameter_groups == {
        'Structural equation': ['alpha_lv', 'beta_x', 'sigma_lv'],
        'Measurement equation: ind1': ['sigma_ind'],
    }


def test_build_biogeme_model_bayesian_wraps_latent_expression_as_distributed_parameter():
    alpha_lv = make_parameter(
        final_name='alpha_lv',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=0.0,
        role=ParameterRole.STRUCTURAL_INTERCEPT,
    )
    beta_x = make_parameter(
        final_name='beta_x',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=0.5,
        role=ParameterRole.STRUCTURAL_COEFFICIENT,
    )
    sigma_lv = make_parameter(
        final_name='sigma_lv',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
        role=ParameterRole.STRUCTURAL_SIGMA,
    )
    sigma_ind = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
        role=ParameterRole.MEASUREMENT_SIGMA,
    )

    latent = FakeLatentVariable(
        structural_equation=FakeStructuralEquation(
            terms=[FakeTerm(SimpleNamespace(final_name='beta_x'), 'x')],
            sigma=SimpleNamespace(final_name='sigma_lv'),
            draw_name='omega',
            draw_type='NORMAL',
            intercept=SimpleNamespace(final_name='alpha_lv'),
        )
    )

    measurement_eq = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(
            intercept=ResolvedConstant(0.0),
            terms=[FakeTerm(ResolvedConstant(1.0), 'LV1')],
        ),
        observed_variable_name='indicator_1',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=SimpleNamespace(final_name='sigma_ind'),
    )

    resolved = FakeResolvedModel(
        parameters={
            'alpha_lv': alpha_lv,
            'beta_x': beta_x,
            'sigma_lv': sigma_lv,
            'sigma_ind': sigma_ind,
        },
        latent_variables={'LV1': latent},
        threshold_systems={},
        measurement_equations={'ind1': measurement_eq},
        metadata=FakeMetadata(estimation_mode=bayesian_mode()),
    )

    result = build_biogeme_model(resolved)

    assert result.conditional_likelihood is None
    assert result.integrated_likelihood is None
    assert result.parameter_groups == {
        'Structural equation': ['alpha_lv', 'beta_x', 'sigma_lv'],
        'Measurement equation: ind1': ['sigma_ind'],
    }

    latent_text = expr_text(result.latent_expressions['LV1'])
    assert 'alpha_lv' in latent_text
    assert 'LV1' in latent_text
    assert 'omega' in latent_text
    assert 'sigma_lv' in latent_text

    ll_text = expr_text(result.conditional_log_likelihood)
    assert 'indicator_1' in ll_text


def test_build_biogeme_model_builds_threshold_expressions_sequentially():
    sigma_ind = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )

    measurement_eq = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y_ord',
        measurement_model=MeasurementModel.ORDERED_LOGIT,
        sigma=SimpleNamespace(final_name='sigma_ind'),
        threshold_system_name='thr',
    )

    threshold_system = FakeThresholdSystem(
        cutpoints=[
            FakeCutpoint(symbol_name='tau1', expression_text='-1'),
            FakeCutpoint(symbol_name='tau2', expression_text='tau1 + 2'),
        ],
        categories=[1, 2, 3],
        neutral_labels=[],
    )

    resolved = FakeResolvedModel(
        parameters={'sigma_ind': sigma_ind},
        latent_variables={},
        threshold_systems={'thr': threshold_system},
        measurement_equations={'ind1': measurement_eq},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    result = build_biogeme_model(resolved)

    assert 'thr' in result.threshold_expressions
    assert len(result.threshold_expressions['thr']) == 2

    tau1_text = expr_text(result.threshold_expressions['thr'][0])
    tau2_text = expr_text(result.threshold_expressions['thr'][1])

    assert '-1' in tau1_text
    assert '1' in tau2_text or '+ 2' in tau2_text


def test_build_biogeme_model_with_zero_sigma_in_latent_equation():
    sigma_ind = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )

    latent = FakeLatentVariable(
        structural_equation=FakeStructuralEquation(
            terms=[FakeTerm(ResolvedConstant(2.0), 'x')],
            sigma=None,
            draw_name='omega',
            draw_type='NORMAL',
        )
    )

    measurement_eq = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(
            intercept=None,
            terms=[FakeTerm(ResolvedConstant(1.0), 'LV1')],
        ),
        observed_variable_name='y',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=SimpleNamespace(final_name='sigma_ind'),
    )

    resolved = FakeResolvedModel(
        parameters={'sigma_ind': sigma_ind},
        latent_variables={'LV1': latent},
        threshold_systems={},
        measurement_equations={'ind': measurement_eq},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    result = build_biogeme_model(resolved)

    latent_text = expr_text(result.latent_expressions['LV1'])
    assert 'x' in latent_text
    assert 'omega' in latent_text


# ------------------------------------------------------------------------------
# Defensive regression tests
# ------------------------------------------------------------------------------


def test_build_biogeme_model_passes_latent_expressions_to_measurements():
    sigma_lv = make_parameter(
        final_name='sigma_lv',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )
    sigma_ind = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
    )

    latent = FakeLatentVariable(
        structural_equation=FakeStructuralEquation(
            terms=[],
            sigma=SimpleNamespace(final_name='sigma_lv'),
            draw_name='omega',
            draw_type='NORMAL',
        )
    )

    measurement_eq = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(
            intercept=None,
            terms=[FakeTerm(ResolvedConstant(1.0), 'LV1')],
        ),
        observed_variable_name='ind_obs',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=SimpleNamespace(final_name='sigma_ind'),
    )

    resolved = FakeResolvedModel(
        parameters={'sigma_lv': sigma_lv, 'sigma_ind': sigma_ind},
        latent_variables={'LV1': latent},
        threshold_systems={},
        measurement_equations={'ind': measurement_eq},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    result = build_biogeme_model(resolved)

    text = expr_text(result.measurement_terms['ind'])
    assert 'LV1' in text or 'omega' in text


def test_build_biogeme_model_returns_parameter_dictionary_with_rendered_expressions():
    c = make_parameter(
        final_name='c',
        creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
        fixed_value=2.0,
    )

    sigma_ind = make_parameter(
        final_name='sigma_ind',
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=1.0,
        role=ParameterRole.MEASUREMENT_SIGMA,
    )

    measurement_eq = FakeMeasurementEquation(
        systematic_part=ResolvedLinearCombination(None, []),
        observed_variable_name='y',
        measurement_model=MeasurementModel.GAUSSIAN,
        sigma=SimpleNamespace(final_name='sigma_ind'),
    )

    resolved = FakeResolvedModel(
        parameters={'c': c, 'sigma_ind': sigma_ind},
        latent_variables={},
        threshold_systems={},
        measurement_equations={'ind': measurement_eq},
        metadata=FakeMetadata(estimation_mode=ml_mode()),
    )

    result = build_biogeme_model(resolved)

    assert isinstance(result.parameters['c'], Numeric)
    assert isinstance(result.parameters['sigma_ind'], Beta)
    assert result.estimated_parameter_names == {
        'sigma_ind': 'sigma_ind',
    }
    assert result.parameter_groups == {
        'Measurement equation: ind': ['sigma_ind'],
    }
