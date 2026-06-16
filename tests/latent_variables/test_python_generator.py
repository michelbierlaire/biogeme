from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from biogeme.latent_variables.context import EstimationMode
from biogeme.latent_variables.model_spec import MeasurementModel
from biogeme.latent_variables.python_generator import (
    _combo_to_python,
    _emit_header,
    _emit_measurement_log_terms_bayesian,
    _emit_measurement_terms_ml,
    _emit_parameter_assignment,
    _emit_parameters,
    _emit_threshold_systems,
    _generate_python_code_bayesian,
    _generate_python_code_ml,
    _term_to_python,
    generate_python_code,
    save_python_code,
)
from biogeme.latent_variables.resolved import ParameterCreationKind


def _param(
    *,
    creation_kind: ParameterCreationKind,
    initial_value: float | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    fixed_value: float | None = None,
    final_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        creation_kind=creation_kind,
        initial_value=initial_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        fixed_value=fixed_value,
        final_name=final_name,
    )


def _constant(value: object) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _named(final_name: str) -> SimpleNamespace:
    return SimpleNamespace(final_name=final_name)


def _term(coefficient: object, variable_name: str) -> SimpleNamespace:
    return SimpleNamespace(coefficient=coefficient, variable_name=variable_name)


def _combo(intercept: object | None, terms: list[object]) -> SimpleNamespace:
    return SimpleNamespace(intercept=intercept, terms=terms)


def _cutpoint(
    symbol_name: str,
    expression_text: str,
    source_parameter_names: list[str],
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol_name=symbol_name,
        expression_text=expression_text,
        source_parameter_names=source_parameter_names,
    )


def _threshold_system(
    cutpoints: list[object],
    *,
    categories: list[int] | None = None,
    neutral_labels: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cutpoints=cutpoints,
        categories=[1, 2, 3] if categories is None else categories,
        neutral_labels=[] if neutral_labels is None else neutral_labels,
    )


def _measurement_equation(
    *,
    systematic_part: object,
    observed_variable_name: str,
    sigma: object | None,
    measurement_model: MeasurementModel,
    threshold_system_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        systematic_part=systematic_part,
        observed_variable_name=observed_variable_name,
        sigma=sigma,
        measurement_model=measurement_model,
        threshold_system_name=threshold_system_name,
    )


def _structural_equation(
    *,
    intercept: object | None,
    terms: list[object],
    draw_name: str,
    draw_type: str,
    sigma: object | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        systematic_part=_combo(intercept, terms),
        draw_name=draw_name,
        draw_type=draw_type,
        sigma=sigma,
    )


def _latent_variable(structural_equation: object) -> SimpleNamespace:
    return SimpleNamespace(structural_equation=structural_equation)


def _resolved_model(
    *,
    estimation_mode: EstimationMode = EstimationMode.MAXIMUM_LIKELIHOOD,
    parameters: dict[str, object] | None = None,
    threshold_systems: dict[str, object] | None = None,
    measurement_equations: dict[str, object] | None = None,
    latent_variables: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(estimation_mode=estimation_mode),
        parameters={} if parameters is None else parameters,
        threshold_systems={} if threshold_systems is None else threshold_systems,
        measurement_equations=(
            {} if measurement_equations is None else measurement_equations
        ),
        latent_variables={} if latent_variables is None else latent_variables,
    )


# ------------------------------------------------------------------------------
# _emit_parameter_assignment
# ------------------------------------------------------------------------------


def test_emit_parameter_assignment_numeric_constant() -> None:
    param = _param(
        creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
        fixed_value=3.5,
    )

    assert _emit_parameter_assignment('alpha', param) == ['alpha = 3.5']


def test_emit_parameter_assignment_log_exp_beta() -> None:
    param = _param(
        creation_kind=ParameterCreationKind.LOG_EXP_BETA,
        initial_value=1.2,
    )

    assert _emit_parameter_assignment('sigma', param) == [
        'sigma_log = Beta("sigma_log", 1.2, None, None, 0)',
        'sigma = exp(sigma_log)',
    ]


def test_emit_parameter_assignment_bounded_beta() -> None:
    param = _param(
        creation_kind=ParameterCreationKind.BOUNDED_BETA,
        initial_value=0.0,
        lower_bound=-1.0,
        upper_bound=2.0,
    )

    assert _emit_parameter_assignment('beta_b', param) == [
        'beta_b = Beta("beta_b", 0.0, -1.0, 2.0, 0)'
    ]


def test_emit_parameter_assignment_free_beta() -> None:
    param = _param(
        creation_kind=ParameterCreationKind.FREE_BETA,
        initial_value=0.5,
        lower_bound=None,
        upper_bound=None,
    )

    assert _emit_parameter_assignment('beta_f', param) == [
        'beta_f = Beta("beta_f", 0.5, None, None, 0)'
    ]


def test_emit_parameter_assignment_fallback_branch_for_fixed_parameter() -> None:
    param = _param(
        creation_kind=ParameterCreationKind.FIXED_BETA,
        fixed_value=7.0,
    )

    assert _emit_parameter_assignment('beta_x', param) == ['beta_x = 7.0']


# ------------------------------------------------------------------------------
# _term_to_python
# ------------------------------------------------------------------------------


def test_term_to_python_uses_symbol_name_when_variable_is_known_symbol() -> None:
    term = _term(_named('beta'), 'LV1')

    result = _term_to_python(term, {'LV1'})

    assert result == 'beta * LV1'


def test_term_to_python_uses_variable_expression_when_not_a_symbol() -> None:
    term = _term(_named('beta'), 'income')

    result = _term_to_python(term, {'LV1'})

    assert result == 'beta * Variable("income")'


def test_term_to_python_uses_variable_expression_when_symbol_names_is_none() -> None:
    term = _term(_named('beta'), 'income')

    result = _term_to_python(term)

    assert result == 'beta * Variable("income")'


def test_term_to_python_uses_constant_value_when_coefficient_has_no_final_name() -> (
    None
):
    term = _term(_constant(2.5), 'x')

    result = _term_to_python(term)

    assert result == '2.5 * Variable("x")'


# ------------------------------------------------------------------------------
# _combo_to_python
# ------------------------------------------------------------------------------


def test_combo_to_python_returns_numeric_zero_for_empty_combination() -> None:
    combo = _combo(None, [])

    assert _combo_to_python(combo) == 'Numeric(0.0)'


def test_combo_to_python_with_named_intercept_and_terms() -> None:
    combo = _combo(
        _named('alpha'),
        [
            _term(_named('beta1'), 'x1'),
            _term(_constant(3.0), 'x2'),
        ],
    )

    assert _combo_to_python(combo) == (
        'alpha + beta1 * Variable("x1") + 3.0 * Variable("x2")'
    )


def test_combo_to_python_with_constant_intercept_and_symbol_term() -> None:
    combo = _combo(
        _constant(1.5),
        [_term(_named('lambda_lv'), 'LV1')],
    )

    assert _combo_to_python(combo, {'LV1'}) == '1.5 + lambda_lv * LV1'


# ------------------------------------------------------------------------------
# _emit_header
# ------------------------------------------------------------------------------


def test_emit_header_ml() -> None:
    lines: list[str] = []

    _emit_header(lines, _resolved_model(), bayesian=False)

    assert lines == [
        '"""Pedagogical runnable Biogeme code for the latent-variable part of the model."""',
        '',
        'from biogeme.expressions import Beta, Draws, MonteCarlo, MultipleProduct, MultipleSum, Numeric, OrderedLogit, OrderedProbit, Variable, exp, log',
        'from biogeme.distributions import normalpdf',
        '',
    ]


def test_emit_header_bayesian() -> None:
    lines: list[str] = []

    _emit_header(lines, _resolved_model(), bayesian=True)

    assert lines == [
        '"""Pedagogical runnable Biogeme code for the latent-variable part of the model."""',
        '',
        'from biogeme.expressions import Beta, DistributedParameter, Draws, MultipleSum, Numeric, OrderedLogLogit, OrderedLogProbit, Variable, exp',
        'from biogeme.distributions import normal_logpdf',
        '',
    ]


# ------------------------------------------------------------------------------
# _emit_parameters
# ------------------------------------------------------------------------------


def test_emit_parameters_emits_sorted_parameters_and_trailing_blank_line() -> None:
    resolved = _resolved_model(
        parameters={
            'zeta': _param(
                creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
                fixed_value=9.0,
            ),
            'alpha': _param(
                creation_kind=ParameterCreationKind.FREE_BETA,
                initial_value=1.0,
                lower_bound=None,
                upper_bound=None,
            ),
        }
    )
    lines: list[str] = []

    _emit_parameters(lines, resolved)

    assert lines == [
        '# ---------------------------------------------------------------------------',
        '# Parameters',
        '# ---------------------------------------------------------------------------',
        'alpha = Beta("alpha", 1.0, None, None, 0)',
        'zeta = 9.0',
        '',
    ]


# ------------------------------------------------------------------------------
# _emit_threshold_systems
# ------------------------------------------------------------------------------


def test_emit_threshold_systems_returns_immediately_when_none() -> None:
    resolved = _resolved_model(threshold_systems={})
    lines = ['before']

    _emit_threshold_systems(lines, resolved)

    assert lines == ['before']


def test_emit_threshold_systems_emits_and_rewrites_tau_sources_before_shorter_names() -> (
    None
):
    resolved = _resolved_model(
        threshold_systems={
            'likert5': _threshold_system(
                [
                    _cutpoint(
                        'tau_1',
                        'tau_10 + tau_1 + alpha + tau_2',
                        ['tau_1', 'tau_10', 'alpha', 'tau_2'],
                    )
                ]
            )
        }
    )
    lines: list[str] = []

    _emit_threshold_systems(lines, resolved)

    assert lines == [
        '# ---------------------------------------------------------------------------',
        '# Threshold systems',
        '# ---------------------------------------------------------------------------',
        '# Threshold system: likert5',
        'likert5_tau_1 = likert5_likert5_tau_10 + likert5_tau_1 + alpha + likert5_tau_2',
        '',
    ]


# ------------------------------------------------------------------------------
# _emit_measurement_terms_ml
# ------------------------------------------------------------------------------


def test_emit_measurement_terms_ml_gaussian_and_ordered_models() -> None:
    resolved = _resolved_model(
        threshold_systems={
            'scale5': _threshold_system(
                [
                    _cutpoint('tau_1', 'x', []),
                    _cutpoint('tau_2', 'y', []),
                ],
                categories=[1, 2, 3],
                neutral_labels=[2],
            )
        },
        latent_variables={'LV1': object()},
        measurement_equations={
            'gauss_ind': _measurement_equation(
                systematic_part=_combo(
                    _named('alpha_g'),
                    [_term(_named('lambda_g'), 'LV1')],
                ),
                observed_variable_name='obs_g',
                sigma=_named('sigma_g'),
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
            'probit_ind': _measurement_equation(
                systematic_part=_combo(None, []),
                observed_variable_name='obs_p',
                sigma=_named('sigma_p'),
                measurement_model=MeasurementModel.ORDERED_PROBIT,
                threshold_system_name='scale5',
            ),
            'logit_ind': _measurement_equation(
                systematic_part=_combo(_constant(1), []),
                observed_variable_name='obs_l',
                sigma=_named('sigma_l'),
                measurement_model=MeasurementModel.ORDERED_LOGIT,
                threshold_system_name='scale5',
            ),
        },
    )
    lines: list[str] = []

    _emit_measurement_terms_ml(lines, resolved)

    assert lines == [
        '# ---------------------------------------------------------------------------',
        '# Measurement equations and likelihood terms',
        '# ---------------------------------------------------------------------------',
        '# Indicator: gauss_ind',
        'mu_gauss_ind = alpha_g + lambda_g * LV1',
        'y_gauss_ind = Variable("obs_g")',
        'term_gauss_ind = normalpdf((y_gauss_ind - mu_gauss_ind) / sigma_g) / sigma_g',
        '',
        '# Indicator: probit_ind',
        'mu_probit_ind = Numeric(0.0)',
        'y_probit_ind = Variable("obs_p")',
        'term_probit_ind = OrderedProbit(eta=mu_probit_ind / sigma_p, cutpoints=[scale5_tau_1 / sigma_p, scale5_tau_2 / sigma_p], y=y_probit_ind, categories=[1, 2, 3], neutral_labels=[2])',
        '',
        '# Indicator: logit_ind',
        'mu_logit_ind = 1',
        'y_logit_ind = Variable("obs_l")',
        'term_logit_ind = OrderedLogit(eta=mu_logit_ind / sigma_l, cutpoints=[scale5_tau_1 / sigma_l, scale5_tau_2 / sigma_l], y=y_logit_ind, categories=[1, 2, 3], neutral_labels=[2])',
        '',
    ]


def test_emit_measurement_terms_ml_raises_when_sigma_missing() -> None:
    resolved = _resolved_model(
        measurement_equations={
            'indicator_x': _measurement_equation(
                systematic_part=_combo(None, []),
                observed_variable_name='obs',
                sigma=None,
                measurement_model=MeasurementModel.GAUSSIAN,
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="Measurement equation for indicator 'indicator_x' is missing a resolved sigma parameter.",
    ):
        _emit_measurement_terms_ml([], resolved)


# ------------------------------------------------------------------------------
# _emit_measurement_log_terms_bayesian
# ------------------------------------------------------------------------------


def test_emit_measurement_log_terms_bayesian_gaussian_and_ordered_models() -> None:
    resolved = _resolved_model(
        threshold_systems={
            'scale5': _threshold_system(
                [
                    _cutpoint('tau_1', 'x', []),
                    _cutpoint('tau_2', 'y', []),
                ],
                categories=[1, 2, 3],
                neutral_labels=[2],
            )
        },
        latent_variables={'LV1': object()},
        measurement_equations={
            'gauss_ind': _measurement_equation(
                systematic_part=_combo(
                    _named('alpha_g'),
                    [_term(_named('lambda_g'), 'LV1')],
                ),
                observed_variable_name='obs_g',
                sigma=_named('sigma_g'),
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
            'probit_ind': _measurement_equation(
                systematic_part=_combo(None, []),
                observed_variable_name='obs_p',
                sigma=_named('sigma_p'),
                measurement_model=MeasurementModel.ORDERED_PROBIT,
                threshold_system_name='scale5',
            ),
            'logit_ind': _measurement_equation(
                systematic_part=_combo(_constant(1), []),
                observed_variable_name='obs_l',
                sigma=_named('sigma_l'),
                measurement_model=MeasurementModel.ORDERED_LOGIT,
                threshold_system_name='scale5',
            ),
        },
    )
    lines: list[str] = []

    _emit_measurement_log_terms_bayesian(lines, resolved)

    assert lines == [
        '# ---------------------------------------------------------------------------',
        '# Measurement equations and log-likelihood terms',
        '# ---------------------------------------------------------------------------',
        '# Indicator: gauss_ind',
        'mu_gauss_ind = alpha_g + lambda_g * LV1',
        'y_gauss_ind = Variable("obs_g")',
        'log_term_gauss_ind = normal_logpdf(y_gauss_ind, mu_gauss_ind, sigma_g)',
        '',
        '# Indicator: probit_ind',
        'mu_probit_ind = Numeric(0.0)',
        'y_probit_ind = Variable("obs_p")',
        'log_term_probit_ind = OrderedLogProbit(eta=mu_probit_ind / sigma_p, cutpoints=[scale5_tau_1 / sigma_p, scale5_tau_2 / sigma_p], y=y_probit_ind, categories=[1, 2, 3], neutral_labels=[2])',
        '',
        '# Indicator: logit_ind',
        'mu_logit_ind = 1',
        'y_logit_ind = Variable("obs_l")',
        'log_term_logit_ind = OrderedLogLogit(eta=mu_logit_ind / sigma_l, cutpoints=[scale5_tau_1 / sigma_l, scale5_tau_2 / sigma_l], y=y_logit_ind, categories=[1, 2, 3], neutral_labels=[2])',
        '',
    ]


def test_emit_measurement_log_terms_bayesian_raises_when_sigma_missing() -> None:
    resolved = _resolved_model(
        measurement_equations={
            'indicator_x': _measurement_equation(
                systematic_part=_combo(None, []),
                observed_variable_name='obs',
                sigma=None,
                measurement_model=MeasurementModel.GAUSSIAN,
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="Measurement equation for indicator 'indicator_x' is missing a resolved sigma parameter.",
    ):
        _emit_measurement_log_terms_bayesian([], resolved)


# ------------------------------------------------------------------------------
# _generate_python_code_ml
# ------------------------------------------------------------------------------


def test_generate_python_code_ml_with_full_content() -> None:
    resolved = _resolved_model(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        parameters={
            'alpha': _param(
                creation_kind=ParameterCreationKind.FREE_BETA,
                initial_value=0.0,
                lower_bound=None,
                upper_bound=None,
            ),
            'sigma_m': _param(
                creation_kind=ParameterCreationKind.LOG_EXP_BETA,
                initial_value=1.0,
            ),
        },
        latent_variables={
            'LV1': _latent_variable(
                _structural_equation(
                    intercept=_named('alpha_lv1'),
                    terms=[
                        _term(_named('beta1'), 'x1'),
                        _term(_constant(2.0), 'x2'),
                    ],
                    draw_name='draw_lv1',
                    draw_type='NORMAL',
                    sigma=_named('sigma_struct'),
                )
            ),
            'LV2': _latent_variable(
                _structural_equation(
                    intercept=_named('alpha_lv2'),
                    terms=[],
                    draw_name='draw_lv2',
                    draw_type='UNIFORM',
                    sigma=None,
                )
            ),
        },
        threshold_systems={
            'scale5': _threshold_system(
                [_cutpoint('tau_1', 'tau_1 + alpha', ['tau_1', 'alpha'])],
                categories=[1, 2, 3],
                neutral_labels=[2],
            )
        },
        measurement_equations={
            'ind_a': _measurement_equation(
                systematic_part=_combo(
                    _named('alpha'),
                    [_term(_named('lambda_a'), 'LV1')],
                ),
                observed_variable_name='obs_a',
                sigma=_named('sigma_m'),
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
            'ind_b': _measurement_equation(
                systematic_part=_combo(None, []),
                observed_variable_name='obs_b',
                sigma=_named('sigma_b'),
                measurement_model=MeasurementModel.ORDERED_LOGIT,
                threshold_system_name='scale5',
            ),
        },
    )

    code = _generate_python_code_ml(resolved)

    expected = (
        '"""Pedagogical runnable Biogeme code for the latent-variable part of the model."""\n'
        '\n'
        'from biogeme.expressions import Beta, Draws, MonteCarlo, MultipleProduct, MultipleSum, Numeric, OrderedLogit, OrderedProbit, Variable, exp, log\n'
        'from biogeme.distributions import normalpdf\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Parameters\n'
        '# ---------------------------------------------------------------------------\n'
        'alpha = Beta("alpha", 0.0, None, None, 0)\n'
        'sigma_m_log = Beta("sigma_m_log", 1.0, None, None, 0)\n'
        'sigma_m = exp(sigma_m_log)\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Structural equations\n'
        '# ---------------------------------------------------------------------------\n'
        'mu_LV1 = alpha_lv1 + beta1 * Variable("x1") + 2.0 * Variable("x2")\n'
        'draw_LV1 = Draws("draw_lv1", draw_type="NORMAL")\n'
        'LV1 = mu_LV1 + sigma_struct * draw_LV1\n'
        '\n'
        'mu_LV2 = alpha_lv2\n'
        'draw_LV2 = Draws("draw_lv2", draw_type="UNIFORM")\n'
        'LV2 = mu_LV2\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Threshold systems\n'
        '# ---------------------------------------------------------------------------\n'
        '# Threshold system: scale5\n'
        'scale5_tau_1 = scale5_tau_1 + alpha\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Measurement equations and likelihood terms\n'
        '# ---------------------------------------------------------------------------\n'
        '# Indicator: ind_a\n'
        'mu_ind_a = alpha + lambda_a * LV1\n'
        'y_ind_a = Variable("obs_a")\n'
        'term_ind_a = normalpdf((y_ind_a - mu_ind_a) / sigma_m) / sigma_m\n'
        '\n'
        '# Indicator: ind_b\n'
        'mu_ind_b = Numeric(0.0)\n'
        'y_ind_b = Variable("obs_b")\n'
        'term_ind_b = OrderedLogit(eta=mu_ind_b / sigma_b, cutpoints=[scale5_tau_1 / sigma_b], y=y_ind_b, categories=[1, 2, 3], neutral_labels=[2])\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Conditional indicator likelihood and Monte Carlo integration\n'
        '# ---------------------------------------------------------------------------\n'
        'conditional_measurement_likelihood = MultipleProduct([term_ind_a, term_ind_b])\n'
        'conditional_log_likelihood = MultipleSum([log(term) for term in [term_ind_a, term_ind_b] ])\n'
        'integrated_measurement_likelihood = MonteCarlo(conditional_measurement_likelihood)\n'
    )

    assert code == expected


def test_generate_python_code_ml_without_measurements() -> None:
    resolved = _resolved_model(
        estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD,
        latent_variables={},
        measurement_equations={},
    )

    code = _generate_python_code_ml(resolved)

    assert 'conditional_measurement_likelihood = Numeric(1.0)\n' in code
    assert 'conditional_log_likelihood = Numeric(0.0)\n' in code
    assert 'integrated_measurement_likelihood = Numeric(1.0)\n' in code


# ------------------------------------------------------------------------------
# _generate_python_code_bayesian
# ------------------------------------------------------------------------------


def test_generate_python_code_bayesian_with_full_content() -> None:
    resolved = _resolved_model(
        estimation_mode=EstimationMode.BAYESIAN,
        parameters={
            'beta': _param(
                creation_kind=ParameterCreationKind.NUMERIC_CONSTANT,
                fixed_value=2.0,
            ),
        },
        latent_variables={
            'LV1': _latent_variable(
                _structural_equation(
                    intercept=_named('alpha_lv1'),
                    terms=[_term(_named('beta1'), 'x1')],
                    draw_name='draw_lv1',
                    draw_type='NORMAL_HALTON',
                    sigma=_named('sigma_struct'),
                )
            ),
            'LV2': _latent_variable(
                _structural_equation(
                    intercept=_named('alpha_lv2'),
                    terms=[],
                    draw_name='draw_lv2',
                    draw_type='UNIFORM',
                    sigma=None,
                )
            ),
        },
        threshold_systems={
            'scale5': _threshold_system(
                [_cutpoint('tau_1', 'alpha', ['alpha'])],
                categories=[1, 2],
                neutral_labels=[],
            )
        },
        measurement_equations={
            'ind_a': _measurement_equation(
                systematic_part=_combo(
                    _named('alpha'), [_term(_named('lambda_a'), 'LV1')]
                ),
                observed_variable_name='obs_a',
                sigma=_named('sigma_a'),
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
            'ind_b': _measurement_equation(
                systematic_part=_combo(None, []),
                observed_variable_name='obs_b',
                sigma=_named('sigma_b'),
                measurement_model=MeasurementModel.ORDERED_PROBIT,
                threshold_system_name='scale5',
            ),
        },
    )

    code = _generate_python_code_bayesian(resolved)

    expected = (
        '"""Pedagogical runnable Biogeme code for the latent-variable part of the model."""\n'
        '\n'
        'from biogeme.expressions import Beta, DistributedParameter, Draws, MultipleSum, Numeric, OrderedLogLogit, OrderedLogProbit, Variable, exp\n'
        'from biogeme.distributions import normal_logpdf\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Parameters\n'
        '# ---------------------------------------------------------------------------\n'
        'beta = 2.0\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Structural equations\n'
        '# ---------------------------------------------------------------------------\n'
        'mu_LV1 = alpha_lv1 + beta1 * Variable("x1")\n'
        'draw_LV1 = Draws("draw_lv1", draw_type="NORMAL_HALTON")\n'
        'stochastic_LV1 = mu_LV1 + sigma_struct * draw_LV1\n'
        'LV1 = DistributedParameter("LV1", stochastic_LV1)\n'
        '\n'
        'mu_LV2 = alpha_lv2\n'
        'draw_LV2 = Draws("draw_lv2", draw_type="UNIFORM")\n'
        'LV2 = DistributedParameter("LV2", mu_LV2)\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Threshold systems\n'
        '# ---------------------------------------------------------------------------\n'
        '# Threshold system: scale5\n'
        'scale5_tau_1 = alpha\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Measurement equations and log-likelihood terms\n'
        '# ---------------------------------------------------------------------------\n'
        '# Indicator: ind_a\n'
        'mu_ind_a = alpha + lambda_a * LV1\n'
        'y_ind_a = Variable("obs_a")\n'
        'log_term_ind_a = normal_logpdf(y_ind_a, mu_ind_a, sigma_a)\n'
        '\n'
        '# Indicator: ind_b\n'
        'mu_ind_b = Numeric(0.0)\n'
        'y_ind_b = Variable("obs_b")\n'
        'log_term_ind_b = OrderedLogProbit(eta=mu_ind_b / sigma_b, cutpoints=[scale5_tau_1 / sigma_b], y=y_ind_b, categories=[1, 2], neutral_labels=[])\n'
        '\n'
        '# ---------------------------------------------------------------------------\n'
        '# Conditional log-likelihood\n'
        '# ---------------------------------------------------------------------------\n'
        'conditional_log_likelihood = MultipleSum([log_term_ind_a, log_term_ind_b])\n'
    )

    assert code == expected


def test_generate_python_code_bayesian_without_measurements() -> None:
    resolved = _resolved_model(
        estimation_mode=EstimationMode.BAYESIAN,
        latent_variables={},
        measurement_equations={},
    )

    code = _generate_python_code_bayesian(resolved)

    assert 'conditional_log_likelihood = Numeric(0.0)\n' in code


# ------------------------------------------------------------------------------
# public API
# ------------------------------------------------------------------------------


def test_generate_python_code_dispatches_to_bayesian() -> None:
    resolved = _resolved_model(estimation_mode=EstimationMode.BAYESIAN)

    code = generate_python_code(resolved)

    assert code == _generate_python_code_bayesian(resolved)


def test_generate_python_code_dispatches_to_ml_for_non_bayesian() -> None:
    resolved = _resolved_model(estimation_mode=EstimationMode.MAXIMUM_LIKELIHOOD)

    code = generate_python_code(resolved)

    assert code == _generate_python_code_ml(resolved)


def test_save_python_code_writes_utf8_text_to_file(tmp_path: Path) -> None:
    path = tmp_path / 'generated.py'
    code = "# -*- coding: utf-8 -*-\nprint('café')\n"

    save_python_code(code, path)

    assert path.read_text(encoding='utf-8') == code


def test_save_python_code_accepts_string_path(tmp_path: Path) -> None:
    path = tmp_path / 'generated_string_path.py'
    code = 'x = 1\n'

    save_python_code(code, str(path))

    assert path.read_text(encoding='utf-8') == code
