from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from biogeme.latent_variables.latex_report import (
    _combo_to_latex,
    generate_latex_report,
    save_latex_report,
)
from biogeme.latent_variables.model_spec import MeasurementModel


def _resolved_constant(value: object) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _resolved_parameter(
    final_name: str,
    *,
    lower_bound: object = None,
    upper_bound: object = None,
    role: str = 'generic_role',
    status: str = 'generic_status',
    notes: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        final_name=final_name,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        role=SimpleNamespace(value=role),
        status=SimpleNamespace(value=status),
        notes=[] if notes is None else notes,
    )


def _term(coefficient: object, variable_name: str) -> SimpleNamespace:
    return SimpleNamespace(coefficient=coefficient, variable_name=variable_name)


def _linear_combination(
    *,
    intercept: object | None,
    terms: list[SimpleNamespace],
) -> SimpleNamespace:
    return SimpleNamespace(intercept=intercept, terms=terms)


def _structural_equation(
    *,
    intercept: object | None,
    terms: list[SimpleNamespace],
    sigma: object | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        systematic_part=_linear_combination(intercept=intercept, terms=terms),
        sigma=sigma,
    )


def _latent_variable(
    *,
    intercept: object | None,
    terms: list[SimpleNamespace],
    sigma: object | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        structural_equation=_structural_equation(
            intercept=intercept,
            terms=terms,
            sigma=sigma,
        )
    )


def _measurement_equation(
    *,
    systematic_part: SimpleNamespace,
    sigma: object | None,
    measurement_model: MeasurementModel,
) -> SimpleNamespace:
    return SimpleNamespace(
        systematic_part=systematic_part,
        sigma=sigma,
        measurement_model=measurement_model,
    )


def _threshold_system(cutpoints: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(cutpoints=cutpoints)


def _cutpoint(symbol_name: str, expression_text: str) -> SimpleNamespace:
    return SimpleNamespace(symbol_name=symbol_name, expression_text=expression_text)


def _normalization_rule(
    reason: str, target_name: str, value: object
) -> SimpleNamespace:
    return SimpleNamespace(reason=reason, target_name=target_name, value=value)


def _resolved_model(
    *,
    metadata: SimpleNamespace | None = None,
    latent_variables: dict[str, object] | None = None,
    measurement_equations: dict[str, object] | None = None,
    threshold_systems: dict[str, object] | None = None,
    normalization: SimpleNamespace | None = None,
    parameters: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=(
            metadata
            if metadata is not None
            else SimpleNamespace(
                n_latent_variables=0,
                n_indicators=0,
                n_threshold_systems=0,
            )
        ),
        latent_variables={} if latent_variables is None else latent_variables,
        measurement_equations=(
            {} if measurement_equations is None else measurement_equations
        ),
        threshold_systems={} if threshold_systems is None else threshold_systems,
        normalization=(
            normalization
            if normalization is not None
            else SimpleNamespace(rules=[], warnings=[])
        ),
        parameters={} if parameters is None else parameters,
    )


@pytest.fixture
def patched_tex(monkeypatch):
    import biogeme.latent_variables.latex_report as latex_report_module

    def fake_tex_escape(text: str) -> str:
        return f'ESC[{text}]'

    def fake_tex_identifier(text: str) -> str:
        return f'ID[{text}]'

    monkeypatch.setattr(latex_report_module, 'tex_escape', fake_tex_escape)
    monkeypatch.setattr(latex_report_module, 'tex_identifier', fake_tex_identifier)
    return latex_report_module


def test_combo_to_latex_returns_zero_for_empty_combination(patched_tex) -> None:
    combo = _linear_combination(intercept=None, terms=[])

    result = _combo_to_latex(combo)

    assert result == '0'


def test_combo_to_latex_with_parameter_intercept_and_mixed_terms(
    patched_tex,
) -> None:
    combo = _linear_combination(
        intercept=_resolved_parameter('alpha'),
        terms=[
            _term(_resolved_parameter('beta_time'), 'time'),
            _term(_resolved_constant(2.5), 'cost'),
        ],
    )

    result = _combo_to_latex(combo)

    assert result == 'ID[alpha] + ID[beta_time]\\,ID[time] + 2.5\\,ID[cost]'


def test_combo_to_latex_with_constant_intercept_only(patched_tex) -> None:
    combo = _linear_combination(
        intercept=_resolved_constant(7),
        terms=[],
    )

    result = _combo_to_latex(combo)

    assert result == 'ESC[7]'


def test_generate_latex_report_covers_all_main_branches_and_content(
    patched_tex,
) -> None:
    resolved = _resolved_model(
        metadata=SimpleNamespace(
            n_latent_variables=2,
            n_indicators=3,
            n_threshold_systems=1,
        ),
        latent_variables={
            'LV_A': _latent_variable(
                intercept=_resolved_parameter('alpha_a'),
                terms=[
                    _term(_resolved_parameter('beta_a'), 'x_a'),
                    _term(_resolved_parameter('beta_b'), 'x_b'),
                ],
                sigma=_resolved_parameter('sigma_a'),
            ),
            'LV_EMPTY': _latent_variable(
                intercept=_resolved_parameter('alpha_empty'),
                terms=[],
                sigma=_resolved_parameter('sigma_empty'),
            ),
        },
        measurement_equations={
            'gauss<1>': _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=_resolved_parameter('alpha_g'),
                    terms=[
                        _term(_resolved_parameter('lambda_g'), 'LV_A'),
                        _term(_resolved_constant(3), 'z_g'),
                    ],
                ),
                sigma=_resolved_parameter('sigma_g'),
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
            'probit&2': _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=None,
                    terms=[],
                ),
                sigma=_resolved_parameter('sigma_p'),
                measurement_model=MeasurementModel.ORDERED_PROBIT,
            ),
            'logit"3"': _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=_resolved_constant(1),
                    terms=[],
                ),
                sigma=_resolved_parameter('sigma_l'),
                measurement_model=MeasurementModel.ORDERED_LOGIT,
            ),
        },
        threshold_systems={
            'type<&>': _threshold_system(
                [
                    _cutpoint('tau_1', 'a_b'),
                    _cutpoint('tau_2', 'c<d>'),
                ]
            )
        },
        normalization=SimpleNamespace(
            rules=[
                _normalization_rule('fixed because <reason>', 'beta_norm', 1),
                _normalization_rule('anchor & scale', 'sigma_norm', 0.5),
            ],
            warnings=['warn <one>', 'warn & two'],
        ),
        parameters={
            'zeta': _resolved_parameter(
                'zeta',
                lower_bound=-1,
                upper_bound=1,
                role='role_z',
                status='status_z',
                notes=['note z1', 'note z2'],
            ),
            'alpha': _resolved_parameter(
                'alpha',
                lower_bound=None,
                upper_bound=None,
                role='role_a',
                status='status_a',
                notes=[],
            ),
        },
    )

    report = generate_latex_report(resolved)

    assert report.endswith('\n')

    assert r'\section{Latent Variable Model}' in report
    assert r'\subsection{Model overview}' in report
    assert (
        'The model contains 2 latent variables, 3 indicators, and 1 ordinal threshold systems.'
        in report
    )

    assert r'\subsection{Structural equations}' in report
    assert (
        r'ID[LV_A] = ID[alpha_a] + ID[beta_a]\,ID[x_a] + ID[beta_b]\,ID[x_b] + ID[sigma_a]\,\omega_{ID[LV_A]}'
        in report
    )
    assert (
        r'ID[LV_EMPTY] = ID[alpha_empty] + ID[sigma_empty]\,\omega_{ID[LV_EMPTY]}'
        in report
    )

    assert r'\subsection{Measurement equations}' in report

    assert r'\paragraph{Indicator ESC[gauss<1>]}' in report
    assert (
        r'I^*_{ESC[gauss<1>]} = ID[alpha_g] + ID[lambda_g]\,ID[LV_A] + 3\,ID[z_g] + ID[sigma_g]\,\varepsilon_{ESC[gauss<1>]}'
        in report
    )
    assert r'I_{ESC[gauss<1>]} = I^*_{ESC[gauss<1>]}' in report

    assert r'\paragraph{Indicator ESC[probit&2]}' in report
    assert (
        r'P(I_{ESC[probit&2]}=j_m\mid x^*) = \Phi\!\left(\frac{\tau_m-0}{ID[sigma_p]}\right) - \Phi\!\left(\frac{\tau_{m-1}-0}{ID[sigma_p]}\right)'
        in report
    )

    assert '\\paragraph{Indicator ESC[logit"3"]}' in report
    assert (
        r'P(I_{ESC[logit"3"]}=j_m\mid x^*) = \Lambda\!\left(\frac{\tau_m-ESC[1]}{ID[sigma_l]}\right) - \Lambda\!\left(\frac{\tau_{m-1}-ESC[1]}{ID[sigma_l]}\right)'
        in report
    )

    assert r'\subsection{Threshold systems}' in report
    assert r'\paragraph{Threshold system ESC[type<&>]}' in report
    assert r'\begin{align*}' in report
    assert r'tau_1 &= ESC[a_b] \\' in report
    assert r'tau_2 &= ESC[c<d>] ' in report
    assert r'\end{align*}' in report

    assert r'\subsection{Normalization}' in report
    assert r'\item ESC[fixed because <reason>] (ESC[beta_norm] = 1)' in report
    assert r'\item ESC[anchor & scale] (ESC[sigma_norm] = 0.5)' in report
    assert r'\paragraph{Warnings}' in report
    assert r'\item ESC[warn <one>]' in report
    assert r'\item ESC[warn & two]' in report

    assert r'\subsection{Parameter table}' in report
    assert r'\begin{tabular}{lllll}' in report
    assert r'\hline' in report
    assert r'Name & Role & Status & Bounds & Notes \\' in report
    assert (
        r'ESC[alpha] & ESC[role_a] & ESC[status_a] & ESC[[-\infty, +\infty]] &  \\'
        in report
    )
    assert (
        r'ESC[zeta] & ESC[role_z] & ESC[status_z] & ESC[[-1, 1]] & ESC[note z1]; ESC[note z2] \\'
        in report
    )
    assert r'\end{tabular}' in report


def test_generate_latex_report_without_thresholds_rules_or_warnings(
    patched_tex,
) -> None:
    resolved = _resolved_model(
        metadata=SimpleNamespace(
            n_latent_variables=0,
            n_indicators=0,
            n_threshold_systems=0,
        ),
        latent_variables={},
        measurement_equations={},
        threshold_systems={},
        normalization=SimpleNamespace(
            rules=[],
            warnings=[],
        ),
        parameters={},
    )

    report = generate_latex_report(resolved)

    assert r'\subsection{Threshold systems}' not in report
    assert r'No explicit normalization plan was provided.' in report
    assert r'\paragraph{Warnings}' not in report
    assert r'\begin{tabular}{lllll}' in report
    assert r'\end{tabular}' in report


def test_generate_latex_report_raises_when_structural_sigma_is_missing(
    patched_tex,
) -> None:
    resolved = _resolved_model(
        metadata=SimpleNamespace(
            n_latent_variables=1,
            n_indicators=0,
            n_threshold_systems=0,
        ),
        latent_variables={
            'LV_BAD': _latent_variable(
                intercept=_resolved_parameter('alpha_bad'),
                terms=[],
                sigma=None,
            )
        },
        measurement_equations={},
        threshold_systems={},
        normalization=SimpleNamespace(rules=[], warnings=[]),
        parameters={},
    )

    with pytest.raises(
        ValueError,
        match="Structural equation for latent variable 'LV_BAD' is missing a resolved sigma parameter.",
    ):
        generate_latex_report(resolved)


def test_generate_latex_report_raises_when_measurement_sigma_is_missing(
    patched_tex,
) -> None:
    resolved = _resolved_model(
        metadata=SimpleNamespace(
            n_latent_variables=0,
            n_indicators=1,
            n_threshold_systems=0,
        ),
        latent_variables={},
        measurement_equations={
            'indicator_x': _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=None,
                    terms=[],
                ),
                sigma=None,
                measurement_model=MeasurementModel.GAUSSIAN,
            )
        },
        threshold_systems={},
        normalization=SimpleNamespace(rules=[], warnings=[]),
        parameters={},
    )

    with pytest.raises(
        ValueError,
        match="Measurement equation for indicator 'indicator_x' is missing a resolved sigma parameter.",
    ):
        generate_latex_report(resolved)


def test_save_latex_report_writes_utf8_to_string_path(tmp_path) -> None:
    report = 'électricité\n\\section{Test}\n'
    path = tmp_path / 'report.tex'

    save_latex_report(report, str(path))

    assert path.read_text(encoding='utf-8') == report


def test_save_latex_report_writes_utf8_to_path_object(tmp_path) -> None:
    report = 'Résumé\n\\subsection{Accentué}\n'
    path = tmp_path / 'nested_report.tex'

    save_latex_report(report, Path(path))

    assert path.read_text(encoding='utf-8') == report
