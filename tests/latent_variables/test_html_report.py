from __future__ import annotations

from types import SimpleNamespace

import pytest
from biogeme.latent_variables.html_report import (
    _combo_to_math_text,
    generate_html_report,
    save_html_report,
)
from biogeme.latent_variables.model_spec import MeasurementModel


def _resolved_constant(value: object) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _resolved_parameter(
    final_name: str,
    *,
    lower_bound: object = None,
    upper_bound: object = None,
    role: str = "generic_role",
    status: str = "generic_status",
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
    terms: list[SimpleNamespace],
    sigma: object | None,
) -> SimpleNamespace:
    return SimpleNamespace(terms=terms, sigma=sigma)


def _latent_variable(
    *,
    terms: list[SimpleNamespace],
    sigma: object | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        structural_equation=_structural_equation(terms=terms, sigma=sigma)
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
    import biogeme.latent_variables.html_report as html_report_module

    def fake_tex_escape(text: str) -> str:
        return f"ESC[{text}]"

    def fake_tex_identifier(text: str) -> str:
        return f"ID[{text}]"

    monkeypatch.setattr(html_report_module, "tex_escape", fake_tex_escape)
    monkeypatch.setattr(html_report_module, "tex_identifier", fake_tex_identifier)
    return html_report_module


def test_combo_to_math_text_returns_zero_for_empty_combination(patched_tex) -> None:
    combo = _linear_combination(intercept=None, terms=[])

    result = _combo_to_math_text(combo)

    assert result == "0"


def test_combo_to_math_text_with_parameter_intercept_and_mixed_terms(
    patched_tex,
) -> None:
    combo = _linear_combination(
        intercept=_resolved_parameter("alpha"),
        terms=[
            _term(_resolved_parameter("beta_time"), "time"),
            _term(_resolved_constant(2.5), "cost"),
        ],
    )

    result = _combo_to_math_text(combo)

    assert result == ("ID[alpha] + ID[beta_time]\\,ID[time] + ESC[2.5]\\,ID[cost]")


def test_combo_to_math_text_with_constant_intercept_only(patched_tex) -> None:
    combo = _linear_combination(
        intercept=_resolved_constant(7),
        terms=[],
    )

    result = _combo_to_math_text(combo)

    assert result == "ESC[7]"


def test_generate_html_report_covers_all_branches_and_escaping(patched_tex) -> None:
    resolved = _resolved_model(
        metadata=SimpleNamespace(
            n_latent_variables=2,
            n_indicators=3,
            n_threshold_systems=1,
        ),
        latent_variables={
            "LV_A": _latent_variable(
                terms=[
                    _term(_resolved_parameter("beta_a"), "x_a"),
                    _term(_resolved_parameter("beta_b"), "x_b"),
                ],
                sigma=_resolved_parameter("sigma_a"),
            ),
            "LV_EMPTY": _latent_variable(
                terms=[],
                sigma=None,
            ),
        },
        measurement_equations={
            "gauss<1>": _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=_resolved_parameter("alpha_g"),
                    terms=[
                        _term(_resolved_parameter("lambda_g"), "LV_A"),
                        _term(_resolved_constant(3), "z_g"),
                    ],
                ),
                sigma=_resolved_parameter("sigma_g"),
                measurement_model=MeasurementModel.GAUSSIAN,
            ),
            "probit&2": _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=None,
                    terms=[],
                ),
                sigma=_resolved_parameter("sigma_p"),
                measurement_model=MeasurementModel.ORDERED_PROBIT,
            ),
            'logit"3"': _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=_resolved_constant(1),
                    terms=[],
                ),
                sigma=_resolved_parameter("sigma_l"),
                measurement_model=MeasurementModel.ORDERED_LOGIT,
            ),
        },
        threshold_systems={
            "type<&>": _threshold_system(
                [
                    _cutpoint("tau<1>", "a_b"),
                    _cutpoint("tau&2", "c<d>"),
                ]
            )
        },
        normalization=SimpleNamespace(
            rules=[
                _normalization_rule("fixed because <reason>", "beta_norm", 1),
                _normalization_rule("anchor & scale", "sigma_norm", 0.5),
            ],
            warnings=["warn <one>", "warn & two"],
        ),
        parameters={
            "zeta": _resolved_parameter(
                "zeta",
                lower_bound=-1,
                upper_bound=1,
                role="role_z",
                status="status_z",
                notes=["note z1", "note z2"],
            ),
            "alpha": _resolved_parameter(
                "alpha",
                lower_bound=None,
                upper_bound=None,
                role="role_a",
                status="status_a",
                notes=[],
            ),
        },
    )

    html = generate_html_report(resolved)

    assert html.startswith("<!DOCTYPE html>")
    assert html.endswith("</body></html>")

    assert '<meta charset="utf-8">' in html
    assert (
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
        in html
    )
    assert "<title>Latent Variable Model Report</title>" in html
    assert "<h1>Latent Variable Model Report</h1>" in html

    assert (
        "<p>The model contains <strong>2</strong> latent variables, "
        "<strong>3</strong> indicators, and <strong>1</strong> ordinal threshold systems.</p>"
        in html
    )

    assert "<h2>Structural equations</h2>" in html
    assert (
        "<p>\\[ID[LV_A] = ID[beta_a]\\,ID[x_a] + ID[beta_b]\\,ID[x_b] + "
        "ID[sigma_a]\\,\\omega_{ESC[LV_A]}\\]</p>" in html
    )
    assert "<p>\\[ID[LV_EMPTY] = 0 + 0\\,\\omega_{ESC[LV_EMPTY]}\\]</p>" in html

    assert "<h2>Measurement equations</h2>" in html

    assert "<h3>gauss&lt;1&gt;</h3>" in html
    assert (
        "<p>\\[I^*_{ESC[gauss<1>]} = ID[alpha_g] + ID[lambda_g]\\,ID[LV_A] + "
        "ESC[3]\\,ID[z_g] + ID[sigma_g]\\,\\varepsilon_{ESC[gauss<1>]}\\]</p>" in html
    )
    assert "<p>\\[I_{ESC[gauss<1>]} = I^*_{ESC[gauss<1>]}\\]</p>" in html

    assert "<h3>probit&amp;2</h3>" in html
    assert (
        "<p>\\[I^*_{ESC[probit&2]} = 0 + ID[sigma_p]\\,\\varepsilon_{ESC[probit&2]}\\]</p>"
        in html
    )
    assert "<p>Ordered probit measurement model.</p>" in html

    assert "<h3>logit&quot;3&quot;</h3>" in html
    assert (
        "<p>\\[I^*_{ESC[logit\"3\"]} = ESC[1] + ID[sigma_l]\\,\\varepsilon_{ESC[logit\"3\"]}\\]</p>"
        in html
    )
    assert "<p>Ordered logit measurement model.</p>" in html

    assert "<h2>Threshold systems</h2>" in html
    assert "<h3>type&lt;&amp;&gt;</h3>" in html
    assert "<ul>" in html
    assert "<li><code>tau&lt;1&gt;</code> = <code>ESC[a_b]</code></li>" in html
    assert "<li><code>tau&amp;2</code> = <code>ESC[c&lt;d&gt;]</code></li>" in html

    assert "<h2>Normalization</h2>" in html
    assert (
        "<li>fixed because &lt;reason&gt; (<code>beta_norm</code> = <code>1</code>)</li>"
        in html
    )
    assert (
        "<li>anchor &amp; scale (<code>sigma_norm</code> = <code>0.5</code>)</li>"
        in html
    )
    assert "<h3>Warnings</h3><ul>" in html
    assert "<li>warn &lt;one&gt;</li>" in html
    assert "<li>warn &amp; two</li>" in html

    assert "<h2>Parameters</h2>" in html
    assert (
        "<table><tr><th>Name</th><th>Role</th><th>Status</th><th>Bounds</th><th>Notes</th></tr>"
        in html
    )
    assert (
        "<tr><td><code>alpha</code></td><td>role_a</td><td>status_a</td>"
        "<td>[None, None]</td><td></td></tr>" in html
    )
    assert (
        "<tr><td><code>zeta</code></td><td>role_z</td><td>status_z</td>"
        "<td>[-1, 1]</td><td>note z1; note z2</td></tr>" in html
    )


def test_generate_html_report_without_thresholds_or_normalization_plan(
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

    html = generate_html_report(resolved)

    assert "<h2>Threshold systems</h2>" not in html
    assert "<p>No explicit normalization plan was provided.</p>" in html
    assert "<h3>Warnings</h3>" not in html
    assert (
        "<table><tr><th>Name</th><th>Role</th><th>Status</th><th>Bounds</th><th>Notes</th></tr></table>"
        in html
    )


def test_generate_html_report_raises_when_measurement_sigma_is_missing(
    patched_tex,
) -> None:
    resolved = _resolved_model(
        metadata=SimpleNamespace(
            n_latent_variables=0,
            n_indicators=1,
            n_threshold_systems=0,
        ),
        measurement_equations={
            "indicator_x": _measurement_equation(
                systematic_part=_linear_combination(
                    intercept=None,
                    terms=[],
                ),
                sigma=None,
                measurement_model=MeasurementModel.GAUSSIAN,
            )
        },
        normalization=SimpleNamespace(rules=[], warnings=[]),
    )

    with pytest.raises(
        ValueError,
        match="Measurement equation for indicator 'indicator_x' is missing a resolved sigma parameter.",
    ):
        generate_html_report(resolved)


def test_save_html_report_writes_utf8_file(tmp_path) -> None:
    report = "<html><body>électricité</body></html>"
    path = tmp_path / "report.html"

    save_html_report(report, path)

    assert path.read_text(encoding="utf-8") == report
