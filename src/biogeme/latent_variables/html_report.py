from __future__ import annotations

"""Generate a static HTML report from a resolved model."""

from html import escape
from pathlib import Path

from .model_spec import MeasurementModel
from .resolved import ResolvedModel
from .tex_utils import tex_escape, tex_identifier


# Helper for TeX-safe rendering of linear combinations for MathJax
def _combo_to_math_text(combo) -> str:
    """Render a linear combination as TeX-safe math text for MathJax."""
    parts: list[str] = []
    if combo.intercept is not None:
        if hasattr(combo.intercept, 'final_name'):
            parts.append(tex_identifier(combo.intercept.final_name))
        else:
            parts.append(tex_escape(str(combo.intercept.value)))
    for term in combo.terms:
        coef = (
            tex_identifier(term.coefficient.final_name)
            if hasattr(term.coefficient, 'final_name')
            else tex_escape(str(term.coefficient.value))
        )
        parts.append(f'{coef}\\,{tex_identifier(term.variable_name)}')
    return ' + '.join(parts) if parts else '0'


def generate_html_report(resolved: ResolvedModel) -> str:
    parts: list[str] = []
    parts.append('<!DOCTYPE html>')
    parts.append('<html><head><meta charset="utf-8">')
    parts.append(
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
    )
    parts.append(
        '<style>body{font-family:Arial,sans-serif;max-width:1000px;margin:2rem auto;line-height:1.5} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ccc;padding:0.4rem} code,pre{background:#f6f6f6;padding:0.2rem 0.4rem}</style>'
    )
    parts.append('<title>Latent Variable Model Report</title></head><body>')
    parts.append('<h1>Latent Variable Model Report</h1>')
    parts.append(
        f'<p>The model contains <strong>{resolved.metadata.n_latent_variables}</strong> latent variables, <strong>{resolved.metadata.n_indicators}</strong> indicators, and <strong>{resolved.metadata.n_threshold_systems}</strong> ordinal threshold systems.</p>'
    )
    parts.append('<h2>Structural equations</h2>')
    for latent_name, latent in resolved.latent_variables.items():
        eq = latent.structural_equation
        rhs = _combo_to_math_text(eq.systematic_part)
        sigma = tex_identifier(eq.sigma.final_name) if eq.sigma is not None else '0'
        parts.append(
            f'<p>\\[{tex_identifier(latent_name)} = {rhs} + {sigma}\\,\\omega_{{{tex_escape(latent_name)}}}\\]</p>'
        )
    parts.append('<h2>Measurement equations</h2>')
    for indicator_name, equation in resolved.measurement_equations.items():
        mu = _combo_to_math_text(equation.systematic_part)
        if equation.sigma is None:
            raise ValueError(
                f"Measurement equation for indicator '{indicator_name}' is missing a resolved sigma parameter."
            )
        sigma = tex_identifier(equation.sigma.final_name)
        parts.append(f'<h3>{escape(indicator_name)}</h3>')
        parts.append(
            f'<p>\\[I^*_{{{tex_escape(indicator_name)}}} = {mu} + {sigma}\\,\\varepsilon_{{{tex_escape(indicator_name)}}}\\]</p>'
        )
        if equation.measurement_model == MeasurementModel.GAUSSIAN:
            parts.append(
                f'<p>\\[I_{{{tex_escape(indicator_name)}}} = I^*_{{{tex_escape(indicator_name)}}}\\]</p>'
            )
        elif equation.measurement_model == MeasurementModel.ORDERED_PROBIT:
            parts.append('<p>Ordered probit measurement model.</p>')
        else:
            parts.append('<p>Ordered logit measurement model.</p>')
    if resolved.threshold_systems:
        parts.append('<h2>Threshold systems</h2>')
        for type_name, system in resolved.threshold_systems.items():
            parts.append(f'<h3>{escape(type_name)}</h3>')
            parts.append('<ul>')
            for cutpoint in system.cutpoints:
                parts.append(
                    f'<li><code>{escape(cutpoint.symbol_name)}</code> = <code>{escape(tex_escape(cutpoint.expression_text))}</code></li>'
                )
            parts.append('</ul>')
    parts.append('<h2>Normalization</h2>')
    if resolved.normalization.rules:
        parts.append('<ul>')
        for rule in resolved.normalization.rules:
            parts.append(
                f'<li>{escape(rule.reason)} (<code>{escape(rule.target_name)}</code> = <code>{escape(str(rule.value))}</code>)</li>'
            )
        parts.append('</ul>')
    else:
        parts.append('<p>No explicit normalization plan was provided.</p>')
    if resolved.normalization.warnings:
        parts.append('<h3>Warnings</h3><ul>')
        for warning in resolved.normalization.warnings:
            parts.append(f'<li>{escape(warning)}</li>')
        parts.append('</ul>')
    parts.append('<h2>Parameters</h2>')
    parts.append(
        '<table><tr><th>Name</th><th>Role</th><th>Status</th><th>Bounds</th><th>Notes</th></tr>'
    )
    for name in sorted(resolved.parameters):
        param = resolved.parameters[name]
        bounds = f'[{param.lower_bound}, {param.upper_bound}]'
        notes = '; '.join(param.notes)
        parts.append(
            f'<tr><td><code>{escape(name)}</code></td><td>{escape(param.role.value)}</td><td>{escape(param.status.value)}</td><td>{escape(bounds)}</td><td>{escape(notes)}</td></tr>'
        )
    parts.append('</table>')
    parts.append('</body></html>')
    return ''.join(parts)


def save_html_report(report: str, path: str | Path) -> None:
    """Save HTML report to a file."""
    Path(path).write_text(report, encoding='utf-8')
