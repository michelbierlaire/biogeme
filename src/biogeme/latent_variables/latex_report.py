"""Generate a full scientific LaTeX report from a resolved model."""

from __future__ import annotations

from pathlib import Path

from .model_spec import MeasurementModel
from .resolved import ResolvedModel
from .tex_utils import tex_escape, tex_identifier


def _combo_to_latex(combo) -> str:
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
            else str(term.coefficient.value)
        )
        parts.append(f'{coef}\\,{tex_identifier(term.variable_name)}')
    return ' + '.join(parts) if parts else '0'


def generate_latex_report(resolved: ResolvedModel) -> str:
    lines: list[str] = []
    lines.append(r'\section{Latent Variable Model}')
    lines.append(r'\subsection{Model overview}')
    lines.append(
        rf'The model contains {resolved.metadata.n_latent_variables} latent variables, {resolved.metadata.n_indicators} indicators, and {resolved.metadata.n_threshold_systems} ordinal threshold systems.'
    )
    lines.append(r'\subsection{Structural equations}')
    for latent_name, latent in resolved.latent_variables.items():
        latent_name_tex = tex_identifier(latent_name)
        eq = latent.structural_equation
        if eq.sigma is None:
            raise ValueError(
                f"Structural equation for latent variable '{latent_name}' is missing a resolved sigma parameter."
            )
        sigma = tex_identifier(eq.sigma.final_name)
        rhs = _combo_to_latex(eq.systematic_part)
        lines.append(r'\[')
        lines.append(
            rf'{latent_name_tex} = {rhs} + {sigma}\,\omega_{{{latent_name_tex}}}'
        )
        lines.append(r'\]')
    lines.append(r'\subsection{Measurement equations}')
    for indicator_name, equation in resolved.measurement_equations.items():
        indicator_name_tex = tex_escape(indicator_name)
        mu = _combo_to_latex(equation.systematic_part)
        if equation.sigma is None:
            raise ValueError(
                f"Measurement equation for indicator '{indicator_name}' is missing a resolved sigma parameter."
            )
        sigma = tex_identifier(equation.sigma.final_name)
        lines.append(rf'\paragraph{{Indicator {indicator_name_tex}}}')
        lines.append(r'\[')
        lines.append(
            rf'I^*_{{{indicator_name_tex}}} = {mu} + {sigma}\,\varepsilon_{{{indicator_name_tex}}}'
        )
        lines.append(r'\]')
        if equation.measurement_model == MeasurementModel.GAUSSIAN:
            lines.append(r'\[')
            lines.append(rf'I_{{{indicator_name_tex}}} = I^*_{{{indicator_name_tex}}}')
            lines.append(r'\]')
        elif equation.measurement_model == MeasurementModel.ORDERED_PROBIT:
            lines.append(r'\[')
            lines.append(
                rf'P(I_{{{indicator_name_tex}}}=j_m\mid x^*) = \Phi\!\left(\frac{{\tau_m-{mu}}}{{{sigma}}}\right) - \Phi\!\left(\frac{{\tau_{{m-1}}-{mu}}}{{{sigma}}}\right)'
            )
            lines.append(r'\]')
        else:
            lines.append(r'\[')
            lines.append(
                rf'P(I_{{{indicator_name_tex}}}=j_m\mid x^*) = \Lambda\!\left(\frac{{\tau_m-{mu}}}{{{sigma}}}\right) - \Lambda\!\left(\frac{{\tau_{{m-1}}-{mu}}}{{{sigma}}}\right)'
            )
            lines.append(r'\]')
    if resolved.threshold_systems:
        lines.append(r'\subsection{Threshold systems}')
        for type_name, system in resolved.threshold_systems.items():
            lines.append(rf'\paragraph{{Threshold system {tex_escape(type_name)}}}')
            lines.append(r'\begin{align*}')
            for i, cutpoint in enumerate(system.cutpoints):
                expr = tex_escape(cutpoint.expression_text)
                end = r'\\' if i < len(system.cutpoints) - 1 else ''
                lines.append(rf'{cutpoint.symbol_name} &= {expr} {end}')
            lines.append(r'\end{align*}')
    lines.append(r'\subsection{Normalization}')
    if resolved.normalization.rules:
        lines.append(r'\begin{itemize}')
        for rule in resolved.normalization.rules:
            lines.append(
                rf'\item {tex_escape(rule.reason)} ({tex_escape(rule.target_name)} = {rule.value})'
            )
        lines.append(r'\end{itemize}')
    else:
        lines.append(r'No explicit normalization plan was provided.')
    if resolved.normalization.warnings:
        lines.append(r'\paragraph{Warnings}')
        lines.append(r'\begin{itemize}')
        for warning in resolved.normalization.warnings:
            lines.append(rf'\item {tex_escape(warning)}')
        lines.append(r'\end{itemize}')
    lines.append(r'\subsection{Parameter table}')
    lines.append(r'\begin{tabular}{lllll}')
    lines.append(r'\hline')
    lines.append(r'Name & Role & Status & Bounds & Notes \\')
    lines.append(r'\hline')
    for name in sorted(resolved.parameters):
        param = resolved.parameters[name]
        lower = '-\\infty' if param.lower_bound is None else str(param.lower_bound)
        upper = '+\\infty' if param.upper_bound is None else str(param.upper_bound)
        bounds = f'[{lower}, {upper}]'
        notes = '; '.join(tex_escape(n) for n in param.notes)
        lines.append(
            rf'{tex_escape(name)} & {tex_escape(param.role.value)} & {tex_escape(param.status.value)} & {tex_escape(bounds)} & {notes} \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


def save_latex_report(report: str, path: str | Path) -> None:
    """Save LaTeX report to a file."""
    Path(path).write_text(report, encoding='utf-8')
