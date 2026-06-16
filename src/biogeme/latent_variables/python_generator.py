from __future__ import annotations

"""Generate runnable pedagogical Python code for the latent-variable part."""

from pathlib import Path

from .context import EstimationMode
from .model_spec import MeasurementModel
from .resolved import ParameterCreationKind, ResolvedLinearCombination, ResolvedModel


def _emit_parameter_assignment(name: str, param) -> list[str]:
    lines: list[str] = []
    if param.creation_kind == ParameterCreationKind.NUMERIC_CONSTANT:
        lines.append(f'{name} = {param.fixed_value}')
    elif param.creation_kind == ParameterCreationKind.LOG_EXP_BETA:
        lines.append(
            f'{name}_log = Beta("{name}_log", {param.initial_value}, None, None, 0)'
        )
        lines.append(f'{name} = exp({name}_log)')
    elif param.creation_kind == ParameterCreationKind.BOUNDED_BETA:
        lines.append(
            f'{name} = Beta("{name}", {param.initial_value}, {param.lower_bound}, {param.upper_bound}, 0)'
        )
    elif param.creation_kind == ParameterCreationKind.FREE_BETA:
        lines.append(
            f'{name} = Beta("{name}", {param.initial_value}, {param.lower_bound}, {param.upper_bound}, 0)'
        )
    else:
        lines.append(f'{name} = {param.fixed_value}')
    return lines


def _term_to_python(term, symbol_names: set[str] | None = None) -> str:
    coefficient = term.coefficient
    if symbol_names is not None and term.variable_name in symbol_names:
        variable_expr = term.variable_name
    else:
        variable_expr = f'Variable("{term.variable_name}")'
    if hasattr(coefficient, 'final_name'):
        return f'{coefficient.final_name} * {variable_expr}'
    return f'{coefficient.value} * {variable_expr}'


def _combo_to_python(
    combo: ResolvedLinearCombination,
    symbol_names: set[str] | None = None,
) -> str:
    pieces: list[str] = []
    if combo.intercept is not None:
        if hasattr(combo.intercept, 'final_name'):
            pieces.append(combo.intercept.final_name)
        else:
            pieces.append(str(combo.intercept.value))
    pieces.extend(_term_to_python(term, symbol_names) for term in combo.terms)
    return ' + '.join(pieces) if pieces else 'Numeric(0.0)'


# ------------------------ Mode-specific helpers and generators ------------------------


def _emit_header(lines: list[str], resolved: ResolvedModel, *, bayesian: bool) -> None:
    lines.append(
        '"""Pedagogical runnable Biogeme code for the latent-variable part of the model."""'
    )
    lines.append('')
    if bayesian:
        lines.append(
            'from biogeme.expressions import Beta, DistributedParameter, Draws, MultipleSum, Numeric, OrderedLogLogit, OrderedLogProbit, Variable, exp'
        )
        lines.append('from biogeme.distributions import normal_logpdf')
    else:
        lines.append(
            'from biogeme.expressions import Beta, Draws, MonteCarlo, MultipleProduct, MultipleSum, Numeric, OrderedLogit, OrderedProbit, Variable, exp, log'
        )
        lines.append('from biogeme.distributions import normalpdf')
    lines.append('')


def _emit_parameters(lines: list[str], resolved: ResolvedModel) -> None:
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Parameters')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    for name in sorted(resolved.parameters):
        lines.extend(_emit_parameter_assignment(name, resolved.parameters[name]))
    lines.append('')


def _emit_threshold_systems(lines: list[str], resolved: ResolvedModel) -> None:
    if not resolved.threshold_systems:
        return
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Threshold systems')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    for type_name, system in resolved.threshold_systems.items():
        lines.append(f'# Threshold system: {type_name}')
        for cutpoint in system.cutpoints:
            symbol = f'{type_name}_{cutpoint.symbol_name}'
            expr = cutpoint.expression_text
            for source in sorted(
                cutpoint.source_parameter_names, key=len, reverse=True
            ):
                expr = expr.replace(
                    source,
                    f'{type_name}_{source}' if source.startswith('tau_') else source,
                )
            lines.append(f'{symbol} = {expr}')
        lines.append('')


def _emit_measurement_terms_ml(lines: list[str], resolved: ResolvedModel) -> None:
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Measurement equations and likelihood terms')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    latent_symbol_names = set(resolved.latent_variables)
    for indicator_name, equation in resolved.measurement_equations.items():
        lines.append(f'# Indicator: {indicator_name}')
        lines.append(
            f'mu_{indicator_name} = {_combo_to_python(equation.systematic_part, latent_symbol_names)}'
        )
        lines.append(
            f'y_{indicator_name} = Variable("{equation.observed_variable_name}")'
        )
        if equation.sigma is None:
            raise ValueError(
                f"Measurement equation for indicator '{indicator_name}' is missing a resolved sigma parameter."
            )
        sigma_name = equation.sigma.final_name
        if equation.measurement_model == MeasurementModel.GAUSSIAN:
            lines.append(
                f'term_{indicator_name} = normalpdf((y_{indicator_name} - mu_{indicator_name}) / {sigma_name}) / {sigma_name}'
            )
        else:
            system = resolved.threshold_systems[equation.threshold_system_name]
            cutpoints = ', '.join(
                f'{equation.threshold_system_name}_{cp.symbol_name} / {sigma_name}'
                for cp in system.cutpoints
            )
            cls = (
                'OrderedProbit'
                if equation.measurement_model == MeasurementModel.ORDERED_PROBIT
                else 'OrderedLogit'
            )
            lines.append(
                f'term_{indicator_name} = {cls}(eta=mu_{indicator_name} / {sigma_name}, cutpoints=[{cutpoints}], y=y_{indicator_name}, categories={system.categories}, neutral_labels={system.neutral_labels})'
            )
        lines.append('')


def _emit_measurement_log_terms_bayesian(
    lines: list[str], resolved: ResolvedModel
) -> None:
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Measurement equations and log-likelihood terms')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    latent_symbol_names = set(resolved.latent_variables)
    for indicator_name, equation in resolved.measurement_equations.items():
        lines.append(f'# Indicator: {indicator_name}')
        lines.append(
            f'mu_{indicator_name} = {_combo_to_python(equation.systematic_part, latent_symbol_names)}'
        )
        lines.append(
            f'y_{indicator_name} = Variable("{equation.observed_variable_name}")'
        )
        if equation.sigma is None:
            raise ValueError(
                f"Measurement equation for indicator '{indicator_name}' is missing a resolved sigma parameter."
            )
        sigma_name = equation.sigma.final_name
        if equation.measurement_model == MeasurementModel.GAUSSIAN:
            lines.append(
                f'log_term_{indicator_name} = normal_logpdf(y_{indicator_name}, mu_{indicator_name}, {sigma_name})'
            )
        else:
            system = resolved.threshold_systems[equation.threshold_system_name]
            cutpoints = ', '.join(
                f'{equation.threshold_system_name}_{cp.symbol_name} / {sigma_name}'
                for cp in system.cutpoints
            )
            cls = (
                'OrderedLogProbit'
                if equation.measurement_model == MeasurementModel.ORDERED_PROBIT
                else 'OrderedLogLogit'
            )
            lines.append(
                f'log_term_{indicator_name} = {cls}(eta=mu_{indicator_name} / {sigma_name}, cutpoints=[{cutpoints}], y=y_{indicator_name}, categories={system.categories}, neutral_labels={system.neutral_labels})'
            )
        lines.append('')


def _generate_python_code_ml(resolved: ResolvedModel) -> str:
    lines: list[str] = []
    _emit_header(lines, resolved, bayesian=False)
    _emit_parameters(lines, resolved)
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Structural equations')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    for latent_name, latent in resolved.latent_variables.items():
        eq = latent.structural_equation
        deterministic = _combo_to_python(eq.systematic_part)
        lines.append(f'mu_{latent_name} = {deterministic}')
        lines.append(
            f'draw_{latent_name} = Draws("{eq.draw_name}", draw_type="{eq.draw_type}")'
        )
        if eq.sigma is not None:
            lines.append(
                f'{latent_name} = mu_{latent_name} + {eq.sigma.final_name} * draw_{latent_name}'
            )
        else:
            lines.append(f'{latent_name} = mu_{latent_name}')
        lines.append('')
    _emit_threshold_systems(lines, resolved)
    _emit_measurement_terms_ml(lines, resolved)
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Conditional indicator likelihood and Monte Carlo integration')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    if resolved.measurement_equations:
        product_terms = ', '.join(
            f'term_{name}' for name in sorted(resolved.measurement_equations)
        )
        lines.append(
            f'conditional_measurement_likelihood = MultipleProduct([{product_terms}])'
        )
        lines.append(
            'conditional_log_likelihood = MultipleSum([log(term) for term in ['
            + product_terms
            + '] ])'
        )
        lines.append(
            'integrated_measurement_likelihood = MonteCarlo(conditional_measurement_likelihood)'
        )
    else:
        lines.append('conditional_measurement_likelihood = Numeric(1.0)')
        lines.append('conditional_log_likelihood = Numeric(0.0)')
        lines.append('integrated_measurement_likelihood = Numeric(1.0)')
    return '\n'.join(lines) + '\n'


def _generate_python_code_bayesian(resolved: ResolvedModel) -> str:
    lines: list[str] = []
    _emit_header(lines, resolved, bayesian=True)
    _emit_parameters(lines, resolved)
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Structural equations')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    for latent_name, latent in resolved.latent_variables.items():
        eq = latent.structural_equation
        deterministic = _combo_to_python(eq.systematic_part)
        lines.append(f'mu_{latent_name} = {deterministic}')
        lines.append(
            f'draw_{latent_name} = Draws("{eq.draw_name}", draw_type="{eq.draw_type}")'
        )
        if eq.sigma is not None:
            lines.append(
                f'stochastic_{latent_name} = mu_{latent_name} + {eq.sigma.final_name} * draw_{latent_name}'
            )
            lines.append(
                f'{latent_name} = DistributedParameter("{latent_name}", stochastic_{latent_name})'
            )
        else:
            lines.append(
                f'{latent_name} = DistributedParameter("{latent_name}", mu_{latent_name})'
            )
        lines.append('')
    _emit_threshold_systems(lines, resolved)
    _emit_measurement_log_terms_bayesian(lines, resolved)
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    lines.append('# Conditional log-likelihood')
    lines.append(
        '# ---------------------------------------------------------------------------'
    )
    if resolved.measurement_equations:
        log_terms = ', '.join(
            f'log_term_{name}' for name in sorted(resolved.measurement_equations)
        )
        lines.append(f'conditional_log_likelihood = MultipleSum([{log_terms}])')
    else:
        lines.append('conditional_log_likelihood = Numeric(0.0)')
    return '\n'.join(lines) + '\n'


def generate_python_code(resolved: ResolvedModel) -> str:
    if resolved.metadata.estimation_mode == EstimationMode.BAYESIAN:
        return _generate_python_code_bayesian(resolved)
    return _generate_python_code_ml(resolved)


def save_python_code(code: str, path: str | Path) -> None:
    """Save generated Python code to a file."""
    Path(path).write_text(code, encoding='utf-8')
