"""Build live Biogeme expressions from a resolved model."""

from __future__ import annotations

from dataclasses import dataclass

from biogeme.expressions import MultipleProduct, MultipleSum, log

from .context import EstimationMode
from .model_spec import MeasurementModel
from .resolved import (
    ParameterCreationKind,
    ParameterRole,
    ParameterStatus,
    ResolvedConstant,
    ResolvedLinearCombination,
    ResolvedModel,
    ResolvedParameter,
)


@dataclass(frozen=True, slots=True)
class BuiltBiogemeModel:
    """Live Biogeme expressions corresponding to the latent-variable model."""

    parameters: dict[str, object]
    estimated_parameter_names: dict[str, str]
    parameter_groups: dict[str, list[str]]
    latent_expressions: dict[str, object]
    threshold_expressions: dict[str, list[object]]
    measurement_terms: dict[str, object]
    conditional_likelihood: object | None
    conditional_log_likelihood: object
    integrated_likelihood: object | None


def _biogeme_parameter_name(param: ResolvedParameter) -> str | None:
    """Return the name of the Biogeme parameter created for a resolved parameter.

    Numeric constants are not estimated and therefore have no Biogeme parameter
    name. Fixed Betas are represented by Biogeme parameters, but they are not
    estimated; they are excluded from the returned name because the report groups
    are intended for estimated parameters.

    :param param: resolved parameter.
    :return: name of the estimated Biogeme parameter, or None if the parameter is
        not estimated.
    """
    if param.status != ParameterStatus.FREE:
        return None
    if param.creation_kind == ParameterCreationKind.NUMERIC_CONSTANT:
        return None
    if param.creation_kind == ParameterCreationKind.LOG_EXP_BETA:
        return f'{param.final_name}_log'
    return param.final_name


def _beta_or_numeric(param: ResolvedParameter):
    from biogeme.expressions import Beta, Numeric, exp

    if param.creation_kind == ParameterCreationKind.NUMERIC_CONSTANT:
        return Numeric(float(param.fixed_value))
    if param.creation_kind == ParameterCreationKind.LOG_EXP_BETA:
        parameter_name = _biogeme_parameter_name(param)
        if parameter_name is None:
            raise ValueError(
                f'No estimated Biogeme parameter name is available for {param.final_name}.'
            )
        return exp(Beta(parameter_name, param.initial_value, None, None, 0))
    if param.creation_kind == ParameterCreationKind.BOUNDED_BETA:
        return Beta(
            param.final_name,
            param.initial_value,
            param.lower_bound,
            param.upper_bound,
            0,
        )
    if param.creation_kind == ParameterCreationKind.FREE_BETA:
        return Beta(
            param.final_name,
            param.initial_value,
            param.lower_bound,
            param.upper_bound,
            0,
        )
    if param.creation_kind == ParameterCreationKind.FIXED_BETA:
        return Beta(
            param.final_name,
            param.initial_value,
            param.lower_bound,
            param.upper_bound,
            1,
        )
    raise ValueError(f'Unsupported parameter creation kind: {param.creation_kind}')


def _parameter_groups(
    resolved: ResolvedModel,
    estimated_parameter_names: dict[str, str],
) -> dict[str, list[str]]:
    """Build report-ready groups of estimated parameters.

    The structural and threshold groups are derived from semantic parameter
    roles. Measurement parameters are grouped by resolved measurement equation.
    The names are the actual names of the Biogeme parameters that are estimated.
    This avoids duplicating naming conventions outside the builder.
    """

    def names_for_roles(roles: set[ParameterRole]) -> list[str]:
        return [
            estimated_parameter_names[parameter.final_name]
            for parameter in resolved.parameters.values()
            if parameter.status == ParameterStatus.FREE
            and parameter.role in roles
            and parameter.final_name in estimated_parameter_names
        ]

    candidate_groups = {
        'Structural equation': names_for_roles(
            {
                ParameterRole.STRUCTURAL_INTERCEPT,
                ParameterRole.STRUCTURAL_COEFFICIENT,
                ParameterRole.STRUCTURAL_SIGMA,
            }
        ),
    }

    for indicator_name, equation in resolved.measurement_equations.items():
        resolved_parameter_names: list[str] = []

        intercept = equation.systematic_part.intercept
        if intercept is not None and not isinstance(intercept, ResolvedConstant):
            resolved_parameter_names.append(intercept.final_name)

        for term in equation.systematic_part.terms:
            coefficient = term.coefficient
            if not isinstance(coefficient, ResolvedConstant):
                resolved_parameter_names.append(coefficient.final_name)

        if equation.sigma is not None:
            resolved_parameter_names.append(equation.sigma.final_name)

        candidate_groups[f'Measurement equation: {indicator_name}'] = [
            estimated_parameter_names[parameter_name]
            for parameter_name in resolved_parameter_names
            if parameter_name in estimated_parameter_names
        ]

    candidate_groups['Thresholds'] = names_for_roles(
        {
            ParameterRole.THRESHOLD_FIRST,
            ParameterRole.THRESHOLD_DELTA,
        }
    )
    return {
        group_name: parameter_names
        for group_name, parameter_names in candidate_groups.items()
        if parameter_names
    }


def _render_linear_combination(
    combo: ResolvedLinearCombination,
    parameters: dict[str, object],
    symbols: dict[str, object] | None = None,
):
    from biogeme.expressions import Numeric, Variable

    expr = Numeric(0.0)
    local_symbols = {} if symbols is None else symbols

    if combo.intercept is not None:
        if isinstance(combo.intercept, ResolvedConstant):
            expr = expr + Numeric(combo.intercept.value)
        else:
            expr = expr + parameters[combo.intercept.final_name]

    for term in combo.terms:
        coefficient = term.coefficient
        variable_expr = local_symbols.get(
            term.variable_name, Variable(term.variable_name)
        )
        if isinstance(coefficient, ResolvedConstant):
            expr = expr + Numeric(coefficient.value) * variable_expr
        else:
            expr = expr + parameters[coefficient.final_name] * variable_expr
    return expr


def _neutral_labels_for_equation(resolved: ResolvedModel, equation) -> list[int]:
    """Return neutral labels from the equation's shared indicator type."""
    indicator_types = getattr(resolved, 'indicator_types', {})
    if not indicator_types:
        return []
    indicator_type = indicator_types.get(equation.type_name)
    return [] if indicator_type is None else indicator_type.neutral_labels


def _mask_neutral_labels(
    expression,
    observed,
    neutral_labels: list[int],
    *,
    neutral_value: float,
):
    """Replace an indicator contribution when its response is neutral."""
    from biogeme.expressions import Elem

    if not neutral_labels:
        return expression
    is_neutral = observed == neutral_labels[0]
    for label in neutral_labels[1:]:
        is_neutral = is_neutral | (observed == label)
    return Elem({0: expression, 1: neutral_value}, is_neutral)


def _build_measurement_terms_ml(
    resolved: ResolvedModel,
    parameters: dict[str, object],
    latent_expressions: dict[str, object],
    threshold_expressions: dict[str, list[object]],
) -> dict[str, object]:
    from biogeme.distributions import normalpdf
    from biogeme.expressions import OrderedLogit, OrderedProbit, Variable

    measurement_terms: dict[str, object] = {}
    for indicator_name, equation in resolved.measurement_equations.items():
        mu = _render_linear_combination(
            equation.systematic_part,
            parameters,
            symbols=latent_expressions,
        )
        y = Variable(equation.observed_variable_name)
        if equation.measurement_model == MeasurementModel.GAUSSIAN:
            if equation.sigma is None:
                raise ValueError(
                    f"Gaussian indicator '{indicator_name}' requires a resolved sigma parameter."
                )
            sigma = parameters[equation.sigma.final_name]
            gaussian_term = normalpdf((y - mu) / sigma) / sigma
            measurement_terms[indicator_name] = _mask_neutral_labels(
                gaussian_term,
                y,
                _neutral_labels_for_equation(resolved, equation),
                neutral_value=1.0,
            )
        else:
            if equation.sigma is None:
                raise ValueError(
                    f"Indicator '{indicator_name}' with measurement model "
                    f"'{equation.measurement_model}' requires a resolved sigma parameter."
                )
            sigma = parameters[equation.sigma.final_name]
            cutpoints = threshold_expressions[equation.threshold_system_name]
            system = resolved.threshold_systems[equation.threshold_system_name]
            cls = (
                OrderedProbit
                if equation.measurement_model == MeasurementModel.ORDERED_PROBIT
                else OrderedLogit
            )
            measurement_terms[indicator_name] = cls(
                eta=mu / sigma,
                cutpoints=[c / sigma for c in cutpoints],
                y=y,
                categories=system.categories,
                neutral_labels=system.neutral_labels,
            )
    return measurement_terms


def _build_measurement_log_terms_bayesian(
    resolved: ResolvedModel,
    parameters: dict[str, object],
    latent_expressions: dict[str, object],
    threshold_expressions: dict[str, list[object]],
) -> dict[str, object]:
    from biogeme.distributions import normal_logpdf
    from biogeme.expressions import OrderedLogLogit, OrderedLogProbit, Variable

    measurement_log_terms: dict[str, object] = {}
    for indicator_name, equation in resolved.measurement_equations.items():
        mu = _render_linear_combination(
            equation.systematic_part,
            parameters,
            symbols=latent_expressions,
        )
        y = Variable(equation.observed_variable_name)
        if equation.sigma is None:
            raise ValueError(
                f"Indicator '{indicator_name}' with measurement model "
                f"'{equation.measurement_model}' requires a resolved sigma parameter."
            )
        sigma = parameters[equation.sigma.final_name]
        if equation.measurement_model == MeasurementModel.GAUSSIAN:
            gaussian_log_term = normal_logpdf(y, mu, sigma)
            measurement_log_terms[indicator_name] = _mask_neutral_labels(
                gaussian_log_term,
                y,
                _neutral_labels_for_equation(resolved, equation),
                neutral_value=0.0,
            )
        else:
            cutpoints = threshold_expressions[equation.threshold_system_name]
            system = resolved.threshold_systems[equation.threshold_system_name]
            cls = (
                OrderedLogProbit
                if equation.measurement_model == MeasurementModel.ORDERED_PROBIT
                else OrderedLogLogit
            )
            measurement_log_terms[indicator_name] = cls(
                eta=mu / sigma,
                cutpoints=[c / sigma for c in cutpoints],
                y=y,
                categories=system.categories,
                neutral_labels=system.neutral_labels,
            )
    return measurement_log_terms


def _build_biogeme_model_ml(
    *,
    parameters: dict[str, object],
    estimated_parameter_names: dict[str, str],
    parameter_groups: dict[str, list[str]],
    latent_expressions: dict[str, object],
    threshold_expressions: dict[str, list[object]],
    measurement_terms: dict[str, object],
) -> BuiltBiogemeModel:
    from biogeme.expressions import MonteCarlo

    conditional_likelihood = MultipleProduct(list(measurement_terms.values()))
    conditional_log_likelihood = MultipleSum(
        [log(term) for term in measurement_terms.values()]
    )
    integrated_likelihood = MonteCarlo(conditional_likelihood)
    return BuiltBiogemeModel(
        parameters=parameters,
        estimated_parameter_names=estimated_parameter_names,
        parameter_groups=parameter_groups,
        latent_expressions=latent_expressions,
        threshold_expressions=threshold_expressions,
        measurement_terms=measurement_terms,
        conditional_likelihood=conditional_likelihood,
        conditional_log_likelihood=conditional_log_likelihood,
        integrated_likelihood=integrated_likelihood,
    )


def _build_biogeme_model_bayesian(
    *,
    parameters: dict[str, object],
    estimated_parameter_names: dict[str, str],
    parameter_groups: dict[str, list[str]],
    latent_expressions: dict[str, object],
    threshold_expressions: dict[str, list[object]],
    measurement_log_terms: dict[str, object],
) -> BuiltBiogemeModel:
    conditional_log_likelihood = MultipleSum(list(measurement_log_terms.values()))
    return BuiltBiogemeModel(
        parameters=parameters,
        estimated_parameter_names=estimated_parameter_names,
        parameter_groups=parameter_groups,
        latent_expressions=latent_expressions,
        threshold_expressions=threshold_expressions,
        measurement_terms=measurement_log_terms,
        conditional_likelihood=None,
        conditional_log_likelihood=conditional_log_likelihood,
        integrated_likelihood=None,
    )


def build_biogeme_model(resolved: ResolvedModel) -> BuiltBiogemeModel:
    from biogeme.expressions import DistributedParameter, Draws

    parameters = {
        name: _beta_or_numeric(param) for name, param in resolved.parameters.items()
    }
    estimated_parameter_names = {
        name: biogeme_name
        for name, param in resolved.parameters.items()
        if (biogeme_name := _biogeme_parameter_name(param)) is not None
    }
    parameter_groups = _parameter_groups(
        resolved=resolved,
        estimated_parameter_names=estimated_parameter_names,
    )

    latent_expressions: dict[str, object] = {}
    is_bayesian = resolved.metadata.estimation_mode == EstimationMode.BAYESIAN
    for latent_name, latent in resolved.latent_variables.items():
        eq = latent.structural_equation
        deterministic = _render_linear_combination(eq.systematic_part, parameters)
        sigma = parameters[eq.sigma.final_name] if eq.sigma is not None else 0.0
        draw = Draws(eq.draw_name, draw_type=eq.draw_type)
        stochastic_expression = deterministic + sigma * draw
        latent_expressions[latent_name] = (
            DistributedParameter(latent_name, stochastic_expression)
            if is_bayesian
            else stochastic_expression
        )

    threshold_expressions: dict[str, list[object]] = {}
    for type_name, system in resolved.threshold_systems.items():
        rendered: list[object] = []
        env: dict[str, object] = {name: parameters[name] for name in parameters}
        for cutpoint in system.cutpoints:
            rendered_expr = eval(cutpoint.expression_text, {}, env)
            env[cutpoint.symbol_name] = rendered_expr
            rendered.append(rendered_expr)
        threshold_expressions[type_name] = rendered

    if is_bayesian:
        measurement_log_terms = _build_measurement_log_terms_bayesian(
            resolved,
            parameters,
            latent_expressions,
            threshold_expressions,
        )
        return _build_biogeme_model_bayesian(
            parameters=parameters,
            estimated_parameter_names=estimated_parameter_names,
            parameter_groups=parameter_groups,
            latent_expressions=latent_expressions,
            threshold_expressions=threshold_expressions,
            measurement_log_terms=measurement_log_terms,
        )

    measurement_terms = _build_measurement_terms_ml(
        resolved,
        parameters,
        latent_expressions,
        threshold_expressions,
    )
    return _build_biogeme_model_ml(
        parameters=parameters,
        estimated_parameter_names=estimated_parameter_names,
        parameter_groups=parameter_groups,
        latent_expressions=latent_expressions,
        threshold_expressions=threshold_expressions,
        measurement_terms=measurement_terms,
    )
