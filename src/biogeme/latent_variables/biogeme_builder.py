"""Build live Biogeme expressions from a resolved model."""

from __future__ import annotations

from dataclasses import dataclass

from biogeme.expressions import MultipleProduct, MultipleSum, log

from .context import EstimationMode
from .model_spec import MeasurementModel
from .resolved import (
    ParameterCreationKind,
    ResolvedConstant,
    ResolvedLinearCombination,
    ResolvedModel,
    ResolvedParameter,
)


@dataclass(frozen=True, slots=True)
class BuiltBiogemeModel:
    """Live Biogeme expressions corresponding to the latent-variable model."""

    parameters: dict[str, object]
    latent_expressions: dict[str, object]
    threshold_expressions: dict[str, list[object]]
    measurement_terms: dict[str, object]
    conditional_likelihood: object | None
    conditional_log_likelihood: object
    integrated_likelihood: object | None


def _beta_or_numeric(param: ResolvedParameter):
    from biogeme.expressions import Beta, Numeric, exp

    if param.creation_kind == ParameterCreationKind.NUMERIC_CONSTANT:
        return Numeric(float(param.fixed_value))
    if param.creation_kind == ParameterCreationKind.LOG_EXP_BETA:
        return exp(Beta(f"{param.final_name}_log", param.initial_value, None, None, 0))
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
    raise ValueError(f"Unsupported parameter creation kind: {param.creation_kind}")


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


def _build_measurement_terms_ml(
    resolved: ResolvedModel,
    parameters: dict[str, object],
    latent_expressions: dict[str, object],
    threshold_expressions: dict[str, list[object]],
) -> dict[str, object]:
    from biogeme.expressions import OrderedLogit, OrderedProbit, Variable
    from biogeme.distributions import normalpdf

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
            measurement_terms[indicator_name] = normalpdf((y - mu) / sigma) / sigma
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
    from biogeme.expressions import OrderedLogLogit, OrderedLogProbit, Variable
    from biogeme.distributions import normal_logpdf

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
            measurement_log_terms[indicator_name] = normal_logpdf(y, mu, sigma)
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
    latent_expressions: dict[str, object],
    threshold_expressions: dict[str, list[object]],
    measurement_log_terms: dict[str, object],
) -> BuiltBiogemeModel:
    conditional_log_likelihood = MultipleSum(list(measurement_log_terms.values()))
    return BuiltBiogemeModel(
        parameters=parameters,
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

    latent_expressions: dict[str, object] = {}
    is_bayesian = resolved.metadata.estimation_mode == EstimationMode.BAYESIAN
    for latent_name, latent in resolved.latent_variables.items():
        eq = latent.structural_equation
        deterministic = _render_linear_combination(
            ResolvedLinearCombination(None, eq.terms), parameters
        )
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
        latent_expressions=latent_expressions,
        threshold_expressions=threshold_expressions,
        measurement_terms=measurement_terms,
    )
