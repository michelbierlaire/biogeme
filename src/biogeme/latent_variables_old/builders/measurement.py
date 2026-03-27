"""Shared measurement-equation builders and dispatcher.

This module builds the common latent-indicator equation used by all
measurement models:

.. math::

    I^* = \mu + \sigma \varepsilon,

where ``mu`` is the systematic part of the measurement equation and
``sigma`` is the measurement scale parameter.

The concrete likelihood contribution depends on the measurement model used by
an indicator:

- ``MeasurementModel.GAUSSIAN`` uses the latent-indicator equation directly
  as a continuous Gaussian measurement model.
- ``MeasurementModel.ORDERED_PROBIT`` uses the same latent-indicator equation
  together with thresholds in an ordered-probit likelihood.
- ``MeasurementModel.ORDERED_LOGIT`` uses the same latent-indicator equation
  together with thresholds in an ordered-logit likelihood.

This module contains the shared measurement-equation logic. It first
constructs the latent-indicator equation and then dispatches to
specialized likelihood builders for the supported measurement models.

Michel Bierlaire
Thu Mar 05 2026, 17:30:40
"""

from __future__ import annotations

from collections.abc import Iterable

from biogeme.expressions import Expression, Variable
from biogeme.latent_variables.builders.latent_indicator import LatentIndicatorEquation

from .context import BuildContext
from .measurement_gaussian import build_gaussian_term
from .measurement_ordinal import build_ordered_logit_term, build_ordered_probit_term
from .utils import PreparedSpecs, resolve_fixed_or_beta, resolve_fixed_or_positive
from ..likert_indicators import MeasurementModel
from ..normalization.parameter_refs import (
    MeasurementIntercept,
    MeasurementLoading,
    MeasurementSigma,
)
from ..normalization.plan import NormalizationPlan


def resolve_measurement_intercept(
    *,
    indicator_name: str,
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> Expression:
    """Resolve measurement intercept (fixed via plan, else free Beta)."""
    return resolve_fixed_or_beta(
        target=MeasurementIntercept(indicator_name),
        context_name=context.naming.measurement_intercept_name(indicator_name),
        init_value=3.14,
        plan=plan,
    )


def resolve_measurement_loading(
    *,
    latent_name: str,
    indicator_name: str,
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> Expression:
    """Resolve measurement loading (fixed via plan, else free Beta)."""
    return resolve_fixed_or_beta(
        target=MeasurementLoading(latent_name, indicator_name),
        context_name=context.naming.measurement_loading_name(
            latent_name, indicator_name
        ),
        init_value=0.1,
        plan=plan,
    )


def resolve_measurement_sigma(
    *,
    indicator_name: str,
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> Expression:
    """Resolve measurement sigma_star (fixed via plan, else free via sigma_factory)."""
    prefix = context.naming.measurement_sigma_prefix(indicator_name)
    free_sigma = context.sigma_factory(prefix=prefix)

    return resolve_fixed_or_positive(
        target=MeasurementSigma(indicator_name),
        plan=plan,
        free_expression=free_sigma,
        require_positive=True,
        positive_name_for_error=f"Measurement sigma for indicator '{indicator_name}'",
    )


def build_mu_for_indicator(
    *,
    indicator_name: str,
    latent_variables: Iterable,
    latent_expressions: dict[str, Expression],
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> Expression:
    """Build the systematic part ``mu`` for one indicator."""
    mu = resolve_measurement_intercept(
        indicator_name=indicator_name, context=context, plan=plan
    )

    for lv in latent_variables:
        if indicator_name not in set(lv.indicators):
            continue
        loading = resolve_measurement_loading(
            latent_name=lv.name,
            indicator_name=indicator_name,
            context=context,
            plan=plan,
        )
        mu = mu + loading * latent_expressions[lv.name]

    return mu


def build_latent_indicator_equation(
    *,
    indicator_name: str,
    prepared: PreparedSpecs,
    latent_expressions: dict[str, Expression],
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> LatentIndicatorEquation:
    """Build the common latent-indicator equation for one indicator.

    :param indicator_name: Name of the indicator.
    :param prepared: Prepared and validated specification objects.
    :param latent_expressions: Built latent-variable expressions.
    :param context: Build context.
    :param plan: Optional normalization plan.
    :return: Common latent-indicator equation.
    """
    mu = build_mu_for_indicator(
        indicator_name=indicator_name,
        latent_variables=prepared.latent_variables,
        latent_expressions=latent_expressions,
        context=context,
        plan=plan,
    )
    sigma = resolve_measurement_sigma(
        indicator_name=indicator_name,
        context=context,
        plan=plan,
    )
    y = Variable(indicator_name)
    return LatentIndicatorEquation(
        indicator_name=indicator_name,
        y=y,
        mu=mu,
        sigma=sigma,
    )


def build_measurement_terms(
    *,
    prepared: PreparedSpecs,
    latent_expressions: dict[str, Expression],
    thresholds_by_type: dict[str, list[Expression]],
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> dict[str, Expression]:
    """Build measurement-likelihood terms for all referenced indicators.

    The dispatcher first builds the common latent-indicator equation and then
    applies the likelihood corresponding to the indicator's measurement model.

    :param prepared: Prepared and validated specification objects.
    :param latent_expressions: Built latent-variable expressions.
    :param thresholds_by_type: Thresholds indexed by indicator type.
    :param context: Build context.
    :param plan: Optional normalization plan.
    :return: Likelihood contribution indexed by indicator name.
    """
    out: dict[str, Expression] = {}

    for indicator_name in sorted(prepared.referenced_indicator_names):
        ind = prepared.indicator_by_name[indicator_name]
        lt = prepared.type_by_name[ind.type_name]
        equation = build_latent_indicator_equation(
            indicator_name=indicator_name,
            prepared=prepared,
            latent_expressions=latent_expressions,
            context=context,
            plan=plan,
        )

        if ind.measurement_model == MeasurementModel.GAUSSIAN:
            out[indicator_name] = build_gaussian_term(equation=equation)
            continue

        cutpoints = thresholds_by_type[lt.type_name]

        if ind.measurement_model == MeasurementModel.ORDERED_PROBIT:
            out[indicator_name] = build_ordered_probit_term(
                equation=equation,
                categories=lt.categories,
                neutral_labels=lt.neutral_labels,
                cutpoints=cutpoints,
                context=context,
            )
            continue

        if ind.measurement_model == MeasurementModel.ORDERED_LOGIT:
            out[indicator_name] = build_ordered_logit_term(
                equation=equation,
                categories=lt.categories,
                neutral_labels=lt.neutral_labels,
                cutpoints=cutpoints,
                context=context,
            )
            continue

        raise ValueError(
            f"Unknown measurement model '{ind.measurement_model}' for indicator '{indicator_name}'."
        )

    return out
