"""JAX-compatible ordered-probit measurement equations for Biogeme latent-variable models.

This module constructs the joint ordered-probit measurement model typically used
in hybrid choice models. It combines:

- latent-variable definitions (structural equations, normalization rules),
- ordinal indicators (intercepts, latent-variable loadings, measurement scales),
- Likert-scale definitions (categories and ordered thresholds),
- consistency checks across model components,
- JAX-friendly Biogeme :class:`~biogeme.expressions.Expression` objects.

Thresholds (cut-points) are obtained from the :class:`~biogeme.latent_variables.likert_indicators.LikertType`
associated with each indicator via :meth:`~biogeme.latent_variables.likert_indicators.LikertType.get_thresholds`.
Threshold sharing relies on parameter naming: if multiple indicators (or types)
produce cut-points with identical parameter names, Biogeme will estimate a
single shared set of parameters.

The main entry points are:

- :func:`measurement_equations_jax`: product of measurement likelihoods.
- :func:`log_measurement_equations_jax`: sum of log measurement likelihoods.

Michel Bierlaire
Thu Dec 11 2025, 16:26:07
"""

from __future__ import annotations

from biogeme.expressions import (
    Expression,
    MultipleProduct,
    MultipleSum,
    OrderedProbit,
    Variable,
    log,
)
from biogeme.tools import assert_sets_equal

from .latent_variables import LatentVariable
from .likert_indicators import LikertIndicator, LikertType


def _ordered_model(
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
    draw_type: str,
) -> dict[str, Expression]:
    """Build ordered-probit likelihood terms for all indicators.

    For each ordinal indicator, this function constructs an
    :class:`~biogeme.expressions.OrderedProbit` term using:

    - the indicator intercept,
    - the indicator-specific measurement scale (except for normalization anchors),
    - the latent-variable structural equations (with the provided ``draw_type``),
    - the ordered cut-points returned by the corresponding
      :class:`~biogeme.latent_variables.likert_indicators.LikertType`.

    The function enforces a measurement-scale normalization: for each latent
    variable, the measurement scale (``sigma_star``) of its normalization
    indicator is fixed to 1.0.

    :param latent_variables:
        Latent-variable objects defining structural equations, the list of
        associated indicators, and the normalization anchor.
    :param likert_indicators:
        Indicator objects providing intercepts, measurement scale parameters, and
        latent-variable loading parameters.
    :param likert_types:
        Likert-type objects defining categories, neutral labels, and the
        threshold parameterization used to build cut-points.
    :param draw_type:
        Draw type assigned to each latent variable for JAX-based computations.
    :raises ValueError:
        If an indicator references an unknown ``type`` (no matching entry in
        ``likert_types``).
    :return:
        Dictionary mapping each indicator name to its ordered-probit likelihood
        expression.
    """

    # Set of indicator names appearing in any latent variable.
    all_indicators = {ind for lv in latent_variables for ind in lv.indicators}

    # Map of indicator objects by name.
    likert_mapping = {likert.name: likert for likert in likert_indicators}

    # Map of Likert types by type label.
    likert_types_mapping = {t.type: t for t in likert_types}

    # Check consistency between indicators declared by LVs and registered indicators.
    assert_sets_equal(
        name_a="Indicators",
        set_a=all_indicators,
        name_b="Likert indicators",
        set_b=set(likert_mapping.keys()),
    )

    # Intercepts: one per indicator.
    intercepts: dict[str, float | Expression] = {
        k: likert_mapping[k].intercept_parameter for k in all_indicators
    }

    # Coefficients linking latent variables to indicators.
    coefficients: dict[tuple[str, str], float | Expression] = {
        (lv.name, indicator): (
            lv.normalization.coefficient
            if indicator == lv.normalization.indicator
            else likert_mapping[indicator].get_lv_coefficient_parameter(
                latent_variable_name=lv.name
            )
        )
        for lv in latent_variables
        for indicator in lv.indicators
    }

    # Scale parameters of measurement error terms.
    sigma_star: dict[str, float | Expression] = {
        indicator: likert_mapping[indicator].scale_parameter
        for indicator in all_indicators
    }

    # Normalization: for each latent variable, force the intercept to 0.
    for lv in latent_variables:
        intercepts[lv.normalization.indicator] = 0.0

    # Normalization: for each group of thresholds, normalize one sigma_star to 1.
    # This is due to the ordered probit.
    for lt in likert_types:
        sigma_star[lt.scale_normalization] = 1.0

    # Linear predictors per indicator.
    models: dict[str, float | Expression] = {
        indicator: intercepts[indicator] for indicator in all_indicators
    }
    for lv in latent_variables:
        lv.draw_type_jax = draw_type
        for indicator in lv.indicators:
            models[indicator] += (
                coefficients[(lv.name, indicator)] * lv.structural_equation_jax
            )

    # Ordered-probit likelihood terms.
    ordered_ll: dict[str, Expression] = {}
    for indicator, model in models.items():
        the_likert = likert_mapping[indicator]
        the_likert_type = likert_types_mapping.get(the_likert.type, None)
        if the_likert_type is None:
            error_msg = (
                f"Unknown type for indicator {the_likert.name}: {the_likert.type}. "
                f"Known types: {list(likert_types_mapping.keys())}"
            )
            raise ValueError(error_msg)

        cutpoints = the_likert_type.get_thresholds()

        ordered_ll[indicator] = OrderedProbit(
            eta=model / sigma_star[indicator],
            cutpoints=[t / sigma_star[indicator] for t in cutpoints],
            y=Variable(indicator),
            categories=the_likert_type.categories,
            neutral_labels=the_likert_type.neutral_labels,
            enforce_order=True,
            eps=1e-12,
        )

    return ordered_ll


def measurement_equations_jax(
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
    draw_type: str,
) -> Expression:
    """Return the product of ordered-probit measurement likelihood terms.

    :param latent_variables:
        Latent-variable objects defining structural equations and associated
        indicators.
    :param likert_indicators:
        Likert indicator objects providing measurement parameters.
    :param likert_types:
        Likert-type objects defining categories and thresholds.
    :param draw_type:
        Draw type assigned to latent variables for JAX.
    :return:
        Expression representing the product of ordered-probit likelihood terms.
    """
    ordered_ll = _ordered_model(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        draw_type=draw_type,
    )
    return MultipleProduct(ordered_ll)


def log_measurement_equations_jax(
    latent_variables: list[LatentVariable],
    likert_indicators: list[LikertIndicator],
    likert_types: list[LikertType],
    draw_type: str,
) -> Expression:
    """Return the sum of log ordered-probit measurement likelihood terms.

    :param latent_variables:
        Latent-variable objects defining structural equations and associated
        indicators.
    :param likert_indicators:
        Likert indicator objects providing measurement parameters.
    :param likert_types:
        Likert-type objects defining categories and thresholds.
    :param draw_type:
        Draw type assigned to latent variables for JAX.
    :return:
        Expression representing the sum of log ordered-probit likelihood terms.
    """
    ordered_ll = _ordered_model(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
        draw_type=draw_type,
    )
    log_ordered_ll = {
        indicator: log(likelihood) for indicator, likelihood in ordered_ll.items()
    }
    return MultipleSum(log_ordered_ll)
