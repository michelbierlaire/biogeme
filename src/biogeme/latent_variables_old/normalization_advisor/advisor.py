"""
Normalization advisor for latent-variable specifications.

This module takes the structural description produced by
``analyzer.analyze_model_structure`` and derives normalization advice.

It does not create a ``NormalizationPlan``. It only answers:

- what must be normalized,
- which indicators are reasonable anchors,
- which concrete fixings are recommended.

The advisor distinguishes two layers of normalization:

- latent-variable normalization, which applies to all measurement models,
- threshold-system normalization, which applies only to ordinal measurement
  models (ordered probit and ordered logit).

Michel Bierlaire
Fri Mar 06 2026, 10:57:25
"""

from __future__ import annotations

from dataclasses import dataclass

from .analyzer import ModelStructure

NORMALIZATION_DISCLAIMER = (
    "The normalization recommendations provided by this advisor are based on "
    "general identification principles and heuristic rules. They are intended "
    "to assist the user in constructing a consistent normalization plan, but "
    "they are provided without guarantee. Every latent-variable model is "
    "different, and the general rules implemented here may not be appropriate "
    "in all situations. Users should therefore review the recommendations "
    "carefully and apply them with caution."
)


@dataclass
class LatentVariableNormalizationAdvice:
    """Normalization advice for one latent variable.

    :param latent_variable: Name of the latent variable.
    :param recommended_indicator: Suggested reference indicator, if any.
    :param location_normalization: Recommended location normalization.
    :param scale_normalization: Recommended scale normalization.
    :param warning: Optional warning message.
    """

    latent_variable: str
    recommended_indicator: str | None
    location_normalization: str
    scale_normalization: str
    warning: str | None = None


@dataclass
class ThresholdSystemNormalizationAdvice:
    """Normalization advice for one ordinal threshold system.

    :param type_name: Name of the threshold system.
    :param symmetric: Whether the threshold system is symmetric.
    :param location_normalization: Recommended location normalization.
    :param scale_normalization: Recommended scale normalization.
    :param reference_indicator: Suggested indicator for sigma normalization.
    :param warning: Optional warning message.
    """

    type_name: str
    symmetric: bool
    location_normalization: str
    scale_normalization: str
    reference_indicator: str | None
    warning: str | None = None


@dataclass
class SuggestedFixing:
    """Suggested concrete fixing.

    :param parameter: Human-readable representation of the parameter to fix.
    :param value: Suggested value.
    :param reason: Explanation of the fixing.
    """

    parameter: str
    value: float
    reason: str


@dataclass
class NormalizationAdvice:
    """Complete normalization advice for a model.

    :param latent_variable_advice: Advice for each latent variable.
    :param threshold_system_advice: Advice for each threshold system.
    :param suggested_fixings: Suggested concrete fixings.
    :param warnings: Global warnings.
    """

    latent_variable_advice: dict[str, LatentVariableNormalizationAdvice]
    threshold_system_advice: dict[str, ThresholdSystemNormalizationAdvice]
    suggested_fixings: list[SuggestedFixing]
    warnings: list[str]
    disclaimer: str


def _choose_reference_indicator(
    indicators: list[str], unique_indicators: list[str]
) -> str | None:
    """Choose a reference indicator deterministically.

    Preference order:
    1. alphabetical among unique indicators,
    2. alphabetical among all indicators.

    :param indicators: All indicators attached to the latent variable.
    :param unique_indicators: Indicators attached only to this latent variable.
    :return: Suggested reference indicator, or None if no indicator exists.
    """
    if unique_indicators:
        return sorted(unique_indicators)[0]
    if indicators:
        return sorted(indicators)[0]
    return None


def _choose_threshold_reference_indicator(indicators: list[str]) -> str | None:
    """Choose the indicator used for sigma normalization in a threshold system.

    :param indicators: Indicators using the threshold system.
    :return: Suggested reference indicator, or None if no indicator exists.
    """
    if not indicators:
        return None
    return sorted(indicators)[0]


def _advise_latent_variables(
    structure: ModelStructure,
) -> tuple[
    dict[str, LatentVariableNormalizationAdvice], list[SuggestedFixing], list[str]
]:
    """Generate advice for latent variables.

    :param structure: Structural model description.
    :return: Tuple with:
        - advice by latent variable,
        - suggested fixings,
        - warnings.
    """
    advice: dict[str, LatentVariableNormalizationAdvice] = {}
    suggested_fixings: list[SuggestedFixing] = []
    warnings: list[str] = []

    for lv_name, lv_info in structure.latent_variables.items():
        reference_indicator = _choose_reference_indicator(
            indicators=lv_info.indicators,
            unique_indicators=lv_info.unique_indicators,
        )

        warning: str | None = None

        if not lv_info.indicators:
            warning = (
                f"Latent variable '{lv_name}' has no indicator. "
                "A reference-indicator normalization cannot be suggested."
            )
            warnings.append(warning)
        elif not lv_info.unique_indicators:
            warning = (
                f"Latent variable '{lv_name}' has no unique indicator. "
                f"The suggested reference indicator '{reference_indicator}' is shared."
            )
            warnings.append(warning)

        advice[lv_name] = LatentVariableNormalizationAdvice(
            latent_variable=lv_name,
            recommended_indicator=reference_indicator,
            location_normalization=(
                "Fix one measurement intercept to 0 "
                "(reference-indicator location normalization)."
            ),
            scale_normalization=(
                "Fix one loading to +1 or -1 "
                "(reference-indicator scale normalization)."
            ),
            warning=warning,
        )

        if reference_indicator is not None:
            suggested_fixings.append(
                SuggestedFixing(
                    parameter=f'MeasurementIntercept("{reference_indicator}")',
                    value=0.0,
                    reason=(
                        f"Location normalization for latent variable '{lv_name}' "
                        f"using reference indicator '{reference_indicator}'."
                    ),
                )
            )
            suggested_fixings.append(
                SuggestedFixing(
                    parameter=(
                        f'MeasurementLoading("{lv_name}", "{reference_indicator}")'
                    ),
                    value=1.0,
                    reason=(
                        f"Scale normalization for latent variable '{lv_name}' "
                        f"using reference indicator '{reference_indicator}'."
                    ),
                )
            )

    return advice, suggested_fixings, warnings


def _advise_threshold_systems(
    structure: ModelStructure,
) -> tuple[
    dict[str, ThresholdSystemNormalizationAdvice],
    list[SuggestedFixing],
    list[str],
]:
    """Generate advice for threshold systems.

    :param structure: Structural model description.
    :return: Tuple with:
        - advice by threshold system,
        - suggested fixings,
        - warnings.
    """
    advice: dict[str, ThresholdSystemNormalizationAdvice] = {}
    suggested_fixings: list[SuggestedFixing] = []
    warnings: list[str] = []

    for type_name, threshold_info in structure.threshold_systems.items():
        reference_indicator = _choose_threshold_reference_indicator(
            threshold_info.indicators
        )

        warning: str | None = None
        if not threshold_info.indicators:
            warning = (
                f"Threshold system '{type_name}' is not used by any indicator. "
                "No normalization can be suggested."
            )
            warnings.append(warning)
        elif len(threshold_info.indicators) == 1:
            warning = (
                f"Threshold system '{type_name}' is used by only one indicator "
                f"('{reference_indicator}')."
            )
            warnings.append(warning)

        if threshold_info.symmetric:
            location_normalization = (
                "Location normalization is handled by symmetry of the threshold system."
            )
        else:
            location_normalization = (
                "Fix the first threshold to 0 "
                "(threshold-location normalization for a non-symmetric system)."
            )

        scale_normalization = (
            "Fix one measurement error scale to 1 "
            "(ordinal-measurement scale normalization)."
        )

        advice[type_name] = ThresholdSystemNormalizationAdvice(
            type_name=type_name,
            symmetric=threshold_info.symmetric,
            location_normalization=location_normalization,
            scale_normalization=scale_normalization,
            reference_indicator=reference_indicator,
            warning=warning,
        )

        if not threshold_info.symmetric:
            suggested_fixings.append(
                SuggestedFixing(
                    parameter=f'ThresholdFirst("{type_name}")',
                    value=0.0,
                    reason=(
                        f"Location normalization for non-symmetric threshold system "
                        f"'{type_name}'."
                    ),
                )
            )

        if reference_indicator is not None:
            suggested_fixings.append(
                SuggestedFixing(
                    parameter=f'MeasurementSigma("{reference_indicator}")',
                    value=1.0,
                    reason=(
                        f"Scale normalization for ordinal threshold system '{type_name}' "
                        f"using indicator '{reference_indicator}'."
                    ),
                )
            )

    return advice, suggested_fixings, warnings


def advise_normalization(structure: ModelStructure) -> NormalizationAdvice:
    """Generate normalization advice from a model structure.

    :param structure: Structural description of the model.
    :return: Normalization advice.
    """
    (
        latent_variable_advice,
        lv_fixings,
        lv_warnings,
    ) = _advise_latent_variables(structure)

    (
        threshold_system_advice,
        threshold_fixings,
        threshold_warnings,
    ) = _advise_threshold_systems(structure)

    global_warnings = lv_warnings + threshold_warnings
    if structure.gaussian_indicator_names:
        global_warnings.append(
            "The model contains Gaussian indicators. Threshold-system normalization advice "
            "therefore applies only to the ordinal part of the model."
        )

    return NormalizationAdvice(
        latent_variable_advice=latent_variable_advice,
        threshold_system_advice=threshold_system_advice,
        suggested_fixings=lv_fixings + threshold_fixings,
        warnings=global_warnings,
        disclaimer=NORMALIZATION_DISCLAIMER,
    )
