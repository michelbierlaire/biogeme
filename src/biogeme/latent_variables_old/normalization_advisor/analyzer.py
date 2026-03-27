"""
Model-structure analyzer for latent-variable specifications.

This module inspects the *specification objects* describing a hybrid choice
model and extracts structural information required for normalization advice.

It performs **no normalization itself** and contains **no Biogeme expressions**.
Its role is purely diagnostic: identify latent variables, indicators, and
threshold systems and how they are connected.

The resulting ``ModelStructure`` object is used by the normalization advisor.

Michel Bierlaire
Fri Mar 06 2026, 13:41:06
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from biogeme.latent_variables.latent_variables import LatentVariable
from biogeme.latent_variables.likert_indicators import (
    LikertIndicator,
    LikertType,
    MeasurementModel,
)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class LatentVariableInfo:
    """Structural information about a latent variable.

    :param name: Name of the latent variable.
    :param explanatory_variables: Variables entering the structural equation.
    :param indicators: Indicators linked to this latent variable.
    :param unique_indicators: Indicators linked only to this latent variable.
    :param shared_indicators: Indicators shared with other latent variables.
    """

    name: str
    explanatory_variables: list[str]
    indicators: list[str]
    unique_indicators: list[str]
    shared_indicators: list[str]


@dataclass
class IndicatorInfo:
    """Structural information about a measurement indicator.

    :param name: Indicator name.
    :param type_name: Likert-type name associated with the indicator.
    :param measurement_model: Measurement model used for this indicator.
    :param latent_variables: Latent variables influencing this indicator.
    """

    name: str
    type_name: str
    measurement_model: MeasurementModel
    latent_variables: list[str]


@dataclass
class ThresholdSystemInfo:
    """Information about a threshold system (e.g., Likert scale).

    :param type_name: Name of the threshold system.
    :param symmetric: Whether thresholds are symmetric around zero.
    :param categories: Ordered category labels.
    :param neutral_labels: Labels corresponding to neutral responses.
    :param indicators: Indicators using this threshold system.
    """

    type_name: str
    symmetric: bool
    categories: list[int]
    neutral_labels: list[int]
    indicators: list[str]


@dataclass
class ModelStructure:
    """Complete structural description of the latent-variable model.

    :param latent_variables: Information about each latent variable.
    :param indicators: Information about each indicator.
    :param threshold_systems: Information about each threshold system.
    :param gaussian_indicator_names: Referenced indicators using the Gaussian model.
    :param ordinal_indicator_names: Referenced indicators using an ordinal model.
    :param gaussian_type_names: Indicator types used by Gaussian indicators.
    :param ordinal_type_names: Indicator types used by ordinal indicators.
    """

    latent_variables: dict[str, LatentVariableInfo]
    indicators: dict[str, IndicatorInfo]
    threshold_systems: dict[str, ThresholdSystemInfo]
    gaussian_indicator_names: list[str]
    ordinal_indicator_names: list[str]
    gaussian_type_names: list[str]
    ordinal_type_names: list[str]


# =============================================================================
# Analyzer
# =============================================================================


def analyze_model_structure(
    latent_variables: Iterable[LatentVariable],
    likert_indicators: Iterable[LikertIndicator],
    likert_types: Iterable[LikertType],
) -> ModelStructure:
    """
    Extract structural information from the model specification.

    :param latent_variables: Specifications of latent variables.
    :param likert_indicators: Specifications of Likert indicators.
    :param likert_types: Definitions of Likert scales.
    :return: A ``ModelStructure`` object describing the model.
    """

    latent_variables = list(latent_variables)
    likert_indicators = list(likert_indicators)
    likert_types = list(likert_types)

    # -------------------------------------------------------------------------
    # Build indicator → latent-variable mapping
    # -------------------------------------------------------------------------

    indicator_to_lvs: dict[str, list[str]] = {}

    for lv in latent_variables:
        for ind in lv.indicators:
            indicator_to_lvs.setdefault(ind, []).append(lv.name)

    # -------------------------------------------------------------------------
    # Indicator information
    # -------------------------------------------------------------------------

    indicator_info: dict[str, IndicatorInfo] = {}

    for ind in likert_indicators:
        indicator_info[ind.name] = IndicatorInfo(
            name=ind.name,
            type_name=ind.type_name,
            measurement_model=ind.measurement_model,
            latent_variables=indicator_to_lvs.get(ind.name, []),
        )

    # -------------------------------------------------------------------------
    # Latent-variable information
    # -------------------------------------------------------------------------

    latent_info: dict[str, LatentVariableInfo] = {}

    for lv in latent_variables:

        indicators = list(lv.indicators)

        unique_indicators: list[str] = []
        shared_indicators: list[str] = []

        for ind in indicators:
            if len(indicator_to_lvs.get(ind, [])) == 1:
                unique_indicators.append(ind)
            else:
                shared_indicators.append(ind)

        latent_info[lv.name] = LatentVariableInfo(
            name=lv.name,
            explanatory_variables=list(lv.structural_equation.explanatory_variables),
            indicators=indicators,
            unique_indicators=unique_indicators,
            shared_indicators=shared_indicators,
        )

    gaussian_indicator_names = sorted(
        [
            ind.name
            for ind in likert_indicators
            if ind.name in indicator_to_lvs
            and ind.measurement_model == MeasurementModel.GAUSSIAN
        ]
    )
    ordinal_indicator_names = sorted(
        [
            ind.name
            for ind in likert_indicators
            if ind.name in indicator_to_lvs
            and ind.measurement_model
            in {
                MeasurementModel.ORDERED_PROBIT,
                MeasurementModel.ORDERED_LOGIT,
            }
        ]
    )
    gaussian_type_names = sorted(
        {indicator_info[name].type_name for name in gaussian_indicator_names}
    )
    ordinal_type_names = sorted(
        {indicator_info[name].type_name for name in ordinal_indicator_names}
    )

    # -------------------------------------------------------------------------
    # Threshold-system information
    # -------------------------------------------------------------------------

    threshold_systems: dict[str, ThresholdSystemInfo] = {}

    for t in likert_types:

        indicators = [
            ind.name
            for ind in likert_indicators
            if ind.type_name == t.type_name and ind.name in ordinal_indicator_names
        ]

        if not indicators:
            continue

        threshold_systems[t.type_name] = ThresholdSystemInfo(
            type_name=t.type_name,
            symmetric=t.symmetric,
            categories=list(t.categories),
            neutral_labels=list(t.neutral_labels),
            indicators=indicators,
        )

    # -------------------------------------------------------------------------
    # Final structure
    # -------------------------------------------------------------------------

    return ModelStructure(
        latent_variables=latent_info,
        indicators=indicator_info,
        threshold_systems=threshold_systems,
        gaussian_indicator_names=gaussian_indicator_names,
        ordinal_indicator_names=ordinal_indicator_names,
        gaussian_type_names=gaussian_type_names,
        ordinal_type_names=ordinal_type_names,
    )
