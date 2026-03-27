"""
High-level wrapper for normalization advice.

This module provides a simple entry point that takes a hybrid choice model
specification as input and returns a human-readable normalization report.

The report is advisory: it explains which elements typically need
normalization, proposes reasonable anchors, and suggests explicit fixings.
It does not generate a normalization plan automatically.

Michel Bierlaire
"""

from __future__ import annotations

from collections.abc import Iterable

from biogeme.latent_variables.latent_variables import LatentVariable
from biogeme.latent_variables.likert_indicators import LikertIndicator, LikertType

from .advisor import advise_normalization
from .analyzer import analyze_model_structure
from .report import generate_normalization_report


def generate_normalization_advice_report(
    *,
    latent_variables: Iterable[LatentVariable],
    likert_indicators: Iterable[LikertIndicator],
    likert_types: Iterable[LikertType],
) -> str:
    """Generate a human-readable normalization report from a model specification.

    The report is generated in three steps:

    - analyze the structure of the latent-variable and indicator specification,
    - derive normalization advice from that structure,
    - format the advice as a human-readable report.

    :param latent_variables: Specifications of latent variables.
    :param likert_indicators: Specifications of indicators, including their
        measurement models.
    :param likert_types: Definitions of Likert scale types.
    :return: Human-readable normalization report.
    """
    structure = analyze_model_structure(
        latent_variables=latent_variables,
        likert_indicators=likert_indicators,
        likert_types=likert_types,
    )

    advice = advise_normalization(structure)

    return generate_normalization_report(advice)
