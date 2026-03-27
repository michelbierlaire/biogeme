"""
Report generation for normalization advice.

This module converts the output of the normalization advisor into a
human-readable textual report. The report reflects the distinction between
latent-variable normalization, which applies to all measurement models, and
threshold-system normalization, which applies only to ordinal measurement
models.

Michel Bierlaire
Fri Mar 06 2026, 10:57:13
"""

from __future__ import annotations

from .advisor import NormalizationAdvice


def _format_section_title(title: str) -> str:
    """Format a section title.

    :param title: Section title.
    :return: Formatted title string.
    """
    line = "-" * len(title)
    return f"{title}\n{line}"


def _format_latent_variable_section(advice: NormalizationAdvice) -> str:
    """Format the latent-variable normalization section.

    :param advice: Normalization advice.
    :return: Formatted section.
    """
    lines: list[str] = [_format_section_title("Latent-variable normalization")]

    if not advice.latent_variable_advice:
        lines.append("No latent variables found.")
        return "\n".join(lines)

    for lv_name in sorted(advice.latent_variable_advice):
        lv_advice = advice.latent_variable_advice[lv_name]

        lines.append(f"\n{lv_name}")
        lines.append(
            f"  Recommended reference indicator: {lv_advice.recommended_indicator}"
        )
        lines.append(f"  Location normalization: {lv_advice.location_normalization}")
        lines.append(f"  Scale normalization: {lv_advice.scale_normalization}")

        if lv_advice.warning is not None:
            lines.append(f"  Warning: {lv_advice.warning}")

    return "\n".join(lines)


def _format_threshold_system_section(advice: NormalizationAdvice) -> str:
    """Format the threshold-system normalization section.

    :param advice: Normalization advice.
    :return: Formatted section.
    """
    lines: list[str] = [
        _format_section_title(
            "Threshold-system normalization (ordinal indicators only)"
        )
    ]

    if not advice.threshold_system_advice:
        lines.append(
            "No ordinal threshold systems found. No threshold-system normalization is needed."
        )
        return "\n".join(lines)

    for type_name in sorted(advice.threshold_system_advice):
        threshold_advice = advice.threshold_system_advice[type_name]

        lines.append(f"\n{type_name}")
        lines.append(f"  Symmetric: {threshold_advice.symmetric}")
        lines.append(
            f"  Recommended reference indicator for sigma normalization: "
            f"{threshold_advice.reference_indicator}"
        )
        lines.append(
            f"  Location normalization: {threshold_advice.location_normalization}"
        )
        lines.append(f"  Scale normalization: {threshold_advice.scale_normalization}")

        if threshold_advice.warning is not None:
            lines.append(f"  Warning: {threshold_advice.warning}")

    return "\n".join(lines)


def _format_suggested_fixings_section(advice: NormalizationAdvice) -> str:
    """Format the suggested-fixings section.

    :param advice: Normalization advice.
    :return: Formatted section.
    """
    lines: list[str] = [_format_section_title("Suggested fixings")]

    if not advice.suggested_fixings:
        lines.append("No fixing could be suggested.")
        return "\n".join(lines)

    for fixing in advice.suggested_fixings:
        lines.append(f"\n- {fixing.parameter} = {fixing.value}")
        lines.append(f"  Reason: {fixing.reason}")

    return "\n".join(lines)


def _format_warnings_section(advice: NormalizationAdvice) -> str:
    """Format the warnings section.

    :param advice: Normalization advice.
    :return: Formatted section.
    """
    lines: list[str] = [_format_section_title("Warnings")]

    if not advice.warnings:
        lines.append("No specific warning.")
        return "\n".join(lines)

    for warning in advice.warnings:
        lines.append(f"- {warning}")

    return "\n".join(lines)


def generate_normalization_report(advice: NormalizationAdvice) -> str:
    """Generate a full normalization report.

    :param advice: Normalization advice.
    :return: Human-readable report as a string.
    """
    sections = [
        _format_section_title("Normalization advice"),
        advice.disclaimer,
        "",
        _format_latent_variable_section(advice),
        "",
        _format_threshold_system_section(advice),
        "",
        _format_suggested_fixings_section(advice),
        "",
        _format_warnings_section(advice),
    ]
    return "\n".join(sections)
