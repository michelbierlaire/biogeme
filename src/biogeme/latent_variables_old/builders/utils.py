"""
Shared builder utilities.

This module holds:
- fixed-or-free resolution helpers (centralized to avoid repeated if/else),
- indexing + validation of specification components.

Michel Bierlaire
Thu Mar 05 2026, 11:20:20
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from biogeme.expressions import Beta, Expression, Numeric

from ..latent_variables import LatentVariable
from ..likert_indicators import LikertIndicator, LikertType, MeasurementModel
from ..normalization.parameter_refs import ParameterRef
from ..normalization.plan import NormalizationPlan


# ---------------------------------------------------------------------
# Fixed-or-free helpers (single responsibility: resolve an expression)
# ---------------------------------------------------------------------


def resolve_fixed_or_beta(
    *,
    target: ParameterRef,
    context_name: str,
    init_value: float = 0.0,
    plan: NormalizationPlan | None,
) -> Expression:
    """Return Numeric(fixed) if fixed in plan; else a free Beta(context_name)."""
    fixed = plan.get(target) if plan is not None else None
    if fixed is not None:
        return Numeric(float(fixed))
    return Beta(context_name, init_value, None, None, 0)


def resolve_fixed_or_positive(
    *,
    target: ParameterRef,
    plan: NormalizationPlan | None,
    free_expression: Expression,
    require_positive: bool = True,
    positive_name_for_error: str,
) -> Expression:
    """Return Numeric(fixed) if fixed in plan; else the provided free expression."""
    fixed = plan.get(target) if plan is not None else None
    if fixed is None:
        return free_expression

    value = float(fixed)
    if require_positive and value <= 0:
        raise ValueError(f"{positive_name_for_error} must be > 0, got {value}.")
    return Numeric(value)


# ---------------------------------------------------------------------
# Spec preparation (single responsibility: validate & index specs once)
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedSpecs:
    """Indexed views of specs used by builders."""

    latent_variables: list[LatentVariable]
    likert_indicators: list[LikertIndicator]
    likert_types: list[LikertType]

    indicator_by_name: dict[str, LikertIndicator]
    type_by_name: dict[str, LikertType]
    referenced_indicator_names: set[str]
    gaussian_indicator_names: set[str]
    ordinal_indicator_names: set[str]
    gaussian_type_names: set[str]
    ordinal_type_names: set[str]


def prepare_specs(
    *,
    latent_variables: Iterable[LatentVariable],
    likert_indicators: Iterable[LikertIndicator],
    likert_types: Iterable[LikertType],
) -> PreparedSpecs:
    """Validate and index the specs once, for consistent downstream building."""
    lvs = list(latent_variables)
    inds = list(likert_indicators)
    types = list(likert_types)

    lv_names = [lv.name for lv in lvs]
    if len(set(lv_names)) != len(lv_names):
        raise ValueError(f"Duplicate latent variable names in spec: {lv_names}")

    indicator_by_name: dict[str, LikertIndicator] = {}
    for ind in inds:
        if ind.name in indicator_by_name:
            raise ValueError(f"Duplicate Likert indicator name in spec: '{ind.name}'.")
        indicator_by_name[ind.name] = ind

    type_by_name: dict[str, LikertType] = {}
    for t in types:
        if t.type_name in type_by_name:
            raise ValueError(f"Duplicate Likert type_name in spec: '{t.type_name}'.")
        if len(t.categories) < 2:
            raise ValueError(
                f"Likert type '{t.type_name}' must have at least 2 categories."
            )
        type_by_name[t.type_name] = t

    referenced = {name for lv in lvs for name in lv.indicators}

    missing_indicators = sorted(referenced - set(indicator_by_name.keys()))
    if missing_indicators:
        raise ValueError(
            f"Indicators referenced by latent variables but not defined: {missing_indicators}"
        )

    # Type existence check for referenced indicators
    unknown_types: set[str] = set()
    for ind_name in referenced:
        tname = indicator_by_name[ind_name].type_name
        if tname not in type_by_name:
            unknown_types.add(tname)

    if unknown_types:
        raise ValueError(
            f"Unknown Likert types referenced by indicators: {sorted(unknown_types)}"
        )

    gaussian_indicator_names = {
        ind_name
        for ind_name in referenced
        if indicator_by_name[ind_name].measurement_model == MeasurementModel.GAUSSIAN
    }
    ordinal_indicator_names = {
        ind_name
        for ind_name in referenced
        if indicator_by_name[ind_name].measurement_model
        in {
            MeasurementModel.ORDERED_PROBIT,
            MeasurementModel.ORDERED_LOGIT,
        }
    }

    gaussian_type_names = {
        indicator_by_name[ind_name].type_name for ind_name in gaussian_indicator_names
    }
    ordinal_type_names = {
        indicator_by_name[ind_name].type_name for ind_name in ordinal_indicator_names
    }

    return PreparedSpecs(
        latent_variables=lvs,
        likert_indicators=inds,
        likert_types=types,
        indicator_by_name=indicator_by_name,
        type_by_name=type_by_name,
        referenced_indicator_names=referenced,
        gaussian_indicator_names=gaussian_indicator_names,
        ordinal_indicator_names=ordinal_indicator_names,
        gaussian_type_names=gaussian_type_names,
        ordinal_type_names=ordinal_type_names,
    )
