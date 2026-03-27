"""Threshold builders for ordinal measurement models.

This module builds cutpoints for ordinal threshold systems associated with
Likert types that are actually used by ordinal indicators
(``MeasurementModel.ORDERED_PROBIT`` or ``MeasurementModel.ORDERED_LOGIT``).

Its single responsibility is to construct cutpoints from:
- LikertType specifications,
- BuildContext (positive factory + naming),
- NormalizationPlan (optional fixings, e.g. first cutpoint for monotone systems).

Gaussian indicators do not use thresholds and their associated types must not
be passed to this module.

Michel Bierlaire
Thu Mar 05 2026, 17:30:40
"""

from __future__ import annotations

from collections.abc import Iterable

from biogeme.expressions import Expression, Numeric

from .context import BuildContext
from .utils import resolve_fixed_or_beta
from ..likert_indicators import LikertType, MeasurementModel
from ..normalization.parameter_refs import ThresholdFirst
from ..normalization.plan import NormalizationPlan


def _ensure_only_ordinal_types(
    *,
    likert_types: Iterable[LikertType],
) -> list[LikertType]:
    """Validate that the provided types correspond to ordinal threshold systems.

    This guard is intentionally conservative: if a type object explicitly carries
    a ``measurement_model`` attribute and that attribute is not ordinal, the
    function raises an error. This prevents Gaussian-only type collections from
    being passed accidentally to the threshold builders.

    Standard ``LikertType`` objects currently do not carry a measurement-model
    field, so in the common case this function simply materializes the iterable
    and returns it unchanged.

    :param likert_types: Candidate Likert types.
    :return: The validated types as a list.
    :raises ValueError: If a provided type is explicitly marked as non-ordinal.
    """
    validated = list(likert_types)
    for lt in validated:
        measurement_model = getattr(lt, "measurement_model", None)
        if measurement_model is None:
            continue
        if measurement_model not in {
            MeasurementModel.ORDERED_PROBIT,
            MeasurementModel.ORDERED_LOGIT,
        }:
            raise ValueError(
                f"Likert type '{lt.type_name}' is associated with non-ordinal measurement model "
                f"'{measurement_model}' and must not be passed to threshold builders."
            )
    return validated


def _build_symmetric_cutpoints(
    *, lt: LikertType, context: BuildContext
) -> list[Expression]:
    """Build symmetric cutpoints centered at 0 (guarantees location normalization)."""
    n_tau = len(lt.categories) - 1
    n_deltas = n_tau // 2

    pos = context.positive_parameter_factory
    naming = context.naming

    deltas: list[Expression] = [
        pos(
            name=naming.threshold_delta_name(lt.type_name, k),
            prefix=naming.threshold_delta_prefix(lt.type_name),
            value=-0.86 + 0.43 * k,
        )
        for k in range(n_deltas)
    ]

    cum: list[Expression] = []
    running: Expression | None = None
    for d in deltas:
        running = d if running is None else running + d
        cum.append(running)

    cutpoints: list[Expression] = []
    for s in reversed(cum):
        cutpoints.append(-s)

    if n_tau % 2 == 1:
        cutpoints.append(Numeric(0.0))

    for s in cum:
        cutpoints.append(s)

    if len(cutpoints) != n_tau:
        raise RuntimeError(
            f"Internal error for '{lt.type_name}': expected {n_tau} cutpoints, got {len(cutpoints)}."
        )
    return cutpoints


def _build_monotone_cutpoints(
    *,
    lt: LikertType,
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> list[Expression]:
    """Build monotone cutpoints using tau_1 and positive increments."""
    n_tau = len(lt.categories) - 1
    pos = context.positive_parameter_factory
    naming = context.naming

    # tau_1: fixed via plan if provided, else free Beta
    tau_1 = resolve_fixed_or_beta(
        target=ThresholdFirst(lt.type_name),
        context_name=naming.threshold_tau1_name(lt.type_name),
        init_value=0.0,
        plan=plan,
    )

    cutpoints: list[Expression] = [tau_1]
    last = tau_1

    # tau_k = tau_{k-1} + delta_{k-1}, delta>0
    for k in range(2, n_tau + 1):
        delta = pos(
            name=naming.threshold_delta_name(lt.type_name, k - 1),
            prefix=naming.threshold_delta_prefix(lt.type_name),
            value=0.3 + 0.5 * (k - 2),
        )
        last = last + delta
        cutpoints.append(last)

    if len(cutpoints) != n_tau:
        raise RuntimeError(
            f"Internal error for '{lt.type_name}': expected {n_tau} cutpoints, got {len(cutpoints)}."
        )
    return cutpoints


def build_cutpoints_for_type(
    *,
    lt: LikertType,
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> list[Expression]:
    """Build cutpoints for one LikertType."""
    if lt.symmetric:
        return _build_symmetric_cutpoints(lt=lt, context=context)
    return _build_monotone_cutpoints(lt=lt, context=context, plan=plan)


def build_all_cutpoints(
    *,
    likert_types: Iterable[LikertType],
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> dict[str, list[Expression]]:
    """Build cutpoints for ordinal Likert types, indexed by ``type_name``.

    :param likert_types: Likert types corresponding to ordinal threshold systems.
    :param context: Build context.
    :param plan: Optional normalization plan.
    :return: Cutpoints indexed by ``type_name``.
    :raises ValueError: If a provided type is explicitly marked as non-ordinal.
    """
    ordinal_types = _ensure_only_ordinal_types(likert_types=likert_types)
    out: dict[str, list[Expression]] = {}
    for lt in ordinal_types:
        if lt.type_name in out:
            raise ValueError(f"Likert type '{lt.type_name}' appears twice.")
        out[lt.type_name] = build_cutpoints_for_type(lt=lt, context=context, plan=plan)
    return out
