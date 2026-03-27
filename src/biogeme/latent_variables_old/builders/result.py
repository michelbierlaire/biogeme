"""
Result objects.

Michel Bierlaire
Thu Mar 05 2026, 17:33:13

"""

from __future__ import annotations

from dataclasses import dataclass

from biogeme.expressions import Expression


@dataclass(frozen=True, slots=True)
class HybridBuildResult:
    """Outputs of the hybrid builder."""

    latent_expressions: dict[str, Expression]
    thresholds_by_type: dict[str, list[Expression]]
    measurement_terms: dict[str, Expression]
    measurement_product: Expression
    measurement_logsum: Expression
