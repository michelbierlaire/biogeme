"""
Structural expression builders.

Single responsibility: build structural (latent variable) expressions from:
- LatentVariable specs
- StructuralEquation specs
- BuildContext (naming, draw type, sigma factory)
- NormalizationPlan (optional fixings)

Michel Bierlaire
Wed Mar 05 2026
"""

from __future__ import annotations

from typing import Iterable

from biogeme.expressions import (
    Beta,
    DistributedParameter,
    Draws,
    Expression,
    LinearTermTuple,
    LinearUtility,
    Numeric,
    Variable,
)

from .context import BuildContext
from .utils import resolve_fixed_or_positive
from ..latent_variables import LatentVariable
from ..normalization.parameter_refs import StructuralCoefficient, StructuralSigma
from ..normalization.plan import NormalizationPlan


def build_structural_deterministic_part(
    *,
    latent_name: str,
    explanatory_variables: Iterable[str],
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> Expression:
    """Build the deterministic linear-in-parameters part for one latent variable."""
    naming = context.naming

    explanatory_variables = list(explanatory_variables)
    if not explanatory_variables:
        return Numeric(0.0)

    coefficients: dict[str, Beta] = {}
    for var in explanatory_variables:
        parameter_name = naming.structural_beta_name(latent_name, var)
        fixed_value = (
            plan.get(StructuralCoefficient(latent_name, var))
            if plan is not None
            else None
        )
        if fixed_value is None:
            coefficients[var] = Beta(parameter_name, 1.1, None, None, 0)
        else:
            coefficients[var] = Beta(parameter_name, float(fixed_value), None, None, 1)

    return LinearUtility(
        [
            LinearTermTuple(beta=coefficients[var], x=Variable(var))
            for var in explanatory_variables
        ]
    )


def resolve_structural_sigma(
    *,
    latent_name: str,
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> Expression:
    """Resolve sigma for one latent variable (fixed via plan, else free via sigma_factory)."""
    prefix = context.naming.structural_sigma_prefix(latent_name)
    free_sigma = context.sigma_factory(prefix=prefix)

    result = resolve_fixed_or_positive(
        target=StructuralSigma(latent_name),
        plan=plan,
        free_expression=free_sigma,
        require_positive=True,
        positive_name_for_error=f"Structural sigma for latent '{latent_name}'",
    )
    return result


def build_structural_expression(
    *,
    latent_name: str,
    explanatory_variables: Iterable[str],
    sigma: Expression,
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> Expression:
    """Build the full structural expression (DistributedParameter) for one latent variable."""
    naming = context.naming
    deterministic = build_structural_deterministic_part(
        latent_name=latent_name,
        explanatory_variables=explanatory_variables,
        context=context,
        plan=plan,
    )
    draws = Draws(
        name=naming.structural_draws_name(latent_name), draw_type=context.draw_type
    )
    return DistributedParameter(latent_name, child=deterministic + sigma * draws)


def build_all_latent_expressions(
    *,
    latent_variables: Iterable[LatentVariable],
    context: BuildContext,
    plan: NormalizationPlan | None,
) -> dict[str, Expression]:
    """Build structural expressions for all latent variables."""
    out: dict[str, Expression] = {}
    for lv in latent_variables:
        if lv.name in out:
            raise ValueError(f"Latent variable '{lv.name}' appears twice.")
        sigma = resolve_structural_sigma(
            latent_name=lv.name, context=context, plan=plan
        )
        out[lv.name] = build_structural_expression(
            latent_name=lv.name,
            explanatory_variables=lv.structural_equation.explanatory_variables,
            sigma=sigma,
            context=context,
            plan=plan,
        )
    return out
