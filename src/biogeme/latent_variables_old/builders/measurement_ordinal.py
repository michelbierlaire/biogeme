"""Ordinal measurement-term builders.

This module contains the likelihood builders for ordinal indicators. It builds
ordered-probit and ordered-logit likelihood contributions from the common
latent-indicator equation defined in :mod:`measurement`.

Responsibilities
----------------
- ``build_ordered_probit_term``: build one ordered-probit likelihood term.
- ``build_ordered_logit_term``: build one ordered-logit likelihood term.

The common latent-indicator equation itself is built elsewhere and passed here
as a ``LatentIndicatorEquation`` object.

Michel Bierlaire
Tue Mar 10 2026, 15:27:04
"""

from __future__ import annotations

from biogeme.expressions import Expression, OrderedLogit, OrderedProbit

from .context import BuildContext
from .latent_indicator import LatentIndicatorEquation


def build_ordered_probit_term(
    *,
    equation: LatentIndicatorEquation,
    categories: list[int],
    neutral_labels: list[int],
    cutpoints: list[Expression],
    context: BuildContext,
) -> Expression:
    """Build the ordered-probit likelihood term for one indicator.

    The common latent-indicator equation is normalized by the measurement scale
    before being passed to :class:`~biogeme.expressions.OrderedProbit`.

    :param equation: Common latent-indicator equation.
    :param categories: Ordered category labels.
    :param neutral_labels: Neutral or missing labels.
    :param cutpoints: Ordered cutpoints for the threshold system.
    :param context: Build context.
    :return: Ordered-probit likelihood contribution.
    """
    return OrderedProbit(
        eta=equation.mu / equation.sigma,
        cutpoints=[t / equation.sigma for t in cutpoints],
        y=equation.y,
        categories=categories,
        neutral_labels=neutral_labels,
        enforce_order=context.ordered_probit_enforce_order,
        eps=context.ordered_probit_eps,
    )


def build_ordered_logit_term(
    *,
    equation: LatentIndicatorEquation,
    categories: list[int],
    neutral_labels: list[int],
    cutpoints: list[Expression],
    context: BuildContext,
) -> Expression:
    """Build the ordered-logit likelihood term for one indicator.

    The common latent-indicator equation is normalized by the measurement scale
    before being passed to :class:`~biogeme.expressions.OrderedLogit`.

    :param equation: Common latent-indicator equation.
    :param categories: Ordered category labels.
    :param neutral_labels: Neutral or missing labels.
    :param cutpoints: Ordered cutpoints for the threshold system.
    :param context: Build context.
    :return: Ordered-logit likelihood contribution.
    """
    return OrderedLogit(
        eta=equation.mu / equation.sigma,
        cutpoints=[t / equation.sigma for t in cutpoints],
        y=equation.y,
        categories=categories,
        neutral_labels=neutral_labels,
        enforce_order=context.ordered_probit_enforce_order,
        eps=context.ordered_probit_eps,
    )
