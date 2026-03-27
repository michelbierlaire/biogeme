"""
Positive-parameter builders (build-time infrastructure).

This module belongs to the *builders* layer. It defines small, focused utilities
used by builders to create strictly positive Biogeme parameters in a way that is
compatible with the chosen estimation mode.

Two parameterizations are supported:

- Log-parameterization (typical in maximum likelihood):
  create an unconstrained Beta in log-space and return its exponential.

- Bound-parameterization (typical in Bayesian estimation):
  create a Beta with a strictly positive lower bound.

Responsibilities
----------------
- The *protocols* define the callable interfaces expected by builder code.
- `make_positive_parameter_factory` creates a factory for generic positive parameters.
- `make_sigma_factory` creates a factory for sigma parameters only.

Michel Bierlaire
Thu Mar 05 2026, 17:33:13
"""

from __future__ import annotations

from typing import Protocol

from biogeme.expressions import Beta, Expression, exp
from biogeme.floating_point import SMALL_POSITIVE


# =============================================================================
# Protocols (interfaces)
# =============================================================================


class PositiveParameterFactory(Protocol):
    """Callable interface producing a strictly positive parameter expression."""

    def __call__(self, *, name: str, prefix: str, value: float) -> Expression:
        """Create a strictly positive parameter.

        Parameters
        ----------
        name:
            Base parameter name (e.g., ``"sigma"`` or ``"delta_0"``).
            The final Biogeme parameter name is derived from ``prefix`` and
            the chosen parameterization strategy.
        prefix:
            Namespace prefix used to avoid name collisions.
        value:
            Initial value used when creating the underlying Biogeme parameter.
            In log-parameterization this is an initial value for the log-parameter.

        Returns
        -------
        biogeme.expressions.Expression
            An expression guaranteed to be strictly positive.
        """
        ...


class SigmaFactory(Protocol):
    """Callable interface producing a strictly positive sigma expression."""

    def __call__(self, *, prefix: str) -> Expression:
        """Create a strictly positive sigma parameter expression.

        Parameters
        ----------
        prefix:
            Namespace prefix used to avoid name collisions.

        Returns
        -------
        biogeme.expressions.Expression
            An expression guaranteed to be strictly positive.
        """
        ...


# =============================================================================
# Helpers (each does one thing)
# =============================================================================


def _positive_parameter_expression(
    *, use_log: bool, name: str, prefix: str, value: float
) -> Expression:
    """Build one positive-parameter expression according to the chosen parameterization.

    Single responsibility: implement the positivity parameterization rule.
    """
    if use_log:
        # Unconstrained parameter in log-space, exponentiated to enforce positivity.
        return exp(Beta(f"{prefix}_{name}_log", value, None, None, 0))

    # Directly constrained parameter with a strict positive lower bound.
    return Beta(f"{prefix}_{name}", value, SMALL_POSITIVE, None, 0)


# =============================================================================
# Public factory builders (each returns a callable)
# =============================================================================


def make_positive_parameter_factory(*, use_log: bool) -> PositiveParameterFactory:
    """Create a factory for generic strictly positive parameters.

    Single responsibility: return a callable that builds positive parameters
    using the chosen parameterization.
    """

    def factory(*, name: str, prefix: str, value: float) -> Expression:
        return _positive_parameter_expression(
            use_log=use_log,
            name=name,
            prefix=prefix,
            value=value,
        )

    return factory


def make_sigma_factory(*, use_log: bool) -> SigmaFactory:
    """Create a factory for strictly positive sigma parameters.

    Single responsibility: return a callable dedicated to sigma creation.
    """
    positive_factory = make_positive_parameter_factory(use_log=use_log)
    default_init = -1.0 if use_log else 1.0

    def sigma_factory(*, prefix: str) -> Expression:
        return positive_factory(name="sigma", prefix=prefix, value=default_init)

    return sigma_factory
