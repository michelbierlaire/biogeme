"""Factory utilities for strictly positive parameters.

This module provides a generic mechanism to define positive model parameters
in a way that is compatible with both maximum likelihood and Bayesian
estimation.

In a maximum likelihood context, positivity is typically enforced by
estimating the logarithm of the parameter and exponentiating it. This avoids
explicit constraints and generally improves numerical stability.

In a Bayesian context, it is often preferable to work directly with the
parameter itself and impose positivity through a lower bound set to a small
positive value. This leads to more interpretable posterior draws and avoids
artificial symmetry in the parameter space.

The use of factories abstracts away these implementation details: components
that rely on positive parameters do not need to know whether a log-transformed
or directly constrained representation is used.
"""

from __future__ import annotations

from typing import Protocol

from biogeme.expressions import Beta, Expression, exp
from biogeme.floating_point import SMALL_POSITIVE


class PositiveParameterFactory(Protocol):
    """Protocol for factories creating strictly positive model parameters.

    Implementations return a Biogeme :class:`~biogeme.expressions.Expression`
    representing a parameter constrained to be strictly positive.
    """

    def __call__(
        self,
        name: str,
        prefix: str,
        value: float,
    ) -> Expression:
        """Create a strictly positive parameter expression.

        :param name:
            Base name of the parameter (for example ``"sigma"`` or ``"delta_0"``).
        :param prefix:
            Prefix used to namespace parameter names (for example an indicator or
            type label).
        :param value:
            Initial value used when creating the underlying Biogeme parameter.
        :return:
            An expression that evaluates to a strictly positive value.
        """
        ...


class SigmaFactory(Protocol):
    """Protocol for factories creating strictly positive sigma (scale) parameters."""

    def __call__(self, prefix: str) -> Expression:
        """Create a strictly positive sigma parameter expression.

        :param prefix:
            Prefix used to namespace the sigma parameter name.
        :return:
            An expression that evaluates to a strictly positive sigma.
        """
        ...


def make_positive_parameter_factory(*, use_log: bool) -> PositiveParameterFactory:
    """Create a factory for strictly positive parameters.

    Two parameterizations are supported:

    - If ``use_log`` is True, the returned factory creates an unconstrained
      parameter in log-space (named ``"<prefix>_<name>_log"``) and returns its
      exponential. This is typically used for maximum-likelihood estimation.
    - If ``use_log`` is False, the returned factory creates a directly
      constrained parameter (named ``"<prefix>_<name>"``) with a lower bound set
      to :data:`~biogeme.floating_point.SMALL_POSITIVE`. This is typically used
      for Bayesian estimation.

    :param use_log:
        If True, define parameters in log-space and exponentiate them.
    :return:
        A callable factory compatible with :class:`PositiveParameterFactory`.
    """

    def factory(
        name: str,
        prefix: str,
        value: float,
    ) -> Expression:
        if use_log:
            return exp(Beta(f"{prefix}_{name}_log", value, None, None, 0))
        return Beta(f"{prefix}_{name}", value, SMALL_POSITIVE, None, 0)

    return factory


def make_sigma_factory(*, use_log: bool) -> SigmaFactory:
    """Create a sigma factory (specialization of the positive-parameter factory).

    The sigma factory creates a strictly positive scale parameter named
    ``"<prefix>_sigma"`` (or ``"<prefix>_sigma_log"`` in log-space) using the
    same parameterization rules as :func:`make_positive_parameter_factory`.

    :param use_log:
        If True, define sigma in log-space and exponentiate it.
    :return:
        A callable factory compatible with :class:`SigmaFactory`.
    """

    positive_factory = make_positive_parameter_factory(use_log=use_log)
    value = -1 if use_log else 1

    def sigma_factory(prefix: str) -> Expression:
        return positive_factory(
            "sigma",
            prefix,
            value=value,
        )

    return sigma_factory
