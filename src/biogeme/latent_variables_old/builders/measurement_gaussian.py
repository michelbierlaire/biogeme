"""Gaussian measurement-term builder.

This module contains the likelihood builder for continuous indicators treated
with a Gaussian measurement equation. It consumes the common latent-indicator
 equation defined in :mod:`measurement`.

For one indicator, the latent-indicator equation is:

.. math::

    I^* = \mu + \sigma \varepsilon,

with :math:`\varepsilon \sim \mathcal N(0, 1)`.

If the observed indicator is treated as continuous, the likelihood contribution
is the Gaussian density:

.. math::

    f(y \mid \mu, \sigma)
    = \frac{1}{\sigma} \phi\left(\frac{y - \mu}{\sigma}\right).

Responsibilities
----------------
- ``build_gaussian_term``: build one Gaussian measurement likelihood term.

Michel Bierlaire
Thu Mar 05 2026, 18:12:00
"""

from __future__ import annotations

from biogeme.distributions import normalpdf
from biogeme.expressions import Expression

from .latent_indicator import LatentIndicatorEquation


def build_gaussian_term(*, equation: LatentIndicatorEquation) -> Expression:
    """Build the Gaussian likelihood term for one indicator.

    The common latent-indicator equation is used directly as a continuous
    Gaussian measurement model.

    :param equation: Common latent-indicator equation.
    :return: Gaussian likelihood contribution.
    """
    standardized_residual = (equation.y - equation.mu) / equation.sigma
    return normalpdf(standardized_residual) / equation.sigma
