"""
Latent-indicator equation container.

This module defines the small data structure used to represent the
*latent-indicator equation* that underlies all measurement models in
Biogeme's hybrid choice implementation.

Conceptual model
----------------
Each indicator is assumed to be generated from a latent continuous
variable

    I* = μ + ε

where

- μ is the systematic component depending on latent variables
  and possibly observed variables,
- ε is a stochastic error term with scale parameter σ.

Different measurement models interpret this latent equation
in different ways:

- **Gaussian model**
  The observed indicator is treated as continuous and
  the likelihood corresponds directly to

      y ~ Normal(μ, σ)

- **Ordered probit model**
  The latent variable I* is discretized using thresholds
  and ε is assumed to be normally distributed.

- **Ordered logit model**
  The latent variable I* is discretized using thresholds
  and ε follows an extreme value distribution.

The class defined here is intentionally minimal: it only stores the
symbolic Biogeme expressions required to build the likelihood terms.
The actual likelihood computation is delegated to the corresponding
measurement builders.
"""

from dataclasses import dataclass

from biogeme.expressions import Expression


@dataclass(frozen=True, slots=True)
class LatentIndicatorEquation:
    """Common latent-indicator equation used by all measurement models.

    :param indicator_name: Name of the observed indicator.
    :param y: Biogeme variable representing the observed indicator.
    :param mu: Systematic part of the latent-indicator equation.
    :param sigma: Measurement scale parameter.
    """

    indicator_name: str
    y: Expression
    mu: Expression
    sigma: Expression
