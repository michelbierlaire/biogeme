"""

Configuration
=============

Central configuration for running estimation scripts.

This module defines the :class:`Config` dataclass, which gathers all high-level
options controlling the behavior of the estimation pipeline. A single instance
of :class:`Config` is typically created in a small configuration file (e.g.
`conf_01.py`, `conf_02.py`) and passed to the main execution routine.

The goal of this module is to provide a clear, typed, and immutable container
for experimental settings, so that the same codebase can be reused across
multiple configurations without duplication.

Michel Bierlaire
Thu Dec 25 2025, 08:08:37
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Config:
    """Configuration of a single estimation run.

    Each field controls a specific modeling or estimation choice:

    - ``name``: Human-readable identifier for the configuration (used for logging
      and output naming).
    - ``latent_variables``: Specifies whether latent variables are included in the
      model (``"zero"``) or whether the full hybrid choice model with two latent
      variables is used (``"two"``).
    - ``choice_model``: Indicates whether a discrete choice model is included
      alongside the latent-variable measurement model (``"yes"`` or ``"no"``).
    - ``estimation``: Estimation paradigm, either Bayesian (``"bayes"``) or
      maximum likelihood (``"ml"``).
    - ``number_of_bayesian_draws_per_chain``: Number of posterior draws per MCMC
      chain when Bayesian estimation is used.
    - ``number_of_monte_carlo_draws``: Number of Monte Carlo draws used for
      numerical integration in maximum likelihood estimation.

    The dataclass is frozen to guarantee immutability during execution and
    improve reproducibility.
    """

    name: str
    latent_variables: Literal["zero", "two"]
    choice_model: Literal["yes", "no"]
    estimation: Literal["bayes", "ml"]
    number_of_bayesian_draws_per_chain: int
    number_of_monte_carlo_draws: int
