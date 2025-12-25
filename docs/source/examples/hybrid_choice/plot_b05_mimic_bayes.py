"""
MIMIC model — Bayesian estimation
=================================

This script estimates a **pure MIMIC model** (latent-variable structural and
measurement equations only) using **Bayesian estimation** in Biogeme, without
an associated discrete choice model.

It is primarily intended to:

- assess identification and normalization under Bayesian inference,
- inspect posterior distributions of latent-variable parameters, and
- provide a Bayesian benchmark for comparison with the maximum likelihood
  MIMIC specification.

The configuration is defined locally in this file and passed to the generic
estimation pipeline via :func:`estimate_model`.

Michel Bierlaire
Thu Dec 25 2025, 08:27:04
"""

import biogeme.biogeme_logging as blog

from config import Config
from estimate import estimate_model

logger = blog.get_screen_logger(level=blog.INFO)

the_config = Config(
    name='b05_mimic_bayes',
    latent_variables="two",
    choice_model="no",
    estimation="bayes",
    number_of_bayesian_draws_per_chain=20_000,
    number_of_monte_carlo_draws=20_000,
)

estimate_model(config=the_config)
