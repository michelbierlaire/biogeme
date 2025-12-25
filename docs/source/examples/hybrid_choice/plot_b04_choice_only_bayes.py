"""
Choice model only — Bayesian estimation
=======================================

This script estimates a **standard discrete choice model** without any latent
variables using **Bayesian estimation** in Biogeme.

It serves as the Bayesian counterpart of the choice-only maximum likelihood
specification and provides a baseline for comparison with:

- the Bayesian hybrid choice model, and
- the corresponding maximum likelihood estimates.

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
    name='b04_choice_only_bayes',
    latent_variables="zero",
    choice_model="yes",
    estimation="bayes",
    number_of_bayesian_draws_per_chain=20_000,
    number_of_monte_carlo_draws=20_000,
)

estimate_model(config=the_config)
