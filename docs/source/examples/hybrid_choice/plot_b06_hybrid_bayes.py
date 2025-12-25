"""

Hybrid choice model — Bayesian estimation
=========================================

This script estimates the **full hybrid choice model**, combining:

- a discrete choice model, and
- a MIMIC model with two latent variables (structural and measurement equations),

using **Bayesian estimation** in Biogeme.

It represents the most complete specification in the model family and is
primarily used to:

- study identification and normalization under Bayesian inference,
- analyze posterior distributions of both choice and latent-variable parameters,
- compare Bayesian and maximum likelihood hybrid models, and
- assess the added value of latent variables relative to simpler specifications.

The configuration is defined locally in this file and passed to the generic
estimation pipeline via :func:`estimate_model`.

Michel Bierlaire
Thu Dec 25 2025, 08:27:43
"""

import biogeme.biogeme_logging as blog

from config import Config
from estimate import estimate_model

logger = blog.get_screen_logger(level=blog.INFO)

the_config = Config(
    name='b06_hybrid_bayes',
    latent_variables="two",
    choice_model="yes",
    estimation="bayes",
    number_of_bayesian_draws_per_chain=20_000,
    number_of_monte_carlo_draws=20_000,
)

estimate_model(config=the_config)
