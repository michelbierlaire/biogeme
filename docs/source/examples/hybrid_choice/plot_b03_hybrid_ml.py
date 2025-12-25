"""
Hybrid choice model — maximum likelihood estimation
===================================================

This script estimates a **hybrid choice model** that combines:

- a discrete choice model, and
- a MIMIC model with two latent variables (structural and measurement equations),

using **maximum likelihood estimation** in Biogeme.

It represents the full model specification, bringing together the choice
component and the latent-variable component, and can be compared against:

- the choice-only model, and
- the MIMIC-only model,

to assess the contribution of latent variables to model performance.

The configuration is defined locally in this file and passed to the generic
estimation pipeline via :func:`estimate_model`.

Michel Bierlaire
Thu Dec 25 2025, 08:25:28
"""

import biogeme.biogeme_logging as blog

from config import Config
from estimate import estimate_model

logger = blog.get_screen_logger(level=blog.INFO)

the_config = Config(
    name='b03_hybrid_ml',
    latent_variables="two",
    choice_model="yes",
    estimation="ml",
    number_of_bayesian_draws_per_chain=20_000,
    number_of_monte_carlo_draws=20_000,
)

estimate_model(config=the_config)
