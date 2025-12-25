"""
MIMIC model — maximum likelihood estimation
===========================================

This script estimates a **pure MIMIC model** (measurement and structural
components only) using **maximum likelihood**, without an associated discrete
choice model.

It is mainly intended to:

- test and validate the latent-variable specification,
- assess identification and normalization issues, and
- serve as a building block for hybrid choice models.

The model configuration is defined locally in this file and passed to the
generic estimation pipeline via :func:`estimate_model`.

Michel Bierlaire
Thu Dec 25 2025, 08:24:35
"""

import biogeme.biogeme_logging as blog

from config import Config
from estimate import estimate_model

logger = blog.get_screen_logger(level=blog.INFO)

the_config = Config(
    name='b02_mimic_ml',
    latent_variables="two",
    choice_model="no",
    estimation="ml",
    number_of_bayesian_draws_per_chain=20_000,
    number_of_monte_carlo_draws=20_000,
)

estimate_model(config=the_config)
