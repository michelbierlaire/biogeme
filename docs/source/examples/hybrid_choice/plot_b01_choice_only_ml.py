"""
Choice model only — maximum likelihood estimation
=================================================

This script runs a **standard discrete choice model** without any latent
variables, estimated by **maximum likelihood** using Biogeme.

It serves as a baseline specification against which hybrid choice models
(with latent variables and measurement equations) can be compared.

The configuration is defined locally in this file and passed to the generic
estimation pipeline via :func:`estimate_model`.

Michel Bierlaire
Thu Dec 25 2025, 08:24:06
"""

import biogeme.biogeme_logging as blog

from config import Config
from estimate import estimate_model

logger = blog.get_screen_logger(level=blog.INFO)

# Choice model only

the_config = Config(
    name='b01_choice_only_ml',
    latent_variables="zero",
    choice_model="yes",
    estimation="ml",
    number_of_bayesian_draws_per_chain=20_000,
    number_of_monte_carlo_draws=20_000,
)

estimate_model(config=the_config)
