"""MIMIC model. Maximum likelihood estimation

Michel Bierlaire
Tue Dec 23 2025, 14:53:49

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
