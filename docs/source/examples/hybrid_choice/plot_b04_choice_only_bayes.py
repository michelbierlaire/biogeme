"""Choice model only. Bayesian estimation

Michel Bierlaire
Tue Dec 23 2025, 14:56:09

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
