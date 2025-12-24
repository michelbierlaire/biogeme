"""Choice model only. Maximum likelihood estimation

Michel Bierlaire
Tue Dec 23 2025, 14:52:48

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
