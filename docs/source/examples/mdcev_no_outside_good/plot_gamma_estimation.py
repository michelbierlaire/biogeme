"""File gamma_estimation.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 11:02:42 2024

Estimation of a MDCEV model with the "gamma_profile" specification.
"""

import biogeme.biogeme_logging as blog
from gamma_specification import the_gamma_profile
from specification import (
    database,
    number_chosen,
    consumed_quantities,
)

# %
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: gamma profile')

# %
results = the_gamma_profile.estimate_parameters(
    database=database,
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
)

# %
print(results.short_summary())

# %
print(results.get_estimated_parameters())
