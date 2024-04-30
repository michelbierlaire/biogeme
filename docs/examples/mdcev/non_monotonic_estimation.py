"""File non_monotonic_estimation.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 18:29:30 2024

Estimation of a MDCEV model with the "non monotonic utility" specification.
"""

import biogeme.biogeme_logging as blog
from non_monotonic_specification import the_non_monotonic
from specification import (
    database,
    number_chosen,
    consumed_quantities,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: non monotonic utility')

results = the_non_monotonic.estimate_parameters(
    database=database,
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
)

# %
print(results.short_summary())

# %
print(results.get_estimated_parameters())
