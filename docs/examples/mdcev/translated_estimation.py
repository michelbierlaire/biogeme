"""File translated_estimation.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 17:54:15 2024

Estimation of a MDCEV model with the "translated utility" specification.
"""

import biogeme.biogeme_logging as blog
from translated_specification import the_translated
from specification import (
    database,
    number_chosen,
    consumed_quantities,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: translated utility')

results = the_translated.estimate_parameters(
    database=database,
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
)

# %
print(results.short_summary())

# %
print(results.get_estimated_parameters())
