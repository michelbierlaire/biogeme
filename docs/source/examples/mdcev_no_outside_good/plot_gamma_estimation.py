"""File gamma_estimation.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 11:02:42 2024

Estimation of a MDCEV model with the "gamma_profile" specification.
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.results_processing import get_pandas_estimated_parameters
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

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
