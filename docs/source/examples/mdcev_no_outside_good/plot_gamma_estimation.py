"""File gamma_estimation.py

Michel Bierlaire, EPFL
Fri Jul 25 2025, 16:36:50
Estimation of a MDCEV model with the "gamma_profile" specification.
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.results_processing import get_pandas_estimated_parameters
from gamma_specification import the_gamma_profile
from process_data import database, number_chosen
from specification import consumed_quantities

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
