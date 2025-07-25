"""File generalized_estimation.py

Michel Bierlaire, EPFL
Fri Jul 25 2025, 16:51:40

Estimation of a MDCEV model with the "generalized translated utility" specification.
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.results_processing import get_pandas_estimated_parameters
from generalized_specification import the_generalized
from process_data import database, number_chosen
from specification import consumed_quantities

# %
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: generalized translated utility')

# %
results = the_generalized.estimate_parameters(
    database=database,
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
    tolerance=4e-5,
)

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
