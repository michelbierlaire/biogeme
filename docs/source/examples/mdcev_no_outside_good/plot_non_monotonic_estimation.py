"""File non_monotonic_estimation.py

Michel Bierlaire, EPFL
Fri Jul 25 2025, 17:14:53

Estimation of a MDCEV model with the "non monotonic utility" specification.
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.results_processing import get_pandas_estimated_parameters
from non_monotonic_specification import the_non_monotonic
from process_data import database, number_chosen
from specification import consumed_quantities

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: non monotonic utility')

results = the_non_monotonic.estimate_parameters(
    database=database,
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
    tolerance=0.0004,
)

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
