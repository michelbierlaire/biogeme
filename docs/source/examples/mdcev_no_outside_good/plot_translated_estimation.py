"""File translated_estimation.py

Michel Bierlaire, EPFL
Fri Jul 25 2025, 17:28:50

Estimation of a MDCEV model with the "translated utility" specification.
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.results_processing import get_pandas_estimated_parameters
from process_data import database, number_chosen
from specification import consumed_quantities
from translated_specification import the_translated

# %
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: translated utility')

# %
# As the model is numerically complex, we adjust the convergence tolerance of the optimization algorithm.
results = the_translated.estimate_parameters(
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
