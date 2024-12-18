"""File translated_estimation.py

:author: Michel Bierlaire, EPFL
:date: Sat Apr 20 17:54:15 2024

Estimation of a MDCEV model with the "translated utility" specification.
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.results_processing import get_pandas_estimated_parameters
from translated_specification import the_translated
from specification import (
    database,
    number_chosen,
    consumed_quantities,
)

# %
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example: translated utility')

# %
# As the model is numerically complex, we adjust the convergence tolerance of the optimization algorithm.
results = the_translated.estimate_parameters(
    database=database,
    number_of_chosen_alternatives=number_chosen,
    consumed_quantities=consumed_quantities,
    tolerance=5.0e-5,
)

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
