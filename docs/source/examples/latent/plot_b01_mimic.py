"""

Multiple Indicators, Multiple Causes (MIMIC)
============================================

MIMIC model with two latent variables.

Michel Bierlaire, EPFL
Sat Nov 08 2025, 15:31:49
"""

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    MonteCarlo,
    log,
)
from biogeme.results_processing import (
    get_pandas_estimated_parameters,
)

from measurement_equations import likert_likelihood_indicator
from optima import (
    read_data,
)
from read_or_estimate import read_or_estimate

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Conditional on the latent variables, we have a logit model (called the kernel)
cond_prob = likert_likelihood_indicator

# %%
# We integrate over omega using numerical integration
log_likelihood = log(MonteCarlo(cond_prob))

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(
    database,
    log_likelihood,
    number_of_draws=10_000,
    calculating_second_derivatives='never',
    numerically_safe=True,
    max_iterations=5000,
)
the_biogeme.model_name = 'b01_mimic'

# %%
# If estimation results are saved on file, we read them to speed up the process.
# If not, we estimate the parameters.
results = read_or_estimate(the_biogeme=the_biogeme, directory='saved_results')

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
