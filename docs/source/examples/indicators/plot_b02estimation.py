"""

Estimation and simulation of a nested logit model
=================================================

 We estimate a nested logit model, and we perform simulation using the
 estimated model.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 16:08:07
"""

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.calculator import get_value_c
from biogeme.data.optima import read_data
from biogeme.models import lognested
from biogeme.results_processing import get_pandas_estimated_parameters

from scenarios import scenario

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example plot_b02estimation')

# %%
# Obtain the specification for the default scenario.
# The definition of the scenarios is available in :ref:`scenarios`.
V, nests, choice, _ = scenario()

# %%
# The choice model is a nested logit, with availability conditions
# For estimation, we need the log of the probability.
log_probability = lognested(util=V, availability=None, nests=nests, choice=choice)

# %%
# Get the database
database = read_data()
# %%
# Create the Biogeme object for estimation.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b02estimation'

# %%
# Estimate the parameters. Perform bootstrapping.
results = the_biogeme.estimate(run_bootstrap=True)

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)


# %%
# Simulation
simulated_choices = get_value_c(
    expression=log_probability,
    betas=results.get_beta_values(),
    database=database,
    numerically_safe=False,
)
display(simulated_choices)

# %%
loglikelihood = get_value_c(
    expression=log_probability,
    betas=results.get_beta_values(),
    database=database,
    aggregation=True,
    numerically_safe=False,
)
print(f'Final log likelihood:     {results.final_log_likelihood}')
print(f'Simulated log likelihood: {loglikelihood}')
