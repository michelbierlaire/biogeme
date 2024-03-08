"""

Estimation and simulation of a nested logit model
=================================================

 We estimate a nested logit model and we perform simulation using the
 estimated model.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 21:05:16 2023

"""

from biogeme import models
import biogeme.biogeme as bio
from optima_data import database
from scenarios import scenario

# %%
# Obtain the specification for the default scenario.
# The definition of the scenarios is available in :ref:`scenarios`.
V, nests, Choice, _ = scenario()

# %%
# The choice model is a nested logit, with availability conditions
# For estimation, we need the log of the probability.
logprob = models.lognested(V, None, nests, Choice)

# %%
# Create the Biogeme object for estimation.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b02estimation'

# %%
# Estimate the parameters. Perform bootstrapping.
the_biogeme.bootstrap_samples = 100
results = the_biogeme.estimate(run_bootstrap=True)

# %%
# Get the results in a pandas table
pandas_results = results.get_estimated_parameters()
pandas_results


# %%
# Simulation
simulated_choices = logprob.get_value_c(
    betas=results.get_beta_values(), database=database
)
simulated_choices

# %%
loglikelihood = logprob.get_value_c(
    betas=results.get_beta_values(),
    database=database,
    aggregation=True,
)
print(f'Final log likelihood:     {results.data.logLike}')
print(f'Simulated log likelihood: {loglikelihood}')
