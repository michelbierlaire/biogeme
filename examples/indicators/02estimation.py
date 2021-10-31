"""File 02simulation.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 09:46:10 2021

 We estimate a nested logit model and we perform simulation using the
 estimated model.

"""
from biogeme import models
import biogeme.biogeme as bio
from scenarios import scenario, database

# Obtain the specification for the default scenario
V, nests, Choice, _ = scenario()

# The choice model is a nested logit, with availability conditions
# For estimation, we need the log of the probability
logprob = models.lognested(V, None, nests, Choice)

# Create the Biogeme object for estimation
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '02estimation'

print('Estimation...')
# Estimate the parameters. Perform bootstrapping.
results = biogeme.estimate(bootstrap=100)

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)

print('Simulation...')

simulated_choices = logprob.getValue_c(
    betas=results.getBetaValues(), database=database
)

print(simulated_choices)

loglikelihood = logprob.getValue_c(
    betas=results.getBetaValues(),
    database=database,
    aggregation=True,
)

print(f'Final log likelihood:     {results.data.logLike}')
print(f'Simulated log likelihood: {loglikelihood}')
