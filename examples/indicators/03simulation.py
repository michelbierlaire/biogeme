"""File 03simulation.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 10:50:48 2021

We use an estimated model to perform various simulations
"""
import sys
import time
import pandas as pd
from biogeme import models
import biogeme.biogeme as bio
import biogeme.exceptions as excep
import biogeme.results as res
from scenarios import scenario, database, normalizedWeight

# Obtain the specification for the default scenario
V, nests, _, _ = scenario()

V_PT = V[0]
V_CAR = V[1]
V_SM = V[2]

# Obtain the expression for the choice probability of each alternative
prob_PT = models.nested(V, None, nests, 0)
prob_CAR = models.nested(V, None, nests, 1)
prob_SM = models.nested(V, None, nests, 2)

# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='02estimation.pickle')
except excep.biogemeError:
    sys.exit(
        'Run first the script 02simulation.py '
        'in order to generate the '
        'file 02estimation.pickle.'
    )


# We now simulate various expressions on the database, and store the
# results in a Pandas dataframe

start_time = time.time()
simulate_formulas = {
    'weight': normalizedWeight.getValue_c(
        betas=results.getBetaValues(), database=database, prepareIds=True
    ),
    'Utility PT': V_PT.getValue_c(
        betas=results.getBetaValues(), database=database, prepareIds=True
    ),
    'Utility car': V_CAR.getValue_c(
        betas=results.getBetaValues(), database=database, prepareIds=True
    ),
    'Utility SM': V_SM.getValue_c(
        betas=results.getBetaValues(), database=database, prepareIds=True
    ),
    'Prob. PT': prob_PT.getValue_c(
        betas=results.getBetaValues(), database=database, prepareIds=True
    ),
    'Prob. car': prob_CAR.getValue_c(
        betas=results.getBetaValues(), database=database, prepareIds=True
    ),
    'Prob. SM': prob_SM.getValue_c(
        betas=results.getBetaValues(), database=database, prepareIds=True
    ),
}

simulated_values = pd.DataFrame.from_dict(
    simulate_formulas,
)
print(
    f'--- Execution time with getValue_c: '
    f'{time.time() - start_time:.2f} seconds ---'
)

# We now perform the same simulation using Biogeme. The results are be
# identical, but the execution time is faster. Indeed, Biogeme
# recycles calculations performed for one expression for the other
# expressions.

# A dictionary with the requested expression must be provided to Biogeme

simulate = {
    'weight': normalizedWeight,
    'Utility PT': V_PT,
    'Utility car': V_CAR,
    'Utility SM': V_SM,
    'Prob. PT': prob_PT,
    'Prob. car': prob_CAR,
    'Prob. SM': prob_SM,
}

start_time = time.time()
biogeme = bio.BIOGEME(database, simulate)
biogeme_simulation = biogeme.simulate(results.getBetaValues())
print(
    f'--- Execution time with Biogeme:    '
    f'{time.time() - start_time:.2f} seconds ---'
)

# Let's print the two results, to show that they are identical

print('Results without Biogeme')
print(simulated_values)
print('Results with Biogeme')
print(biogeme_simulation)
