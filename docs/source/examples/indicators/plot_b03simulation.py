"""

Simulation of a choice model
============================

We use an estimated model to perform various simulations.

Michel Bierlaire, EPFL
Tue Apr 29 2025, 09:30:56
"""

import sys
import time

import pandas as pd

from biogeme.biogeme import BIOGEME
from biogeme.calculator import get_value_c
from biogeme.data.optima import normalized_weight, read_data
from biogeme.models import nested
from biogeme.results_processing import EstimationResults
from scenarios import scenario

# %%
# Obtain the specification for the default scenario.
# The definition of the scenarios is available in :ref:`scenarios`.
V, nests, _, _ = scenario()

V_PT = V[0]
V_CAR = V[1]
V_SM = V[2]

# %%
# Obtain the expression for the choice probability of each alternative.
prob_PT = nested(V, None, nests, 0)
prob_CAR = nested(V, None, nests, 1)
prob_SM = nested(V, None, nests, 2)

# Read the estimation results from the file
try:
    results = EstimationResults.from_yaml_file(
        filename='saved_results/b02estimation.yaml'
    )
except FileNotFoundError:
    sys.exit(
        'Run first the script b02simulation.py '
        'in order to generate the '
        'file b02estimation.yaml.'
    )

# %%
# Read the database
database = read_data()

# %%
# We now simulate various expressions on the database, and store the
# results in a Pandas dataframe.
# %%
start_time = time.time()
simulate_formulas = {
    'weight': get_value_c(
        expression=normalized_weight, betas=results.get_beta_values(), database=database
    ),
    'Utility PT': get_value_c(
        expression=V_PT, betas=results.get_beta_values(), database=database
    ),
    'Utility car': get_value_c(
        expression=V_CAR, betas=results.get_beta_values(), database=database
    ),
    'Utility SM': get_value_c(
        expression=V_SM, betas=results.get_beta_values(), database=database
    ),
    'Prob. PT': get_value_c(
        expression=prob_PT, betas=results.get_beta_values(), database=database
    ),
    'Prob. car': get_value_c(
        expression=prob_CAR, betas=results.get_beta_values(), database=database
    ),
    'Prob. SM': get_value_c(
        expression=prob_SM, betas=results.get_beta_values(), database=database
    ),
}

# %%
simulated_values = pd.DataFrame.from_dict(
    simulate_formulas,
)

# %%
print(
    f'--- Execution time with getValue_c: '
    f'{time.time() - start_time:.2f} seconds ---'
)

# %%
# We now perform the same simulation using Biogeme. The results are
# identical, but the execution time is faster. Indeed, Biogeme
# recycles calculations performed for one expression for the other
# expressions.

# %%
# A dictionary with the requested expression must be provided to Biogeme
simulate = {
    'weight': normalized_weight,
    'Utility PT': V_PT,
    'Utility car': V_CAR,
    'Utility SM': V_SM,
    'Prob. PT': prob_PT,
    'Prob. car': prob_CAR,
    'Prob. SM': prob_SM,
}

# %%
start_time = time.time()
the_biogeme = BIOGEME(database, simulate)
biogeme_simulation = the_biogeme.simulate(results.get_beta_values())

# %%
print(
    f'--- Execution time with Biogeme:    '
    f'{time.time() - start_time:.2f} seconds ---'
)

# %%
# Let's print the two results, to show that they are identical

# %%
# Without Biogeme
print(simulated_values)

# %%
# With Biogeme
print(biogeme_simulation)
