"""

Simulation of a choice model
============================

We use an estimated model to perform various simulations.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 16:56:26
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
v, nests, _, _ = scenario()

v_pt = v[0]
v_car = v[1]
v_sm = v[2]

# %%
# Obtain the expression for the choice probability of each alternative.
prob_pt = nested(v, None, nests, 0)
prob_car = nested(v, None, nests, 1)
prob_sm = nested(v, None, nests, 2)

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
        expression=normalized_weight,
        betas=results.get_beta_values(),
        database=database,
        numerically_safe=False,
        use_jit=True,
    ),
    'Utility PT': get_value_c(
        expression=v_pt,
        betas=results.get_beta_values(),
        database=database,
        numerically_safe=False,
        use_jit=True,
    ),
    'Utility car': get_value_c(
        expression=v_car,
        betas=results.get_beta_values(),
        database=database,
        numerically_safe=False,
        use_jit=True,
    ),
    'Utility SM': get_value_c(
        expression=v_sm,
        betas=results.get_beta_values(),
        database=database,
        numerically_safe=False,
        use_jit=True,
    ),
    'Prob. PT': get_value_c(
        expression=prob_pt,
        betas=results.get_beta_values(),
        database=database,
        numerically_safe=False,
        use_jit=True,
    ),
    'Prob. car': get_value_c(
        expression=prob_car,
        betas=results.get_beta_values(),
        database=database,
        numerically_safe=False,
        use_jit=True,
    ),
    'Prob. SM': get_value_c(
        expression=prob_sm,
        betas=results.get_beta_values(),
        database=database,
        numerically_safe=False,
        use_jit=True,
    ),
}

simulated_values = pd.DataFrame.from_dict(simulate_formulas)
end_time = time.time()

# %%
print(
    f'--- Execution time without Biogeme:    '
    f'{end_time - start_time:.2f} seconds ---'
)

# %%
# We now perform the same simulation using Biogeme. The results are
# identical, but the syntax is simpler and the execution time is a little bit faster. Indeed, Biogeme
# recycles calculations performed for one expression for the other
# expressions.

# %%
# A dictionary with the requested expression must be provided to Biogeme
simulate = {
    'weight': normalized_weight,
    'Utility PT': v_pt,
    'Utility car': v_car,
    'Utility SM': v_sm,
    'Prob. PT': prob_pt,
    'Prob. car': prob_car,
    'Prob. SM': prob_sm,
}

# %%
start_time = time.time()
the_biogeme = BIOGEME(database, simulate)
the_betas = results.get_beta_values()
biogeme_simulation = the_biogeme.simulate(results.get_beta_values())
end_time = time.time()

# %%
print(
    f'--- Execution time with Biogeme:       '
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
