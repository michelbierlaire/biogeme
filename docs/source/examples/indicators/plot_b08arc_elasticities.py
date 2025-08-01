"""

Arc elasticities
================

We use a previously estimated nested logit model and calculate arc elasticities.

Details about this example are available in Section 3 of `Bierlaire (2018)
Calculating indicators with PandasBiogeme
<http://transp-or.epfl.ch/documents/technicalReports/Bier18a.pdf>`_

Michel Bierlaire, EPFL
Sat Jun 28 2025, 20:59:31
"""

import sys

from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.data.optima import normalized_weight, read_data
from biogeme.models import nested
from biogeme.results_processing import EstimationResults

from scenarios import scenario

# %%
# Obtain the specification for the default scenario
# The definition of the scenarios is available in :ref:`scenarios`.
v, nests, _, MarginalCostPT = scenario()

# %%
# Obtain the expression for the choice probability of the public transportation.
prob_pt = nested(v, None, nests, 0)

# %%
# We investigate a scenario where the price for public transportation
# increases by 20%. We extract the corresponding scenario.
v_after, _, _, marginal_cost_pt_after = scenario(factor=1.2)
prob_pt_after = nested(v_after, None, nests, 0)

# %%
# Disaggregate elasticities
direct_elas_pt = (
    (prob_pt_after - prob_pt)
    * MarginalCostPT
    / (prob_pt * (marginal_cost_pt_after - MarginalCostPT))
)

# %%
# Formulas to simulate.
simulate = {
    'weight': normalized_weight,
    'Prob. PT': prob_pt,
    'direct_elas_pt': direct_elas_pt,
}

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, simulate)

# %%
# Read the estimation results from the file
try:
    results = EstimationResults.from_yaml_file(
        filename='saved_results/b02estimation.yaml'
    )
except FileNotFoundError:
    sys.exit(
        'Run first the script b02estimation.py in order to generate '
        'the file b02estimation.yaml.'
    )

# %%
# `simulated_values` is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulated_values = the_biogeme.simulate(results.get_beta_values())
display(simulated_values)

# %%
# We calculate the aggregate elasticities.

# %%
# First, the weighted probabilities.
simulated_values['Weighted prob. PT'] = (
    simulated_values['weight'] * simulated_values['Prob. PT']
)

# %%
# Then the denominator of the aggregate elasticity expression.
denominator_pt = simulated_values['Weighted prob. PT'].sum()

# %%
# And finally the aggregate elasticities themselves.
direct_elas_pt = (
    simulated_values['Weighted prob. PT']
    * simulated_values['direct_elas_pt']
    / denominator_pt
).sum()

print(
    f'Aggregate direct arc elasticity of public transportation wrt cost: '
    f'{direct_elas_pt:.3g}'
)
