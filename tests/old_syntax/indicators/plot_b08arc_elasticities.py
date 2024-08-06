"""

Arc elasticities
================

We use a previously estimated nested logit model and calculate arc elasticities.

Details about this example are available in Section 3 of `Bierlaire (2018)
Calculating indicators with PandasBiogeme
<http://transp-or.epfl.ch/documents/technicalReports/Bier18a.pdf>`_

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 21:00:10 2023

"""

import sys
import biogeme.biogeme as bio
import biogeme.exceptions as excep
from biogeme import models
import biogeme.results as res
from optima_data import database, normalized_weight
from scenarios import scenario

# %%
# Obtain the specification for the default scenario
# The definition of the scenarios is available in :ref:`scenarios`.
V, nests, _, MarginalCostPT = scenario()

# %%
# Obtain the expression for the choice probability of the public transportation.
prob_PT = models.nested(V, None, nests, 0)

# %%
# We investigate a scenario where the price for public transportation
# increases by 20%. We extract the corresponding scenario.
V_after, _, _, MarginalCostPT_after = scenario(factor=1.2)
prob_PT_after = models.nested(V_after, None, nests, 0)

# %%
# Disaggregate elasticities
direct_elas_pt = (
    (prob_PT_after - prob_PT)
    * MarginalCostPT
    / (prob_PT * (MarginalCostPT_after - MarginalCostPT))
)

# %%
# Formulas to simulate.
simulate = {
    'weight': normalized_weight,
    'Prob. PT': prob_PT,
    'direct_elas_pt': direct_elas_pt,
}

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, simulate)

# %%
# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='saved_results/b02estimation.pickle')
except excep.BiogemeError:
    sys.exit(
        'Run first the script b02estimation.py in order to generate '
        'the file b02estimation.pickle.'
    )

# %%
# `simulated_values` is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulated_values = the_biogeme.simulate(results.getBetaValues())
simulated_values

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
