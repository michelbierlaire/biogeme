"""File 08arc_elasticities.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 14:52:48 2021

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We calculate disaggregate and aggregate direct arc elasticities.
"""

import sys
import biogeme.biogeme as bio
import biogeme.exceptions as excep
from biogeme import models
import biogeme.results as res
from scenarios import (
    scenario,
    database,
    normalizedWeight,
)

# Obtain the specification for the default scenario
V, nests, _, MarginalCostPT = scenario()

# Obtain the expression for the choice probability of each alternative
prob_PT = models.nested(V, None, nests, 0)

# We investigate a scenario where the price for public transportation
# increases by 20%
V_after, _, _, MarginalCostPT_after = scenario(factor=1.2)
prob_PT_after = models.nested(V_after, None, nests, 0)


# Disaggregate elasticities
direct_elas_pt = (
    (prob_PT_after - prob_PT)
    * MarginalCostPT
    / (prob_PT * (MarginalCostPT_after - MarginalCostPT))
)

simulate = {
    'weight': normalizedWeight,
    'Prob. PT': prob_PT,
    'direct_elas_pt': direct_elas_pt,
}

biogeme = bio.BIOGEME(database, simulate)

# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='02estimation.pickle')
except excep.biogemeError:
    sys.exit(
        'Run first the script 02estimation.py in order to generate '
        'the file 02estimation.pickle.'
    )

# simulated_values is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulated_values = biogeme.simulate(results.getBetaValues())

# We calculate the elasticities
simulated_values['Weighted prob. PT'] = (
    simulated_values['weight'] * simulated_values['Prob. PT']
)

denominator_pt = simulated_values['Weighted prob. PT'].sum()

direct_elas_pt = (
    simulated_values['Weighted prob. PT']
    * simulated_values['direct_elas_pt']
    / denominator_pt
).sum()

print(
    f'Aggregate direct arc elasticity of public transportation wrt cost: '
    f'{direct_elas_pt:.3g}'
)
