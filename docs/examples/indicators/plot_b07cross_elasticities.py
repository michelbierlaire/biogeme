"""

Cross point elasticities
========================

We use a previously estimated nested logit model and calculate
disaggregate and aggregate cross point elasticities.

Details about this example are available in Section 3 of `Bierlaire (2018)
Calculating indicators with PandasBiogeme
<http://transp-or.epfl.ch/documents/technicalReports/Bier18a.pdf>`_

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 21:00:58 2023

"""

import sys
import biogeme.biogeme as bio
from biogeme import models
import biogeme.exceptions as excep
import biogeme.results as res
from biogeme.expressions import Derive
from optima_data import database, normalized_weight

from scenarios import (
    scenario,
    TimePT,
    TimeCar,
    MarginalCostPT,
    CostCarCHF,
)


# %%
# Obtain the specification for the default scenario
# The definition of the scenarios is available in :ref:`scenarios`.
V, nests, _, _ = scenario()

# %%
# Obtain the expression for the choice probability of each alternative.
prob_PT = models.nested(V, None, nests, 0)
prob_CAR = models.nested(V, None, nests, 1)
prob_SM = models.nested(V, None, nests, 2)

# %%
# The choice model is a nested logit.
prob_PT = models.nested(V, None, nests, 0)
prob_CAR = models.nested(V, None, nests, 1)
prob_SM = models.nested(V, None, nests, 2)

# %%
# Calculation of the cross elasticities.
# We use the 'Derive' operator to calculate the derivatives.
cross_elas_pt_time = Derive(prob_PT, 'TimeCar') * TimeCar / prob_PT
cross_elas_pt_cost = Derive(prob_PT, 'CostCarCHF') * CostCarCHF / prob_PT
cross_elas_car_time = Derive(prob_CAR, 'TimePT') * TimePT / prob_CAR
cross_elas_car_cost = Derive(prob_CAR, 'MarginalCostPT') * MarginalCostPT / prob_CAR

# %%
# Formulas to simulate.
simulate = {
    'weight': normalized_weight,
    'Prob. car': prob_CAR,
    'Prob. public transportation': prob_PT,
    'Prob. slow modes': prob_SM,
    'cross_elas_pt_time': cross_elas_pt_time,
    'cross_elas_pt_cost': cross_elas_pt_cost,
    'cross_elas_car_time': cross_elas_car_time,
    'cross_elas_car_cost': cross_elas_car_cost,
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
simulated_values['Weighted prob. car'] = (
    simulated_values['weight'] * simulated_values['Prob. car']
)
simulated_values['Weighted prob. PT'] = (
    simulated_values['weight'] * simulated_values['Prob. public transportation']
)

# %%
# Then the denominators of the aggregate elasticity expressions.
denominator_car = simulated_values['Weighted prob. car'].sum()
denominator_pt = simulated_values['Weighted prob. PT'].sum()

# %%
# And finally the aggregate elasticities themselves.

# %%
# Elasticity of car with respect to public transportation travel time.
cross_elas_term_car_time = (
    simulated_values['Weighted prob. car']
    * simulated_values['cross_elas_car_time']
    / denominator_car
).sum()
print(
    f'Aggregate cross elasticity of car wrt PT time: ' f'{cross_elas_term_car_time:.3g}'
)

# %%
# Elasticity of car with respect to public transportation travel cost.
cross_elas_term_car_cost = (
    simulated_values['Weighted prob. car']
    * simulated_values['cross_elas_car_cost']
    / denominator_car
).sum()
print(
    f'Aggregate cross elasticity of car wrt PT cost: ' f'{cross_elas_term_car_cost:.3g}'
)

# %%
# Elasticity of public transportatiom with respect to car travel time.
cross_elas_term_pt_time = (
    simulated_values['Weighted prob. PT']
    * simulated_values['cross_elas_pt_time']
    / denominator_pt
).sum()
print(
    f'Aggregate cross elasticity of PT wrt car time: ' f'{cross_elas_term_pt_time:.3g}'
)

# %%
# Elasticity of public transportatiom with respect to car travel cost.
cross_elas_term_pt_cost = (
    simulated_values['Weighted prob. PT']
    * simulated_values['cross_elas_pt_cost']
    / denominator_pt
).sum()
print(
    f'Aggregate cross direct elasticity of PT wrt car cost: '
    f'{cross_elas_term_pt_cost:.3g}'
)
