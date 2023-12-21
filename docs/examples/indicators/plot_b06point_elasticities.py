"""

Direct point elasticities
=========================

We use a previously estimated nested logit model and calculate
 disaggregate and aggregate direct point elasticities.

Details about this example are available in Section 3 of `Bierlaire (2018)
Calculating indicators with PandasBiogeme
<http://transp-or.epfl.ch/documents/technicalReports/Bier18a.pdf>`_

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 21:01:41 2023

"""

import sys
import biogeme.biogeme as bio
from biogeme import models
import biogeme.results as res
import biogeme.exceptions as excep
from biogeme.expressions import Derive
from optima_data import database, normalized_weight

from scenarios import (
    scenario,
    TimePT,
    TimeCar,
    MarginalCostPT,
    CostCarCHF,
    distance_km,
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
# Calculation of the direct elasticities.
# We use the 'Derive' operator to calculate the derivatives.
direct_elas_pt_time = Derive(prob_PT, 'TimePT') * TimePT / prob_PT

direct_elas_pt_cost = Derive(prob_PT, 'MarginalCostPT') * MarginalCostPT / prob_PT

direct_elas_car_time = Derive(prob_CAR, 'TimeCar') * TimeCar / prob_CAR

direct_elas_car_cost = Derive(prob_CAR, 'CostCarCHF') * CostCarCHF / prob_CAR

direct_elas_sm_dist = Derive(prob_SM, 'distance_km') * distance_km / prob_SM

# %%
# Formulas to simulate.
simulate = {
    'weight': normalized_weight,
    'Prob. car': prob_CAR,
    'Prob. public transportation': prob_PT,
    'Prob. slow modes': prob_SM,
    'direct_elas_pt_time': direct_elas_pt_time,
    'direct_elas_pt_cost': direct_elas_pt_cost,
    'direct_elas_car_time': direct_elas_car_time,
    'direct_elas_car_cost': direct_elas_car_cost,
    'direct_elas_sm_dist': direct_elas_sm_dist,
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
# `simulated_values` is a Pandas dataframe with the same number of rows as
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
simulated_values['Weighted prob. SM'] = (
    simulated_values['weight'] * simulated_values['Prob. slow modes']
)

# %%
# Then the denominators of the aggregate elasticity expressions.
denominator_car = simulated_values['Weighted prob. car'].sum()
denominator_pt = simulated_values['Weighted prob. PT'].sum()
denominator_sm = simulated_values['Weighted prob. SM'].sum()

# %%
# And finally the aggregate elasticities themselves.

# %%
# Elasticity of car with respect to time.
direct_elas_term_car_time = (
    simulated_values['Weighted prob. car']
    * simulated_values['direct_elas_car_time']
    / denominator_car
).sum()

print(
    f'Aggregate direct point elasticity of car wrt time: '
    f'{direct_elas_term_car_time:.3g}'
)

# %%
# Elasticity of car with respect to cost.
direct_elas_term_car_cost = (
    simulated_values['Weighted prob. car']
    * simulated_values['direct_elas_car_cost']
    / denominator_car
).sum()
print(
    f'Aggregate direct point elasticity of car wrt cost: '
    f'{direct_elas_term_car_cost:.3g}'
)

# %%
# Elasticity of public transportation with respect to time.
direct_elas_term_pt_time = (
    simulated_values['Weighted prob. PT']
    * simulated_values['direct_elas_pt_time']
    / denominator_pt
).sum()
print(
    f'Aggregate direct point elasticity of PT wrt time: '
    f'{direct_elas_term_pt_time:.3g}'
)

# %%
# Elasticity of public transportation with respect to cost.
direct_elas_term_pt_cost = (
    simulated_values['Weighted prob. PT']
    * simulated_values['direct_elas_pt_cost']
    / denominator_pt
).sum()
print(
    f'Aggregate direct point elasticity of PT wrt cost: '
    f'{direct_elas_term_pt_cost:.3g}'
)

# %%
# Elasticity of slow modes with respect to distance.
direct_elas_term_sm_dist = (
    simulated_values['Weighted prob. SM']
    * simulated_values['direct_elas_sm_dist']
    / denominator_sm
).sum()
print(
    f'Aggregate direct point elasticity of SM wrt distance: '
    f'{direct_elas_term_sm_dist:.3g}'
)
