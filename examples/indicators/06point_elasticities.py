"""File 06_point_elesticities.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 14:12:44 2021

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We calculate disaggregate and aggregate direct point elasticities.
"""

import sys
import biogeme.biogeme as bio
from biogeme import models
import biogeme.results as res
import biogeme.exceptions as excep
from biogeme.expressions import Derive
from scenarios import (
    scenario,
    database,
    normalizedWeight,
    TimePT,
    TimeCar,
    MarginalCostPT,
    CostCarCHF,
    distance_km,
)


# Obtain the specification for the default scenario
V, nests, _, _ = scenario()

# Obtain the expression for the choice probability of each alternative
prob_PT = models.nested(V, None, nests, 0)
prob_CAR = models.nested(V, None, nests, 1)
prob_SM = models.nested(V, None, nests, 2)

# Calculation of the direct elasticities.
# We use the 'Derive' operator to calculate the derivatives.

direct_elas_pt_time = Derive(prob_PT, 'TimePT') * TimePT / prob_PT

direct_elas_pt_cost = (
    Derive(prob_PT, 'MarginalCostPT') * MarginalCostPT / prob_PT
)

direct_elas_car_time = Derive(prob_CAR, 'TimeCar') * TimeCar / prob_CAR

direct_elas_car_cost = Derive(prob_CAR, 'CostCarCHF') * CostCarCHF / prob_CAR

direct_elas_sm_dist = Derive(prob_SM, 'distance_km') * distance_km / prob_SM

# Simulate the formulas
simulate = {
    'weight': normalizedWeight,
    'Prob. car': prob_CAR,
    'Prob. public transportation': prob_PT,
    'Prob. slow modes': prob_SM,
    'direct_elas_pt_time': direct_elas_pt_time,
    'direct_elas_pt_cost': direct_elas_pt_cost,
    'direct_elas_car_time': direct_elas_car_time,
    'direct_elas_car_cost': direct_elas_car_cost,
    'direct_elas_sm_dist': direct_elas_sm_dist,
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

# simulatedValues is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulated_values = biogeme.simulate(results.getBetaValues())

# We calculate the aggregate elasticities

# First, the weighted probabilities
simulated_values['Weighted prob. car'] = (
    simulated_values['weight'] * simulated_values['Prob. car']
)
simulated_values['Weighted prob. PT'] = (
    simulated_values['weight']
    * simulated_values['Prob. public transportation']
)
simulated_values['Weighted prob. SM'] = (
    simulated_values['weight'] * simulated_values['Prob. slow modes']
)

# Then the denominator of the aggregate elasticity expression.
denominator_car = simulated_values['Weighted prob. car'].sum()
denominator_pt = simulated_values['Weighted prob. PT'].sum()
denominator_sm = simulated_values['Weighted prob. SM'].sum()

# And finally the aggregate elasticities themselves.
direct_elas_term_car_time = (
    simulated_values['Weighted prob. car']
    * simulated_values['direct_elas_car_time']
    / denominator_car
).sum()
print(
    f'Aggregate direct point elasticity of car wrt time: '
    f'{direct_elas_term_car_time:.3g}'
)

direct_elas_term_car_cost = (
    simulated_values['Weighted prob. car']
    * simulated_values['direct_elas_car_cost']
    / denominator_car
).sum()
print(
    f'Aggregate direct point elasticity of car wrt cost: '
    f'{direct_elas_term_car_cost:.3g}'
)

direct_elas_term_pt_time = (
    simulated_values['Weighted prob. PT']
    * simulated_values['direct_elas_pt_time']
    / denominator_pt
).sum()
print(
    f'Aggregate direct point elasticity of PT wrt time: '
    f'{direct_elas_term_pt_time:.3g}'
)

direct_elas_term_pt_cost = (
    simulated_values['Weighted prob. PT']
    * simulated_values['direct_elas_pt_cost']
    / denominator_pt
).sum()
print(
    f'Aggregate direct point elasticity of PT wrt cost: '
    f'{direct_elas_term_pt_cost:.3g}'
)

direct_elas_term_sm_dist = (
    simulated_values['Weighted prob. SM']
    * simulated_values['direct_elas_sm_dist']
    / denominator_sm
).sum()
print(
    f'Aggregate direct point elasticity of SM wrt distance: '
    f'{direct_elas_term_sm_dist:.3g}'
)
