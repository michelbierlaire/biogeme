"""File 07cross_elasticities.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 14:45:34 2021

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We calculate disaggregate and aggregate cross point elasticities.
"""

import sys
import biogeme.biogeme as bio
from biogeme import models
import biogeme.exceptions as excep
import biogeme.results as res
from biogeme.expressions import Derive
from scenarios import (
    scenario,
    database,
    normalizedWeight,
    TimePT,
    TimeCar,
    MarginalCostPT,
    CostCarCHF,
)


# Obtain the specification for the default scenario
V, nests, _, _ = scenario()

# Obtain the expression for the choice probability of each alternative
prob_PT = models.nested(V, None, nests, 0)
prob_CAR = models.nested(V, None, nests, 1)
prob_SM = models.nested(V, None, nests, 2)

# The choice model is a nested logit
prob_PT = models.nested(V, None, nests, 0)
prob_CAR = models.nested(V, None, nests, 1)
prob_SM = models.nested(V, None, nests, 2)

# Calculation of the cross elasticities.
# We use the 'Derive' operator to calculate the derivatives.
cross_elas_pt_time = Derive(prob_PT, 'TimeCar') * TimeCar / prob_PT
cross_elas_pt_cost = Derive(prob_PT, 'CostCarCHF') * CostCarCHF / prob_PT
cross_elas_car_time = Derive(prob_CAR, 'TimePT') * TimePT / prob_CAR
cross_elas_car_cost = (
    Derive(prob_CAR, 'MarginalCostPT') * MarginalCostPT / prob_CAR
)

simulate = {
    'weight': normalizedWeight,
    'Prob. car': prob_CAR,
    'Prob. public transportation': prob_PT,
    'Prob. slow modes': prob_SM,
    'cross_elas_pt_time': cross_elas_pt_time,
    'cross_elas_pt_cost': cross_elas_pt_cost,
    'cross_elas_car_time': cross_elas_car_time,
    'cross_elas_car_cost': cross_elas_car_cost,
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
simulated_values['Weighted prob. car'] = (
    simulated_values['weight'] * simulated_values['Prob. car']
)
simulated_values['Weighted prob. PT'] = (
    simulated_values['weight']
    * simulated_values['Prob. public transportation']
)

denominator_car = simulated_values['Weighted prob. car'].sum()
denominator_pt = simulated_values['Weighted prob. PT'].sum()

cross_elas_term_car_time = (
    simulated_values['Weighted prob. car']
    * simulated_values['cross_elas_car_time']
    / denominator_car
).sum()
print(
    f'Aggregate cross elasticity of car wrt time: '
    f'{cross_elas_term_car_time:.3g}'
)

cross_elas_term_car_cost = (
    simulated_values['Weighted prob. car']
    * simulated_values['cross_elas_car_cost']
    / denominator_car
).sum()
print(
    f'Aggregate cross elasticity of car wrt cost: '
    f'{cross_elas_term_car_cost:.3g}'
)

cross_elas_term_pt_time = (
    simulated_values['Weighted prob. PT']
    * simulated_values['cross_elas_pt_time']
    / denominator_pt
).sum()
print(
    f'Aggregate cross elasticity of PT wrt car time: '
    f'{cross_elas_term_pt_time:.3g}'
)

cross_elas_term_pt_cost = (
    simulated_values['Weighted prob. PT']
    * simulated_values['cross_elas_pt_cost']
    / denominator_pt
).sum()
print(
    f'Aggregate cross direct elasticity of PT wrt car cost: '
    f'{cross_elas_term_pt_cost:.3g}'
)
