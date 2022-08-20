"""File 04market_shares.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 10:14:43 2021

We use an estimated model to calcualte market shares.
"""
import sys
from biogeme import models
import biogeme.biogeme as bio
import biogeme.exceptions as excep
import biogeme.results as res
from scenarios import scenario, database, normalizedWeight

# Obtain the specification for the default scenario
V, nests, _, _ = scenario()

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

# We now simulate the choice probabilities and the weight

simulate = {
    'weight': normalizedWeight,
    'Prob. PT': prob_PT,
    'Prob. car': prob_CAR,
    'Prob. SM': prob_SM,
}

biogeme = bio.BIOGEME(database, simulate)
simulated_values = biogeme.simulate(results.getBetaValues())

# We also calculate confidence intervals for the calculated quantities

betas = biogeme.freeBetaNames()
b = results.getBetasForSensitivityAnalysis(betas)
left, right = biogeme.confidenceIntervals(b, 0.9)


# Market shares are calculated using the weighted mean of the
# individual probabilities

# Alternative car
simulated_values['Weighted prob. car'] = (
    simulated_values['weight'] * simulated_values['Prob. car']
)
left['Weighted prob. car'] = left['weight'] * left['Prob. car']
right['Weighted prob. car'] = right['weight'] * right['Prob. car']

marketShare_car = simulated_values['Weighted prob. car'].mean()
marketShare_car_left = left['Weighted prob. car'].mean()
marketShare_car_right = right['Weighted prob. car'].mean()

# Alternative public transportation
simulated_values['Weighted prob. PT'] = (
    simulated_values['weight'] * simulated_values['Prob. PT']
)
left['Weighted prob. PT'] = left['weight'] * left['Prob. PT']
right['Weighted prob. PT'] = right['weight'] * right['Prob. PT']

marketShare_PT = simulated_values['Weighted prob. PT'].mean()
marketShare_PT_left = left['Weighted prob. PT'].mean()
marketShare_PT_right = right['Weighted prob. PT'].mean()

# Alternative slow modes
simulated_values['Weighted prob. SM'] = (
    simulated_values['weight'] * simulated_values['Prob. SM']
)
left['Weighted prob. SM'] = left['weight'] * left['Prob. SM']
right['Weighted prob. SM'] = right['weight'] * right['Prob. SM']

marketShare_SM = simulated_values['Weighted prob. SM'].mean()
marketShare_SM_left = left['Weighted prob. SM'].mean()
marketShare_SM_right = right['Weighted prob. SM'].mean()

# Reporting
print(
    f'Market share for car: {100*marketShare_car:.1f}% '
    f'[{100*marketShare_car_left:.1f}%, '
    f'{100*marketShare_car_right:.1f}%]'
)
print(
    f'Market share for PT:  {100*marketShare_PT:.1f}% '
    f'[{100*marketShare_PT_left:.1f}%, '
    f'{100*marketShare_PT_right:.1f}%]'
)
print(
    f'Market share for SM:   {100*marketShare_SM:.1f}% '
    f'[{100*marketShare_SM_left:.1f}%, '
    f'{100*marketShare_SM_right:.1f}%]'
)
