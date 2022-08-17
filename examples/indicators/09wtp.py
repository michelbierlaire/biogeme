"""File 09wtp.py

:author: Michel Bierlaire, EPFL
:date: Sun Oct 31 15:11:28 2021

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We calculate and plot willingness to pay.
"""

import sys
import matplotlib.pyplot as plt
import biogeme.biogeme as bio
import biogeme.exceptions as excep
import biogeme.results as res

from biogeme.expressions import Derive
from scenarios import (
    scenario,
    database,
    normalizedWeight,
)

V, _, _, _ = scenario()

V_PT = V[0]
V_CAR = V[1]

WTP_PT_TIME = Derive(V_PT, 'TimePT') / Derive(V_PT, 'MarginalCostPT')
WTP_CAR_TIME = Derive(V_CAR, 'TimeCar') / Derive(V_CAR, 'CostCarCHF')

simulate = {
    'weight': normalizedWeight,
    'WTP PT time': WTP_PT_TIME,
    'WTP CAR time': WTP_CAR_TIME,
}

biogeme = bio.BIOGEME(database, simulate)

# Read the estimation results from the file.
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

wtpcar = (
    60 * simulated_values['WTP CAR time'] * simulated_values['weight']
).mean()

# Calculate confidence intervals
b = results.getBetasForSensitivityAnalysis(biogeme.freeBetaNames())

# Returns data frame containing, for each simulated value, the left
# and right bounds of the confidence interval calculated by simulation.
left, right = biogeme.confidenceIntervals(b, 0.9)

wtpcar_left = (60 * left['WTP CAR time'] * left['weight']).mean()
wtpcar_right = (60 * right['WTP CAR time'] * right['weight']).mean()
print(
    f'Average WTP for car: {wtpcar:.3g} '
    f'CI:[{wtpcar_left:.3g}, {wtpcar_right:.3g}]'
)


# In this specific case, there are only two distinct values in the
# population: for workers and non workers
print(
    'Unique values:      ',
    [f'{i:.3g}' for i in 60 * simulated_values['WTP CAR time'].unique()],
)


def wtpForSubgroup(theFilter):
    """
    Check the value for groups of the population. Define a function that
    works for any filter to avoid repeating code.

    :param theFilter: pandas filter
    :type theFilter: numpy.Series(bool)

    :return: willingness-to-pay for car and confidence interval
    :rtype: tuple(float, float, float)
    """
    size = theFilter.sum()
    sim = simulated_values[theFilter]
    totalWeight = sim['weight'].sum()
    weight = sim['weight'] * size / totalWeight
    _wtpcar = (60 * sim['WTP CAR time'] * weight).mean()
    _wtpcar_left = (60 * left[theFilter]['WTP CAR time'] * weight).mean()
    _wtpcar_right = (60 * right[theFilter]['WTP CAR time'] * weight).mean()
    return _wtpcar, _wtpcar_left, _wtpcar_right


# full time workers.
aFilter = database.data['OccupStat'] == 1
w, l, r = wtpForSubgroup(aFilter)
print(f'WTP car for workers: {w:.3g} CI:[{l:.3g}, {r:.3g}]')

# females
aFilter = database.data['Gender'] == 2
w, l, r = wtpForSubgroup(aFilter)
print(f'WTP car for females: {w:.3g} CI:[{l:.3g}, {r:.3g}]')

# males
aFilter = database.data['Gender'] == 1
w, l, r = wtpForSubgroup(aFilter)
print(f'WTP car for males  : {w:.3g} CI:[{l:.3g}, {r:.3g}]')

# We plot the distribution of WTP in the population. In this case,
# there are only two values

plt.hist(
    60 * simulated_values['WTP CAR time'],
    weights=simulated_values['weight'],
)
plt.xlabel('WTP (CHF/hour)')
plt.ylabel('Individuals')
plt.show()
