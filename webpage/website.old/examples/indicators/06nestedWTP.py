"""File 06nestedWTP.py

:author: Michel Bierlaire, EPFL
:date: Wed Sep 11 14:01:00 2019

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We calculate and plot willingness to pay.
"""

import matplotlib.pyplot as plt
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.results as res
import biogeme.messaging as msg

from biogeme.expressions import Beta, Derive


logger = msg.bioMessage()
logger.setGeneral()

# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

confidenceInterval = True

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# Normalize the weights
sumWeight = database.data['Weight'].sum()
numberOfRows = database.data.shape[0]
normalizedWeight = Weight * numberOfRows / sumWeight

# Calculate the number of accurences of a value in the database
numberOfMales = database.count('Gender', 1)
print(f'Number of males:   {numberOfMales}')
numberOfFemales = database.count('Gender', 2)
print(f'Number of females: {numberOfFemales}')

# For more complex conditions, we use Pandas
unreportedGender = database.data[(database.data['Gender'] != 1) & \
                                 (database.data['Gender'] != 2)].count()['Gender']
print(f'Unreported gender: {unreportedGender}')


# List of parameters. Their value will be set later.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0, None, None, 0)
BETA_TIME_FULLTIME = Beta('BETA_TIME_FULLTIME', 0, None, None, 0)
BETA_TIME_OTHER = Beta('BETA_TIME_OTHER', 0, None, None, 0)
BETA_DIST_MALE = Beta('BETA_DIST_MALE', 0, None, None, 0)
BETA_DIST_FEMALE = Beta('BETA_DIST_FEMALE', 0, None, None, 0)
BETA_DIST_UNREPORTED = Beta('BETA_DIST_UNREPORTED', 0, None, None, 0)
BETA_COST = Beta('BETA_COST', 0, None, None, 0)

# Define new variables. Must be consistent with estimation results.
TimePT_scaled = TimePT / 200
TimeCar_scaled = TimeCar / 200
MarginalCostPT_scaled = MarginalCostPT / 10
CostCarCHF_scaled = CostCarCHF / 10
distance_km_scaled = distance_km / 5
male = (Gender == 1)
female = (Gender == 2)
unreportedGender = (Gender == -1)
fulltime = (OccupStat == 1)
notfulltime = (OccupStat != 1)

# Definition of utility functions:
V_PT = ASC_PT + BETA_TIME_FULLTIME * TimePT_scaled * fulltime + \
    BETA_TIME_OTHER * TimePT_scaled * notfulltime + \
    BETA_COST * MarginalCostPT_scaled
V_CAR = ASC_CAR + \
    BETA_TIME_FULLTIME * TimeCar_scaled * fulltime + \
    BETA_TIME_OTHER * TimeCar_scaled * notfulltime + \
    BETA_COST * CostCarCHF_scaled
V_SM = ASC_SM + \
    BETA_DIST_MALE * distance_km_scaled * male + \
    BETA_DIST_FEMALE * distance_km_scaled * female + \
    BETA_DIST_UNREPORTED * distance_km_scaled * unreportedGender

# Associate utility functions with the numbering of alternatives
V = {0: V_PT,
     1: V_CAR,
     2: V_SM}

# Definition of the nests:
# 1: nests parameter
# 2: list of alternatives

MU_NOCAR = Beta('MU_NOCAR', 1.0, 1.0, None, 0)

CAR_NEST = 1.0, [1]
NO_CAR_NEST = MU_NOCAR, [0, 2]
nests = CAR_NEST, NO_CAR_NEST

WTP_PT_TIME = Derive(V_PT, 'TimePT') / Derive(V_PT, 'MarginalCostPT')
WTP_CAR_TIME = Derive(V_CAR, 'TimeCar') / Derive(V_CAR, 'CostCarCHF')

simulate = {'weight': normalizedWeight,
            'WTP PT time': WTP_PT_TIME,
            'WTP CAR time': WTP_CAR_TIME}

biogeme = bio.BIOGEME(database, simulate, removeUnusedVariables=False)
biogeme.modelName = '06nestedWTP'

# Read the estimation results from the file.
try:
    results = res.bioResults(pickleFile='01nestedEstimation.pickle')
except FileNotFoundError:
    sys.exit('Run first the script 01nestedEstimation.py in order to generate '
             'the file 01nestedEstimation.pickle.')

# simulatedValues is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulatedValues = biogeme.simulate(results.getBetaValues())

wtpcar = (60 * simulatedValues['WTP CAR time'] * simulatedValues['weight']).mean()

# Calculate confidence intervals
b = results.getBetasForSensitivityAnalysis(biogeme.freeBetaNames, size=100)

# Returns data frame containing, for each simulated value, the left
# and right bounds of the confidence interval calculated by simulation.
left, right = biogeme.confidenceIntervals(b, 0.9)

wtpcar_left = (60 * left['WTP CAR time'] * left['weight']).mean()
wtpcar_right = (60 * right['WTP CAR time'] * right['weight']).mean()
print(f'Average WTP for car: {wtpcar:.3g} '
      f'CI:[{wtpcar_left:.3g}, {wtpcar_right:.3g}]')


# In this specific case, there are only two distinct values in the
# population: for workers and non workers
print('Unique values: ',
      [f'{i:.3g}' for i in 60 * simulatedValues['WTP CAR time'].unique()])

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
    sim = simulatedValues[theFilter]
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
print(type(aFilter))
w, l, r = wtpForSubgroup(aFilter)
print(f'WTP car for females: {w:.3g} CI:[{l:.3g}, {r:.3g}]')

# males
aFilter = database.data['Gender'] == 1
w, l, r = wtpForSubgroup(aFilter)
print(f'WTP car for males: {w:.3g} CI:[{l:.3g}, {r:.3g}]')

# We plot the distribution of WTP in the population. In this case,
# there are only two values

plt.hist(60 * simulatedValues['WTP CAR time'],
         weights=simulatedValues['weight'])
plt.xlabel('WTP (CHF/hour)')
plt.ylabel('Individuals')
plt.show()
