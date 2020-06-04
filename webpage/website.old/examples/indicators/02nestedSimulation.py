"""File 02nestedSimulation.py

:author: Michel Bierlaire, EPFL
:date: Wed Sep 11 10:23:51 2019

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We simulate market shares and revenues.
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.results as res
from biogeme.expressions import Beta


# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

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

# The choice model is a nested logit
prob_pt = models.nested(V, None, nests, 0)
prob_car = models.nested(V, None, nests, 1)
prob_sm = models.nested(V, None, nests, 2)

simulate = {'weight': normalizedWeight,
            'Prob. car': prob_car,
            'Prob. public transp.': prob_pt,
            'Prob. slow modes': prob_sm,
            'Revenue public transportation': prob_pt * MarginalCostPT}

biogeme = bio.BIOGEME(database, simulate)
biogeme.modelName = '02nestedSimulation'

# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='01nestedEstimation.pickle')
except FileNotFoundError:
    sys.exit('Run first the script 01nestedEstimation.py in order to generate the '
             'file 01nestedEstimation.pickle.')

# simulatedValues is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulatedValues = biogeme.simulate(results.getBetaValues())

# Calculate confidence intervals
betas = biogeme.freeBetaNames
b = results.getBetasForSensitivityAnalysis(betas, size=100)

# Returns data frame containing, for each simulated value, the left
# and right bounds of the confidence interval calculated by
# simulation.
left, right = biogeme.confidenceIntervals(b, 0.9)

# We calculate now the market shares and their confidence intervals
simulatedValues['Weighted prob. car'] = \
    simulatedValues['weight'] * simulatedValues['Prob. car']
left['Weighted prob. car'] = left['weight'] * left['Prob. car']
right['Weighted prob. car'] = right['weight'] * right['Prob. car']

marketShare_car = simulatedValues['Weighted prob. car'].mean()

marketShare_car_left = left['Weighted prob. car'].mean()

marketShare_car_right = right['Weighted prob. car'].mean()

print(f'Market share for car: {100*marketShare_car:.1f}% '
      f'[{100*marketShare_car_left:.1f}%, {100*marketShare_car_right:.1f}%]')

simulatedValues['Weighted prob. PT'] = simulatedValues['weight'] * \
    simulatedValues['Prob. public transp.']

marketShare_pt = simulatedValues['Weighted prob. PT'].mean()

marketShare_pt_left = (left['Prob. public transp.'] * left['weight']).mean()

marketShare_pt_right = (right['Prob. public transp.'] * right['weight']).mean()

print(f'Market share for PT: {100*marketShare_pt:.1f}% '
      f'[{100*marketShare_pt_left:.1f}%, {100*marketShare_pt_right:.1f}%]')

marketShare_sm = (simulatedValues['Prob. slow modes'] *
                  simulatedValues['weight']).mean()

marketShare_sm_left = (left['Prob. slow modes'] * left['weight']).mean()

marketShare_sm_right = (right['Prob. slow modes'] * right['weight']).mean()

print(f'Market share for slow modes: {100*marketShare_sm:.1f}% '
      f'[{100*marketShare_sm_left:.1f}%, {100*marketShare_sm_right:.1f}%]')

# and, similarly, the revenues
revenues_pt = (simulatedValues['Revenue public transportation'] *
               simulatedValues['weight']).sum()
revenues_pt_left = (left['Revenue public transportation'] *
                    left['weight']).sum()
revenues_pt_right = (right['Revenue public transportation'] *
                     right['weight']).sum()
print(f'Revenues for PT: {revenues_pt:.3f} [{revenues_pt_left:.3f}, '
      f'{revenues_pt_right:.3f}]')
