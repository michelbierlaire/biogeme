"""File 05nestedElasticities.py

:author: Michel Bierlaire, EPFL
:date: Wed Sep 11 13:41:33 2019

 We use a previously estimated nested logit model.
 Three alternatives: public transporation, car and slow modes.
 RP data.
 We calculate disaggregate and aggregate direct arc elasticities.
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

# For more complex conditions, using directly Pandas
unreportedGender = database.data[(database.data['Gender'] != 1)
                                 & (database.data['Gender'] != 2)].count()['Gender']
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

# We investigate a scenario where the distance increases by one kilometer.
delta_dist = 1.0
distance_km_scaled_after = (distance_km + delta_dist) / 5

# Utility of the slow mode whem the distance increases by 1 kilometer.
V_SM_after = ASC_SM + \
    BETA_DIST_MALE * distance_km_scaled_after * male + \
    BETA_DIST_FEMALE * distance_km_scaled_after * female + \
    BETA_DIST_UNREPORTED * distance_km_scaled_after * unreportedGender

# Associate utility functions with the numbering of alternatives
V_after = {0: V_PT,
           1: V_CAR,
           2: V_SM_after}

# Definition of the nests:
# 1: nests parameter
# 2: list of alternatives

prob_sm_after = models.nested(V_after, None, nests, 2)

# Disaggregate elasticities
direct_elas_sm_dist = (prob_sm_after - prob_sm) * \
    distance_km / (prob_sm * delta_dist)

simulate = {'weight': normalizedWeight,
            'Prob. slow modes': prob_sm,
            'direct_elas_sm_dist': direct_elas_sm_dist}

biogeme = bio.BIOGEME(database, simulate)
biogeme.modelName = '05nestedElasticities'

# Read the estimation results from the file
try:
    results = res.bioResults(pickleFile='01nestedEstimation.pickle')
except FileNotFoundError:
    sys.exit('Run first the script 01nestedEstimation.py in order to generate '
             f'the file 01nestedEstimation.pickle.')

# simulatedValues is a Panda dataframe with the same number of rows as
# the database, and as many columns as formulas to simulate.
simulatedValues = biogeme.simulate(results.getBetaValues())

# We calculate the elasticities
simulatedValues['Weighted prob. slow modes'] = \
    simulatedValues['weight'] * simulatedValues['Prob. slow modes']

denominator_sm = simulatedValues['Weighted prob. slow modes'].sum()

direct_elas_sm_dist = (simulatedValues['Weighted prob. slow modes']
                       * simulatedValues['direct_elas_sm_dist'] /
                       denominator_sm).sum()

print(f'Aggregate direct arc elasticity of slow modes wrt distance: '
      f'{direct_elas_sm_dist:.3g}')
