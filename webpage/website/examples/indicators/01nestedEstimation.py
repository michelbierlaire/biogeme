"""File 01nestedEstimation.py

:author: Michel Bierlaire, EPFL
:date: Wed Sep 11 09:59:55 2019

 Estimation of a nested logit model, that will be used for simuation.
 Three alternatives: public transporation, car and slow modes.
 RP data.
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta

# Read the data
df = pd.read_csv('optima.dat', sep='\t')
database = db.Database('optima', df)

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Exclude observations such that the chosen alternative is -1
database.remove(Choice == -1.0)

# List of parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_PT = Beta('ASC_PT', 0, None, None, 1)
ASC_SM = Beta('ASC_SM', 0, None, None, 0)
BETA_TIME_FULLTIME = Beta('BETA_TIME_FULLTIME', 0, None, None, 0)
BETA_TIME_OTHER = Beta('BETA_TIME_OTHER', 0, None, None, 0)
BETA_DIST_MALE = Beta('BETA_DIST_MALE', 0, None, None, 0)
BETA_DIST_FEMALE = Beta('BETA_DIST_FEMALE', 0, None, None, 0)
BETA_DIST_UNREPORTED = Beta('BETA_DIST_UNREPORTED', 0, None, None, 0)
BETA_COST = Beta('BETA_COST', 0, None, None, 0)

# Definition of variables:
# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0.
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

# The choice model is a nested logit, with availability conditions
logprob = models.lognested(V, None, nests, Choice)

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '01nestedEstimation'

# Estimate the parameters. Calculate also the standard errors using
# bootstrapping.
results = biogeme.estimate(bootstrap=100)

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
