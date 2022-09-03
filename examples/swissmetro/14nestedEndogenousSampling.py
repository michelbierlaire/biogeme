"""File 14nestedEndogenousSampling.py

:author: Michel Bierlaire, EPFL
:date: Sun Sep  8 19:26:25 2019

 Example of a nested logit model, with the corrections for endogenous sampling.
 Three alternatives: Train, Car and Swissmetro
 Train and car are in the same nest.
 SP data
"""

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
from biogeme.expressions import Beta, Variable

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
# print(database.data.describe())

PURPOSE = Variable('PURPOSE')
CHOICE = Variable('CHOICE')
GA = Variable('GA')
TRAIN_CO = Variable('TRAIN_CO')
CAR_AV = Variable('CAR_AV')
SP = Variable('SP')
TRAIN_AV = Variable('TRAIN_AV')
TRAIN_TT = Variable('TRAIN_TT')
SM_TT = Variable('SM_TT')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')
SM_CO = Variable('SM_CO')
SM_AV = Variable('SM_AV')

# Removing some observations can be done directly using pandas.
# remove = (((database.data.PURPOSE != 1) &
#           (database.data.PURPOSE != 3)) |
#          (database.data.CHOICE == 0))
# database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
MU = Beta('MU', 1, 1, 10, 0)

# In this example, we assume that the three modes exist, and that the
# sampling protocal is choice-based. The probability that a respondent
# belongs to the sample is R_i.
R_TRAIN = 4.42e-2
R_SM = 3.36e-3
R_CAR = 7.5e-3

# The correction terms are the log of these quantities
correction = {1: np.log(R_TRAIN), 2: np.log(R_SM), 3: np.log(R_CAR)}

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
TRAIN_COST_SCALED = database.DefineVariable(
    'TRAIN_COST_SCALED', TRAIN_COST / 100
)
SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', SM_TT / 100.0)
SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests:
# 1: nests parameter
# 2: list of alternatives
existing = MU, [1, 3]
future = 1.0, [2]
nests = existing, future

# The choice model is a nested logit, with corrections for endogenous sampling
# We first obtain the expression of the Gi function for nested logit.
Gi = models.getMevForNested(V, av, nests)

# Then we calculate the MEV log probability, accounting for the correction.
logprob = models.logmev_endogenousSampling(V, Gi, av, correction, CHOICE)

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '14nestedEndogenousSampling'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)
