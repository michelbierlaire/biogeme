"""File 21probit.py

:author: Michel Bierlaire, EPFL
:date: Mon Sep  9 10:14:57 2019

 Example of a binary probit model.
 Two alternatives: Train and Car
 SP data
"""

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.messaging as msg
from biogeme.expressions import Beta, DefineVariable, bioNormalCdf, Elem, log

# Read the data
df = pd.read_csv('swissmetro.dat', '\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Removing some observations can be done directly using pandas.
#remove = (((database.data.PURPOSE != 1) &
#           (database.data.PURPOSE != 3)) |
#          (database.data.CHOICE == 0))
#database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility. As we
# estimate a binary model, we remove observations where Swissmetro was
# chosen (CHOICE == 2). We also remove observations where one of the
# two alternatives is not available.

CAR_AV_SP = DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0), database)
TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0), database)
exclude = (TRAIN_AV_SP == 0) + (CAR_AV_SP == 0) + (CHOICE == 2) + \
    ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables by adding columns to the database.
# This is recommended for estimation. And not recommended for simulation.
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0, database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100, database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0, database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100, database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100, database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100, database)

# Definition of the utility functions
# We estimate a binary probit model. There are only two alternatives.
V1 = B_TIME * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V3 = ASC_CAR + \
     B_TIME * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED

# Associate choice probability with the numbering of alternatives
P = {1: bioNormalCdf(V1 - V3),
     3: bioNormalCdf(V3 - V1)}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = log(Elem(P, CHOICE))

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
#logger.setWarning()
#logger.setGeneral()
#logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '21probit'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)
