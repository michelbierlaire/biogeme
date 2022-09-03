"""File 12panel_flat.py

:author: Michel Bierlaire, EPFL
:date: Mon Feb 14 13:10:32 2022

 Example of a mixture of logit models, using Monte-Carlo integration.
 The datafile is organized as panel data, but a flat version is generated.
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
from biogeme.expressions import (
    Beta,
    Variable,
    Variable,
    bioDraws,
    MonteCarlo,
    log,
    exp,
    bioMultSum,
)

np.random.seed(seed=90267)

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
orig_database = db.Database('swissmetro', df)
exclude = (
    (Variable('PURPOSE') != 1) * (Variable('PURPOSE') != 3)
    + (Variable('CHOICE') == 0)
) > 0
orig_database.remove(exclude)

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

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: adding columns to the orig_database
CAR_AV_SP = orig_database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = orig_database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
_ = orig_database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
_ = orig_database.DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100)
_ = orig_database.DefineVariable('SM_TT_SCALED', SM_TT / 100.0)
_ = orig_database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
_ = orig_database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
_ = orig_database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)


# They are organized as panel data. The variable ID identifies each individual.
orig_database.panel("ID")

# We flatten the database, so that each row corresponds to one individual
flat_df = orig_database.generateFlatPanelDataframe(identical_columns=None)
for i in flat_df.columns:
    print(i)
database = db.Database('swissmetro_flat', flat_df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
# print(database.data.describe())

# Parameters to be estimated
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation
B_TIME = Beta('B_TIME', 0, None, None, 0)

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL_ANTI')

# We do the same for the constants, to address serial correlation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR_S = Beta('ASC_CAR_S', 1, None, None, 0)
ASC_CAR_RND = ASC_CAR + ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL_ANTI')

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_TRAIN_S = Beta('ASC_TRAIN_S', 1, None, None, 0)
ASC_TRAIN_RND = ASC_TRAIN + ASC_TRAIN_S * bioDraws(
    'ASC_TRAIN_RND', 'NORMAL_ANTI'
)

ASC_SM = Beta('ASC_SM', 0, None, None, 1)
ASC_SM_S = Beta('ASC_SM_S', 1, None, None, 0)
ASC_SM_RND = ASC_SM + ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL_ANTI')


# Definition of the utility functions
V1 = [
    ASC_TRAIN_RND
    + B_TIME_RND * Variable(f'{t}_TRAIN_TT_SCALED')
    + B_COST * Variable(f'{t}_TRAIN_COST_SCALED')
    for t in range(1, 10)
]

V2 = [
    ASC_SM_RND
    + B_TIME_RND * Variable(f'{t}_SM_TT_SCALED')
    + B_COST * Variable(f'{t}_SM_COST_SCALED')
    for t in range(1, 10)
]

V3 = [
    ASC_CAR_RND
    + B_TIME_RND * Variable(f'{t}_CAR_TT_SCALED')
    + B_COST * Variable(f'{t}_CAR_CO_SCALED')
    for t in range(1, 10)
]

# Associate utility functions with the numbering of alternatives
V = [{1: V1[t], 2: V2[t], 3: V3[t]} for t in range(9)]

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Conditional to the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel)
obsprob = [
    models.loglogit(V[t], av, Variable(f'{t+1}_CHOICE')) for t in range(9)
]
condprobIndiv = exp(bioMultSum(obsprob))
# Conditional to the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.

# We integrate over the random parameters using Monte-Carlo
logprob = log(MonteCarlo(condprobIndiv))

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
logger.setDetailed()
# logger.setDebug()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob, numberOfDraws=100000)
biogeme.modelName = '12panel_flat'

# Estimate the parameters.
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)
