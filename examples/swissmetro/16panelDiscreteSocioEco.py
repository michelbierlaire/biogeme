"""File 16panelDiscreteSocioEco.py

:author: Michel Bierlaire, EPFL
:date: Mon Feb 14 14:47:54 2022

 Example of a discrete mixture of logit models, also called latent class model.
 The class membership model includes socio-economic variables.
 The datafile is organized as panel data.
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

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

# Read the data
df = pd.read_csv('swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# They are organized as panel data. The variable ID identifies each individual.
database.panel("ID")

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
INCOME = Variable('INCOME')

# Removing some observations can be done directly using pandas.
# remove = (((database.data.PURPOSE != 1) &
#           (database.data.PURPOSE != 3)) |
#          (database.data.CHOICE == 0))
# database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
_ = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
_ = database.DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100)
_ = database.DefineVariable('SM_TT_SCALED', SM_TT / 100.0)
_ = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
_ = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
_ = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)

flat_df = database.generateFlatPanelDataframe(identical_columns=None)
flat_database = db.Database('swissmetro_flat', flat_df)

# Parameters to be estimated. One version for each latent class.
numberOfClasses = 2
B_COST = [
    Beta(f'B_COST_class{i}', 0, None, None, 0) for i in range(numberOfClasses)
]

# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation
B_TIME = [
    Beta(f'B_TIME_class{i}', 0, None, None, 0) for i in range(numberOfClasses)
]

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = [
    Beta(f'B_TIME_S_class{i}', 1, None, None, 0)
    for i in range(numberOfClasses)
]
B_TIME_RND = [
    B_TIME[i] + B_TIME_S[i] * bioDraws(f'B_TIME_RND_class{i}', 'NORMAL_ANTI')
    for i in range(numberOfClasses)
]

# We do the same for the constants, to address serial correlation.
ASC_CAR = [
    Beta(f'ASC_CAR_class{i}', 0, None, None, 0) for i in range(numberOfClasses)
]
ASC_CAR_S = [
    Beta(f'ASC_CAR_S_class{i}', 1, None, None, 0)
    for i in range(numberOfClasses)
]
ASC_CAR_RND = [
    ASC_CAR[i]
    + ASC_CAR_S[i] * bioDraws(f'ASC_CAR_RND_class{i}', 'NORMAL_ANTI')
    for i in range(numberOfClasses)
]

ASC_TRAIN = [
    Beta(f'ASC_TRAIN_class{i}', 0, None, None, 0)
    for i in range(numberOfClasses)
]
ASC_TRAIN_S = [
    Beta(f'ASC_TRAIN_S_class{i}', 1, None, None, 0)
    for i in range(numberOfClasses)
]
ASC_TRAIN_RND = [
    ASC_TRAIN[i]
    + ASC_TRAIN_S[i] * bioDraws(f'ASC_TRAIN_RND_class{i}', 'NORMAL_ANTI')
    for i in range(numberOfClasses)
]

ASC_SM = [
    Beta(f'ASC_SM_class{i}', 0, None, None, 1) for i in range(numberOfClasses)
]
ASC_SM_S = [
    Beta(f'ASC_SM_S_class{i}', 1, None, None, 0)
    for i in range(numberOfClasses)
]
ASC_SM_RND = [
    ASC_SM[i] + ASC_SM_S[i] * bioDraws(f'ASC_SM_RND_class{i}', 'NORMAL_ANTI')
    for i in range(numberOfClasses)
]

# Parameters for the class membership model
CLASS_CTE = Beta('CLASS_CTE', 0, None, None, 0)
CLASS_INC = Beta('CLASS_INC', 0, None, None, 0)


# In class 0, it is assumed that the time coefficient is zero
B_TIME_RND[0] = 0

# Utility functions
V1 = [
    [
        ASC_TRAIN_RND[i]
        + B_TIME_RND[i] * Variable(f'{t}_TRAIN_TT_SCALED')
        + B_COST[i] * Variable(f'{t}_TRAIN_COST_SCALED')
        for t in range(1, 10)
    ]
    for i in range(numberOfClasses)
]
V2 = [
    [
        ASC_SM_RND[i]
        + B_TIME_RND[i] * Variable(f'{t}_SM_TT_SCALED')
        + B_COST[i] * Variable(f'{t}_SM_COST_SCALED')
        for t in range(1, 10)
    ]
    for i in range(numberOfClasses)
]
V3 = [
    [
        ASC_CAR_RND[i]
        + B_TIME_RND[i] * Variable(f'{t}_CAR_TT_SCALED')
        + B_COST[i] * Variable(f'{t}_CAR_CO_SCALED')
        for t in range(1, 10)
    ]
    for i in range(numberOfClasses)
]
V = [
    [{1: V1[i][t], 2: V2[i][t], 3: V3[i][t]} for t in range(9)]
    for i in range(numberOfClasses)
]

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# The choice model is a discrete mixture of logit, with availability conditions
# We calculate the conditional probability for each class
prob = [
    exp(
        bioMultSum(
            [
                models.loglogit(V[i][t], av, Variable(f'{t+1}_CHOICE'))
                for t in range(9)
            ]
        )
    )
    for i in range(numberOfClasses)
]

# Class membership model
W = CLASS_CTE + CLASS_INC * INCOME
PROB_class0 = models.logit({0: W, 1: 0}, None, 0)
PROB_class1 = models.logit({0: W, 1: 0}, None, 1)

# Conditional to the random variables, likelihood for the individual.
probIndiv = PROB_class0 * prob[0] + PROB_class1 * prob[1]

# We integrate over the random variables using Monte-Carlo
logprob = log(MonteCarlo(probIndiv))

# Define level of verbosity
logger = msg.bioMessage()
# logger.setSilent()
# logger.setWarning()
logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(flat_database, logprob, numberOfDraws=100000)
biogeme.modelName = '16panelDiscreteSocioEco'

# Estimate the parameters.
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)
