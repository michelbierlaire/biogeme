"""File 01logit_p.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 15 11:02:18 2020

 Example of a logit model.
 Three alternatives: Train, Car and Swissmetro
 SP data

The Swissmetro data is organized such that each row contains all the
responses of one individual.

"""

# pylint: disable=invalid-name, undefined-variable

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioMultSum

# Read the data
df = pd.read_csv('swissmetro_panel.dat', sep='\t')
database = db.Database('swissmetro', df)

# Number of observations for each individual. The are numbered from 0
# to 8 in the dat set.
nbrQuestions = 9

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Removing some observations
exclude = (PURPOSE != 1) * (PURPOSE != 3)
for q in range(nbrQuestions):
    exclude = exclude + (Variable(f'CHOICE_{q}') == 0)
database.remove(exclude > 0)

print(
    f'The database has {database.data.shape[0]} observations, '
    f'and {database.data.shape[1]} columns'
)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Definition of new variables
SM_COST = [Variable(f'SM_CO_{q}') * (GA == 0) for q in range(nbrQuestions)]

TRAIN_COST = [
    Variable(f'TRAIN_CO_{q}') * (GA == 0) for q in range(nbrQuestions)
]

# Definition of new variables: adding columns to the database
CAR_AV_SP = [
    database.DefineVariable(
        f'CAR_AV_SP_{q}', Variable(f'CAR_AV_{q}') * (SP != 0)
    )
    for q in range(nbrQuestions)
]

TRAIN_AV_SP = [
    database.DefineVariable(
        f'TRAIN_AV_SP_{q}', Variable(f'TRAIN_AV_{q}') * (SP != 0)
    )
    for q in range(nbrQuestions)
]

SM_AV = [Variable(f'SM_AV_{q}') for q in range(nbrQuestions)]

TRAIN_TT_SCALED = [
    database.DefineVariable(
        f'TRAIN_TT_SCALED_{q}', Variable(f'TRAIN_TT_{q}') / 100.0
    )
    for q in range(nbrQuestions)
]

TRAIN_COST_SCALED = [
    database.DefineVariable(f'TRAIN_COST_SCALED_{q}', TRAIN_COST[q] / 100)
    for q in range(nbrQuestions)
]

SM_TT_SCALED = [
    database.DefineVariable(
        f'SM_TT_SCALED_{q}', Variable(f'SM_TT_{q}') / 100.0
    )
    for q in range(nbrQuestions)
]

SM_COST_SCALED = [
    database.DefineVariable(f'SM_COST_SCALED_{q}', SM_COST[q] / 100)
    for q in range(nbrQuestions)
]

CAR_TT_SCALED = [
    database.DefineVariable(
        f'CAR_TT_SCALED_{q}', Variable(f'CAR_TT_{q}') / 100
    )
    for q in range(nbrQuestions)
]

CAR_CO_SCALED = [
    database.DefineVariable(
        f'CAR_CO_SCALED_{q}', Variable(f'CAR_CO_{q}') / 100
    )
    for q in range(nbrQuestions)
]

# Definition of the utility functions
V1 = [
    ASC_TRAIN + B_TIME * TRAIN_TT_SCALED[q] + B_COST * TRAIN_COST_SCALED[q]
    for q in range(nbrQuestions)
]
V2 = [
    ASC_SM + B_TIME * SM_TT_SCALED[q] + B_COST * SM_COST_SCALED[q]
    for q in range(nbrQuestions)
]
V3 = [
    ASC_CAR + B_TIME * CAR_TT_SCALED[q] + B_COST * CAR_CO_SCALED[q]
    for q in range(nbrQuestions)
]

# Associate utility functions with the numbering of alternatives
V = [{1: V1[q], 2: V2[q], 3: V3[q]} for q in range(nbrQuestions)]

# Associate the availability conditions with the alternatives
av = [
    {1: TRAIN_AV_SP[q], 2: SM_AV[q], 3: CAR_AV_SP[q]}
    for q in range(nbrQuestions)
]

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = [
    models.loglogit(V[q], av[q], Variable(f'CHOICE_{q}'))
    for q in range(nbrQuestions)
]

# Create the Biogeme object
biogeme = bio.BIOGEME(database, bioMultSum(logprob))
biogeme.modelName = '01logit_p'

# Estimate the parameters
results = biogeme.estimate()
biogeme.createLogFile()

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
