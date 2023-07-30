"""File swissmetro.py

Processing of the Swissmetro data, organized such that each row
contains all the responses of one individual.

"""

import pandas as pd
import biogeme.database as db

from biogeme.expressions import Variable

# Number of observations for each individual. The are numbered from 0
# to 8 in the dat set.
NBR_QUESTIONS = 9

# Read the data
df = pd.read_csv('swissmetro_panel.dat', sep='\t')
database = db.Database('swissmetro', df)


PURPOSE = Variable('PURPOSE')
GA = Variable('GA')
SP = Variable('SP')

# Removing some observations
exclude = (PURPOSE != 1) * (PURPOSE != 3)
for q in range(NBR_QUESTIONS):
    exclude = exclude + (Variable(f'CHOICE_{q}') == 0)
database.remove(exclude > 0)

print(
    f'The database has {database.data.shape[0]} observations, '
    f'and {database.data.shape[1]} columns'
)


# Definition of new variables
SM_COST = [Variable(f'SM_CO_{q}') * (GA == 0) for q in range(NBR_QUESTIONS)]

TRAIN_COST = [Variable(f'TRAIN_CO_{q}') * (GA == 0) for q in range(NBR_QUESTIONS)]

# Definition of new variables: adding columns to the database
CAR_AV_SP = [
    database.DefineVariable(f'CAR_AV_SP_{q}', Variable(f'CAR_AV_{q}') * (SP != 0))
    for q in range(NBR_QUESTIONS)
]

TRAIN_AV_SP = [
    database.DefineVariable(f'TRAIN_AV_SP_{q}', Variable(f'TRAIN_AV_{q}') * (SP != 0))
    for q in range(NBR_QUESTIONS)
]

SM_AV = [Variable(f'SM_AV_{q}') for q in range(NBR_QUESTIONS)]

TRAIN_TT_SCALED = [
    database.DefineVariable(f'TRAIN_TT_SCALED_{q}', Variable(f'TRAIN_TT_{q}') / 100.0)
    for q in range(NBR_QUESTIONS)
]

TRAIN_COST_SCALED = [
    database.DefineVariable(f'TRAIN_COST_SCALED_{q}', TRAIN_COST[q] / 100)
    for q in range(NBR_QUESTIONS)
]

SM_TT_SCALED = [
    database.DefineVariable(f'SM_TT_SCALED_{q}', Variable(f'SM_TT_{q}') / 100.0)
    for q in range(NBR_QUESTIONS)
]

SM_COST_SCALED = [
    database.DefineVariable(f'SM_COST_SCALED_{q}', SM_COST[q] / 100)
    for q in range(NBR_QUESTIONS)
]

CAR_TT_SCALED = [
    database.DefineVariable(f'CAR_TT_SCALED_{q}', Variable(f'CAR_TT_{q}') / 100)
    for q in range(NBR_QUESTIONS)
]

CAR_CO_SCALED = [
    database.DefineVariable(f'CAR_CO_SCALED_{q}', Variable(f'CAR_CO_{q}') / 100)
    for q in range(NBR_QUESTIONS)
]
