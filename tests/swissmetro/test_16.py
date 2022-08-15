import os
import unittest
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta,
    log,
    exp,
    bioDraws,
    MonteCarlo,
    bioMultSum,
    Variable,
)


myPath = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{myPath}/swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)
database.panel('ID')

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
# print(database.data.describe())

globals().update(database.variables)

exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)


SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
TRAIN_COST_SCALED = database.DefineVariable(
    'TRAIN_COST_SCALED', TRAIN_COST / 100
)
SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', SM_TT / 100.0)
SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)

# Associate the availability conditions with the alternatives

CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))


flat_df = database.generateFlatPanelDataframe(identical_columns=None)
flat_database = db.Database('swissmetro_flat', flat_df)

# Define a random parameter, normally distirbuted, designed to be used
# for Monte-Carlo simulation
SIGMA_CAR = Beta('SIGMA_CAR', 3.7, None, None, 0)
SIGMA_SM = Beta('SIGMA_SM', 0.759, None, None, 0)
SIGMA_TRAIN = Beta('SIGMA_TRAIN', 3.02, None, None, 0)

EC_CAR = SIGMA_CAR * bioDraws('EC_CAR', 'NORMAL')
EC_SM = SIGMA_SM * bioDraws('EC_SM', 'NORMAL')
EC_TRAIN = SIGMA_TRAIN * bioDraws('EC_TRAIN', 'NORMAL')

ASC_CAR = Beta('ASC_CAR', 0.136, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', -1, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', -6.3, None, 0, 0)
B_COST = Beta('B_COST', -3.29, None, 0, 0)


# For latent class 1, whete the time coefficient is zero
V11 = [
    ASC_TRAIN + B_COST * Variable(f'{t}_TRAIN_COST_SCALED') + EC_TRAIN
    for t in range(1, 10)
]
V12 = [
    ASC_SM + B_COST * Variable(f'{t}_SM_COST_SCALED') + EC_SM
    for t in range(1, 10)
]
V13 = [
    ASC_CAR + B_COST * Variable(f'{t}_CAR_CO_SCALED') + EC_CAR
    for t in range(1, 10)
]

V1 = [{1: V11[t], 2: V12[t], 3: V13[t]} for t in range(9)]

# For latent class 2, whete the time coefficient is estimated
V21 = [
    ASC_TRAIN
    + B_TIME * Variable(f'{t}_TRAIN_TT_SCALED')
    + B_COST * Variable(f'{t}_TRAIN_COST_SCALED')
    + EC_TRAIN
    for t in range(1, 10)
]
V22 = [
    ASC_SM
    + B_TIME * Variable(f'{t}_SM_TT_SCALED')
    + B_COST * Variable(f'{t}_SM_COST_SCALED')
    + EC_SM
    for t in range(1, 10)
]
V23 = [
    ASC_CAR
    + B_TIME * Variable(f'{t}_CAR_TT_SCALED')
    + B_COST * Variable(f'{t}_CAR_CO_SCALED')
    + EC_CAR
    for t in range(1, 10)
]

V2 = [{1: V21[t], 2: V22[t], 3: V23[t]} for t in range(9)]

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


# Class membership model
# Class membership model
CLASS_CTE = Beta('CLASS_CTE', 0, None, None, 0)
CLASS_INC = Beta('CLASS_INC', 0, None, None, 0)
W1 = CLASS_CTE + CLASS_INC * INCOME
probClass1 = models.logit({1: W1, 2: 0}, None, 1)
probClass2 = models.logit({1: W1, 2: 0}, None, 2)

# The choice model is a discrete mixture of logit, with availability conditions
# Conditional to the random variables, likelihood if the individual is
# in class 1
obsprob1 = [
    models.loglogit(V1[t], av, Variable(f'{t+1}_CHOICE')) for t in range(9)
]
prob1 = exp(bioMultSum(obsprob1))

# The choice model is a discrete mixture of logit, with availability conditions
# Conditional to the random variables, likelihood if the individual is
# in class 2
obsprob2 = [
    models.loglogit(V2[t], av, Variable(f'{t+1}_CHOICE')) for t in range(9)
]
prob2 = exp(bioMultSum(obsprob2))

# Conditional to the random variables, likelihood for the individual.
probIndiv = probClass1 * prob1 + probClass2 * prob2

# We integrate over the random variables using Monte-Carlo
logprob = log(MonteCarlo(probIndiv))


class test_16(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(flat_database, logprob, numberOfDraws=5, seed=10)
        biogeme.saveIterations = False
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -4071.8391253004647, 2)


if __name__ == '__main__':
    unittest.main()
