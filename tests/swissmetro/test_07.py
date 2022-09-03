import os
import unittest
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, log


myPath = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{myPath}/swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
# print(database.data.describe())

globals().update(database.variables)

# Here we use the 'biogeme' way for backward compatibility
exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)


ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, 0, 0)
B_COST = Beta('B_COST', 0, None, 0, 0)


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

# For latent class 1, whete the time coefficient is zero
V11 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED
V12 = ASC_SM + B_COST * SM_COST_SCALED
V13 = ASC_CAR + B_COST * CAR_CO_SCALED

V1 = {1: V11, 2: V12, 3: V13}

# For latent class 2, whete the time coefficient is estimated
V21 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V22 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V23 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

V2 = {1: V21, 2: V22, 3: V23}


# Associate the availability conditions with the alternatives

CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


# Class membership model
W_OTHER = Beta('W_OTHER', 0.5, 0, 1, 0)
probClass1 = 1 - W_OTHER
probClass2 = W_OTHER

# The choice model is a discrete mixture of logit, with availability conditions
prob1 = models.logit(V1, av, CHOICE)
prob2 = models.logit(V2, av, CHOICE)
prob = probClass1 * prob1 + probClass2 * prob2
logprob = log(prob)


class test_07(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.saveIterations = False
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -5208.4980304812725, 2)


if __name__ == '__main__':
    unittest.main()
