import os
import unittest
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Derive

myPath = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{myPath}/swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
# print(database.data.describe())

globals().update(database.variables)

exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)


ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

MU_EXISTING = Beta('MU_EXISTING', 1, 1, None, 0)
MU_PUBLIC = Beta('MU_PUBLIC', 1, 1, None, 0)
ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.5, 0, 1, 0)
ALPHA_PUBLIC = 1 - ALPHA_EXISTING


SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

TRAIN_TT_SCALED = TRAIN_TT / 100.0
TRAIN_COST_SCALED = database.DefineVariable(
    'TRAIN_COST_SCALED', TRAIN_COST / 100
)
SM_TT_SCALED = database.DefineVariable('SM_TT_SCALED', SM_TT / 100.0)
SM_COST_SCALED = database.DefineVariable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.DefineVariable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.DefineVariable('CAR_CO_SCALED', CAR_CO / 100)

V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


# Associate the availability conditions with the alternatives
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests:
alpha_existing = {1: ALPHA_EXISTING, 2: 0.0, 3: 1.0}

alpha_public = {1: ALPHA_PUBLIC, 2: 1.0, 3: 0.0}

nest_existing = MU_EXISTING, alpha_existing
nest_public = MU_PUBLIC, alpha_public
nests = nest_existing, nest_public

# The choice model is a cross-nested logit, with availability conditions
logprob = models.logcnl_avail(V, av, nests, CHOICE)
prob1 = models.cnl_avail(V, av, nests, 1)
genelas1 = Derive(prob1, 'TRAIN_TT') * TRAIN_TT / prob1
simulate = {'Prob. train': prob1, 'Elas. 1': genelas1}


class test_11(unittest.TestCase):
    def testEstimationAndSimulation(self):
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.saveIterations = False
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -5214.049, 2)
        biosim = bio.BIOGEME(database, simulate)
        simresults = biosim.simulate(results.getBetaValues())
        self.assertAlmostEqual(
            sum(simresults['Prob. train']), 888.3883902853023, 1
        )
        self.assertAlmostEqual(
            sum(simresults['Elas. 1']), -17897.702976576973, 0
        )


if __name__ == '__main__':
    unittest.main()
