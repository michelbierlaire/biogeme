import os
import unittest
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Elem, Derive

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

ASC_TRAIN = Beta('ASC_TRAIN', -0.701188, None, None, 0)
B_TIME = Beta('B_TIME', -1.27786, None, None, 0)
B_COST = Beta('B_COST', -1.08379, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 0)
ASC_CAR = Beta('ASC_CAR', -0.154633, None, None, 0)

SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

TRAIN_TT_SCALED = TRAIN_TT / 100.0
TRAIN_COST_SCALED = TRAIN_COST / 100
SM_TT_SCALED = SM_TT / 100.0
SM_COST_SCALED = SM_COST / 100.0
CAR_TT_SCALED = CAR_TT / 100.0
CAR_CO_SCALED = CAR_CO / 100.0

V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


# Associate the availability conditions with the alternatives
CAR_AV_SP = database.DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# The choice model is a logit, with availability conditions
prob1 = Elem({0: 0, 1: models.logit(V, av, 1)}, av[1])

# Elasticities can be computed. We illustrate below two
# formulas. Check in the output file that they produce the same
# result.

# First, the general definition of elasticities. This illustrates the
# use of the Derive expression, and can be used with any model,
# however complicated it is. Note the quotes in the Derive opertor.

genelas1 = Derive(prob1, 'TRAIN_TT') * TRAIN_TT / prob1

# Second, the elasticity of logit models. See Ben-Akiva and Lerman for
# the formula

logitelas1 = TRAIN_AV_SP * (1.0 - prob1) * TRAIN_TT_SCALED * B_TIME

simulate = {
    'P1': prob1,
    'logit elas. 1': logitelas1,
    'generic elas. 1': genelas1,
}


class test_01simul(unittest.TestCase):
    def testSimulation(self):
        biogeme = bio.BIOGEME(database, simulate)
        biogeme.saveIterations = False
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        biogeme.modelName = '01logit_simul'
        results = biogeme.simulate()
        self.assertAlmostEqual(sum(results['P1']), 907.9992101964821, 2)
        self.assertAlmostEqual(
            sum(results['logit elas. 1']), -12673.838605478186, 2
        )
        self.assertAlmostEqual(
            sum(results['generic elas. 1']), -12673.838605478186, 2
        )


if __name__ == '__main__':
    unittest.main()
