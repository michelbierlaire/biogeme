import os
import unittest
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.distributions as dist
from biogeme.expressions import Beta, log, Elem


myPath = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{myPath}/swissmetro.dat', sep='\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
# print(database.data.describe())

globals().update(database.variables)

exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
database.remove(exclude)


B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

tau1 = Beta('tau1', -1, None, 0, 0)
delta2 = Beta('delta2', 2, 0, None, 0)

tau2 = tau1 + delta2


TRAIN_COST = TRAIN_CO * (GA == 0)

TRAIN_TT_SCALED = database.DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
TRAIN_COST_SCALED = database.DefineVariable(
    'TRAIN_COST_SCALED', TRAIN_COST / 100
)

#  Utility

U = B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED


ChoiceProba = {
    1: 1 - dist.logisticcdf(U - tau1),
    2: dist.logisticcdf(U - tau1) - dist.logisticcdf(U - tau2),
    3: dist.logisticcdf(U - tau2),
}

logprob = log(Elem(ChoiceProba, CHOICE))


class test_18(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(database, logprob)
        biogeme.saveIterations = False
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -5789.309, 2)


if __name__ == '__main__':
    unittest.main()
