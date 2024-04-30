import unittest

import biogeme.biogeme as bio
import biogeme.distributions as dist
from biogeme.data.swissmetro import (
    read_data,
    PURPOSE,
    CHOICE,
    GA,
    TRAIN_CO,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
)
from biogeme.expressions import Beta, log, Elem

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)


B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

tau1 = Beta('tau1', -1, None, 0, 0)
delta2 = Beta('delta2', 2, 0, None, 0)

tau2 = tau1 + delta2


TRAIN_COST = TRAIN_CO * (GA == 0)


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
        biogeme = bio.BIOGEME(database, logprob, parameters=None)
        biogeme.save_iterations = False
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -5789.309, 2)


if __name__ == '__main__':
    unittest.main()
