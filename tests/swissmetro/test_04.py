import unittest

import biogeme.biogeme as bio
from biogeme import models
from biogeme.data.swissmetro import (
    read_data,
    PURPOSE,
    CHOICE,
    GA,
    TRAIN_CO,
    SM_CO,
    SM_AV,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    CAR_AV_SP,
)
from biogeme.expressions import Beta, log

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Biogeme cannot compute the log of 0. Therefore, whenever the cost
# is 0, the log of 1 computed instead.
LOG_CAR_COST = database.define_variable(
    'LOG_CAR_COST',
    (CAR_CO_SCALED != 0) * log(CAR_CO_SCALED + 1 * (CAR_CO_SCALED == 0)),
)
LOG_TRAIN_COST = database.define_variable(
    'LOG_TRAIN_COST',
    (TRAIN_COST_SCALED != 0) * log(TRAIN_COST_SCALED + 1 * (TRAIN_COST_SCALED == 0)),
)
LOG_SM_COST = database.define_variable(
    'LOG_SM_COST',
    (SM_COST_SCALED != 0) * log(SM_COST_SCALED + 1 * (SM_COST_SCALED == 0)),
)

V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * LOG_TRAIN_COST
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * LOG_SM_COST
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * LOG_CAR_COST


# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


class test_04(unittest.TestCase):
    def testEstimation(self):
        logprob = models.loglogit(V, av, CHOICE)
        biogeme = bio.BIOGEME(database, logprob, parameters=None)
        biogeme.save_iterations = False
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -5423.299, 2)


if __name__ == '__main__':
    unittest.main()
