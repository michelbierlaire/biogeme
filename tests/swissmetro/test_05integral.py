import unittest

import biogeme.biogeme as bio
import biogeme.distributions as dist
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
from biogeme.expressions import (
    Beta,
    log,
    RandomVariable,
    Integrate,
)

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)


ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distirbuted, designed to be used
# for Monte-Carlo simulation

omega = RandomVariable('omega')
density = dist.normalpdf(omega)
B_TIME_RND = B_TIME + B_TIME_S * omega


# Utility functions

# If the person has a GA (season ticket) her incremental cost is actually 0
# rather than the cost value gathered from the
# network data.
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)


V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# The choice model is a logit, with availability conditions
condprob = models.logit(V, av, CHOICE)
prob = Integrate(condprob * density, 'omega')
logprob = log(prob)


class test_02(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(database, logprob, parameters=None)
        biogeme.save_iterations = False
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -5214.879, 2)


if __name__ == '__main__':
    unittest.main()
