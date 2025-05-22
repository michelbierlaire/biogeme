import unittest

import biogeme.biogeme as bio
import biogeme.distributions as dist
from biogeme import models
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    PURPOSE,
    SM_AV,
    SM_CO,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_CO,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    read_data,
)
from biogeme.expressions import Beta, IntegrateNormal, RandomVariable, exp, log

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)


ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, 0.0001, 10, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distirbuted, designed to be used
# for Monte-Carlo simulation

omega = RandomVariable('omega')
B_TIME_RND = -exp(B_TIME + B_TIME_S * omega)
density = dist.normalpdf(omega)


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
prob = IntegrateNormal(condprob, 'omega')
logprob = log(prob)


class test_17(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(
            database,
            logprob,
            parameters=None,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
        )
        results = biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, -5231.50634765625, 2)


if __name__ == '__main__':
    unittest.main()
