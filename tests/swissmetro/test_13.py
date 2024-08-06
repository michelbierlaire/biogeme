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
from biogeme.expressions import (
    Beta,
    bioDraws,
    log,
    MonteCarlo,
    PanelLikelihoodTrajectory,
)
from biogeme.parameters import Parameters

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

database.panel('ID')

ASC_CAR = Beta('ASC_CAR', -0.165109, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', -0.756938, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', -0.756938, None, None, 0)
B_COST = Beta('B_COST', -0.756938, None, None, 0)

SIGMA_CAR = Beta('SIGMA_CAR', 0.262894, None, None, 0)
SIGMA_SM = Beta('SIGMA_SM', 0, None, None, 1)
SIGMA_TRAIN = Beta('SIGMA_TRAIN', -0.756938, None, None, 0)


# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
EC_CAR = SIGMA_CAR * bioDraws('EC_CAR', 'NORMAL')
EC_SM = SIGMA_SM * bioDraws('EC_SM', 'NORMAL')
EC_TRAIN = SIGMA_TRAIN * bioDraws('EC_TRAIN', 'NORMAL')


SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + EC_TRAIN
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + EC_SM
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED + EC_CAR

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

obsprob = models.logit(V, av, CHOICE)
condprobIndiv = PanelLikelihoodTrajectory(obsprob)
logprob = log(MonteCarlo(condprobIndiv))


class test_13(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(database, logprob, number_of_draws=100, seed=111)
        biogeme.save_iterations = False
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -3890.874328534381, 2)


if __name__ == '__main__':
    unittest.main()
