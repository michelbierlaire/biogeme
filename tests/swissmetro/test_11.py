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
    TRAIN_TT,
)
from biogeme.expressions import Beta, Derive
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
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


V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests:
alpha_existing = {1: ALPHA_EXISTING, 2: 0.0, 3: 1.0}

alpha_public = {1: ALPHA_PUBLIC, 2: 1.0, 3: 0.0}

nest_existing = OneNestForCrossNestedLogit(
    nest_param=MU_EXISTING, dict_of_alpha=alpha_existing, name='existing'
)
nest_public = OneNestForCrossNestedLogit(
    nest_param=MU_PUBLIC, dict_of_alpha=alpha_public, name='public'
)
nests = NestsForCrossNestedLogit(
    choice_set=[1, 2, 3], tuple_of_nests=(nest_existing, nest_public)
)

# The choice model is a cross-nested logit, with availability conditions
logprob = models.logcnl(V, av, nests, CHOICE)
prob1 = models.cnl(V, av, nests, 1)
genelas1 = Derive(prob1, 'TRAIN_TT') * TRAIN_TT / prob1
simulate = {'Prob. train': prob1, 'Elas. 1': genelas1}


class test_11(unittest.TestCase):
    def testEstimationAndSimulation(self):
        biogeme = bio.BIOGEME(database, logprob, parameters=None)
        biogeme.save_iterations = False
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -5214.049, 2)
        biosim = bio.BIOGEME(database, simulate)
        simresults = biosim.simulate(results.get_beta_values())
        self.assertAlmostEqual(
            sum(simresults['Prob. train']), 888.3883902853023, delta=1
        )


if __name__ == '__main__':
    unittest.main()
