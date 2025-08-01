import unittest

import biogeme.biogeme as bio
import biogeme.biogeme_logging as blog
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
    SM_HE,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_CO,
    TRAIN_COST_SCALED,
    TRAIN_HE,
    TRAIN_TT,
    read_data,
)
from biogeme.expressions import Beta, Derive
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit

logger = blog.get_screen_logger(level=blog.INFO)

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

ASC_CAR = Beta('ASC_CAR', -0.6, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', -0.3, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME_SM = Beta('B_TIME_SM', -1, None, None, 0)
B_TIME_TRAIN = Beta('B_TIME_TRAIN', -1.07, None, None, 0)
B_TIME_CAR = Beta('B_TIME_CAR', -0.86, None, None, 0)
B_COST = Beta('B_COST', -0.97, None, None, 0)
B_HEADWAY_SM = Beta('B_HEADWAY_SM', -0.008, None, None, 0)
B_HEADWAY_TRAIN = Beta('B_HEADWAY_TRAIN', -0.004, None, None, 0)
GA_TRAIN = Beta('GA_TRAIN', 1.14, None, None, 0)
GA_SM = Beta('GA_SM', -0.14, None, None, 0)

MU_EXISTING = Beta('MU_EXISTING', 1.77, 1, None, 0)
MU_PUBLIC = Beta('MU_PUBLIC', 1.84, 1, None, 0)
ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.645, 0, 1, 0)
ALPHA_PUBLIC = 1.0 - ALPHA_EXISTING


SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

V1 = (
    ASC_TRAIN
    + B_TIME_TRAIN * TRAIN_TT / 100
    + B_COST * TRAIN_COST_SCALED
    + B_HEADWAY_TRAIN * TRAIN_HE
    + GA_TRAIN * GA
)
V2 = (
    ASC_SM
    + B_TIME_SM * SM_TT_SCALED
    + B_COST * SM_COST_SCALED
    + B_HEADWAY_SM * SM_HE
    + GA_SM * GA
)
V3 = ASC_CAR + B_TIME_CAR * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

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
log_prob = models.logcnl(V, av, nests, CHOICE)
prob1 = models.cnl(V, av, nests, 1)
gen_elas1 = Derive(prob1, 'TRAIN_TT') * TRAIN_TT / prob1
simulate = {'Prob. train': prob1, 'Elas. 1': gen_elas1}


class test_11(unittest.TestCase):
    def testEstimationAndSimulation(self):
        self.assertTrue(log_prob.is_complex())
        biogeme = bio.BIOGEME(
            database,
            log_prob,
            generate_html=False,
            generate_yaml=False,
            save_iterations=False,
        )
        biogeme.model_name = 'test_11'
        results = biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, -4997.865308202665, 2)
        biosim = bio.BIOGEME(database, simulate)
        simresults = biosim.simulate(results.get_beta_values())
        self.assertAlmostEqual(
            sum(simresults['Prob. train']), 902.4604168804027, delta=1
        )


if __name__ == '__main__':
    unittest.main()
