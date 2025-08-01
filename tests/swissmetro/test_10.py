import unittest

import biogeme.biogeme as bio
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
from biogeme.expressions import Beta
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)


ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Note that MU cannot be zero
MU = Beta('MU', 0.5, 0.000001, 1.0, 0)


SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)


V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests:
# 1: nests parameter
# 2: list of alternatives

existing = OneNestForNestedLogit(
    nest_param=1, list_of_alternatives=[1, 3], name='existing'
)
nests = NestsForNestedLogit(choice_set=[1, 2, 3], tuple_of_nests=(existing,))

# The choice model is a nested logit, with availability conditions
log_prob = models.lognested_mev_mu(V, av, nests, CHOICE, MU)


class test_10(unittest.TestCase):
    def testEstimation(self):
        self.assertTrue(log_prob.is_complex())
        biogeme = bio.BIOGEME(
            database,
            log_prob,
            parameters=None,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
        )
        results = biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, -5236.90001357885, 2)


if __name__ == '__main__':
    unittest.main()
