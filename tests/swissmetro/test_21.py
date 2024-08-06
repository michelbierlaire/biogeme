import unittest

import biogeme.biogeme as bio
from biogeme.data.swissmetro import (
    read_data,
    PURPOSE,
    CHOICE,
    GA,
    TRAIN_CO,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    CAR_AV_SP,
)
from biogeme.expressions import Beta, bioNormalCdf, Elem, log

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

# As we estimate a binary model, we remove observations where
# Swissmetro was chosen (CHOICE == 2). We also remove observations
# where one of the two alternatives is not available.

exclude = (TRAIN_AV_SP == 0) + (CAR_AV_SP == 0) + (CHOICE == 2) + (
    (PURPOSE != 1) * (PURPOSE != 3)
) > 0


database.remove(exclude)


ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)


TRAIN_COST = TRAIN_CO * (GA == 0)


# We estimate a binary probit model. There are only two alternatives.
V1 = B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate choice probability with the numbering of alternatives

P = {1: bioNormalCdf(V1 - V3), 3: bioNormalCdf(V3 - V1)}


prob = Elem(P, CHOICE)


class test_02(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(database, log(prob))
        biogeme.save_iterations = False
        biogeme.generate_html = False
        biogeme.generate_pickle = False
        results = biogeme.estimate()
        self.assertAlmostEqual(results.data.logLike, -986.1888, 2)


if __name__ == '__main__':
    unittest.main()
