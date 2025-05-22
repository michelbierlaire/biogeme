import unittest

from biogeme.biogeme import BIOGEME
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    PURPOSE,
    TRAIN_AV_SP,
    TRAIN_CO,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    read_data,
)
from biogeme.expressions import Beta, Elem, NormalCdf, log

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

P = {1: NormalCdf(V1 - V3), 3: NormalCdf(V3 - V1)}


prob = Elem(P, CHOICE)


class test_21(unittest.TestCase):
    def testEstimation(self):
        biogeme = BIOGEME(
            database,
            log(prob),
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
        )
        results = biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, -986.1887862710962, 10)


if __name__ == '__main__':
    unittest.main()
