import unittest

import numpy as np

from biogeme import models
from biogeme.biogeme import BIOGEME
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
from biogeme.draws import RandomNumberGeneratorTuple
from biogeme.expressions import Beta, Draws, MonteCarlo, PanelLikelihoodTrajectory, log

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

database.panel('ID')

#

ASC_CAR = Beta('ASC_CAR', -0.458093, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', -1.110543, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', -0.710008, None, None, 0)
B_COST = Beta('B_COST', -0.988997, None, None, 0)

SIGMA_CAR = Beta('SIGMA_CAR', 0.130422, None, None, 0)
SIGMA_SM = Beta('SIGMA_SM', 0.230524, None, None, 0)
SIGMA_TRAIN = Beta('SIGMA_TRAIN', 0.168784, None, None, 0)


# Provide my own random number generator to the database.
# See the numpy.random documentation to obtain a list of other distributions.
def the_triangular_generator(sample_size, number_of_draws):
    return np.random.triangular(-1, 0, 1, (sample_size, number_of_draws))


my_random_number_generators = {
    'TRIANGULAR': RandomNumberGeneratorTuple(
        the_triangular_generator,
        'Triangular distribution T(-1,0,1)',
    )
}

# Define a random parameter, with a triangular distribution, designed
# to be used for Monte-Carlo simulation
EC_CAR = SIGMA_CAR * Draws('EC_CAR', 'TRIANGULAR')
EC_SM = SIGMA_SM * Draws('EC_SM', 'TRIANGULAR')
EC_TRAIN = SIGMA_TRAIN * Draws('EC_TRAIN', 'TRIANGULAR')


SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)


V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + EC_TRAIN
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + EC_SM
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED + EC_CAR

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}


av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

obs_prob = models.logit(V, av, CHOICE)
cond_prob_indiv = PanelLikelihoodTrajectory(obs_prob)
log_prob = log(MonteCarlo(cond_prob_indiv))


class test_26(unittest.TestCase):
    def testEstimation(self):
        biogeme = BIOGEME(
            database,
            log_prob,
            number_of_draws=100,
            seed=1111,
            random_number_generators=my_random_number_generators,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
        )
        results = biogeme.estimate()
        self.assertAlmostEqual(
            results.final_log_likelihood, -3915.6714670664164, delta=10
        )


if __name__ == '__main__':
    unittest.main()
