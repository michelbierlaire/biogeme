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
from biogeme.expressions import Beta, Draws, MonteCarlo, PanelLikelihoodTrajectory, log
from biogeme.results_processing import EstimationResults

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)
database.panel('ID')

ASC_CAR = Beta('ASC_CAR', -0.039775, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', -0.739801, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', -6.131332, None, 0, 0)
B_COST = Beta('B_COST', -2.955561, None, 0, 0)

SIGMA_CAR = Beta('SIGMA_CAR', 3.830629, None, None, 0)
SIGMA_SM = Beta('SIGMA_SM', 0.936053, None, None, 0)
SIGMA_TRAIN = Beta('SIGMA_TRAIN', 2.898149, None, None, 0)

# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
EC_CAR = SIGMA_CAR * Draws('EC_CAR', 'NORMAL')
EC_SM = SIGMA_SM * Draws('EC_SM', 'NORMAL')
EC_TRAIN = SIGMA_TRAIN * Draws('EC_TRAIN', 'NORMAL')

SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)


# For latent class 1, where the time coefficient is zero
V11 = ASC_TRAIN + B_COST * TRAIN_COST_SCALED + EC_TRAIN
V12 = ASC_SM + B_COST * SM_COST_SCALED + EC_SM
V13 = ASC_CAR + B_COST * CAR_CO_SCALED + EC_CAR

V1 = {1: V11, 2: V12, 3: V13}

# For latent class 2, where the time coefficient is estimated
V21 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED + EC_TRAIN
V22 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED + EC_SM
V23 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED + EC_CAR

V2 = {1: V21, 2: V22, 3: V23}


av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


# Class membership model
W_OTHER = Beta('W_OTHER', 0.803059, 0, 1, 0)
probClass1 = 1 - W_OTHER
probClass2 = W_OTHER

# The choice model is a discrete mixture of logit, with availability conditions
prob1 = PanelLikelihoodTrajectory(models.logit(V1, av, CHOICE))
prob2 = PanelLikelihoodTrajectory(models.logit(V2, av, CHOICE))
prob_indiv = probClass1 * prob1 + probClass2 * prob2
log_prob = log(MonteCarlo(prob_indiv))


class test_15(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(
            database,
            log_prob,
            number_of_draws=100,
            seed=1111,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
        )
        results: EstimationResults = biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, -3639.291359844362, 2)


if __name__ == '__main__':
    unittest.main()
