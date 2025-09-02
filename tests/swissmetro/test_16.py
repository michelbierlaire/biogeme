import unittest

import biogeme.biogeme as bio
from biogeme import models
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    GA,
    INCOME,
    PURPOSE,
    SM_AV,
    SM_CO,
    TRAIN_AV_SP,
    TRAIN_CO,
    read_data,
)
from biogeme.database import Database, PanelDatabase
from biogeme.default_parameters import MISSING_VALUE
from biogeme.expressions import Beta, Draws, MonteCarlo, MultipleSum, Variable, exp, log

database = read_data()
# Keep only trip purposes 1 (commuter) and 3 (business)
exclude = ((PURPOSE != 1) * (PURPOSE != 3)) > 0
database.remove(exclude)

panel_database = PanelDatabase(database=database, panel_column='ID')

flat_dataframe, largest_group = panel_database.flatten_database(
    missing_data=MISSING_VALUE
)
flat_database = Database(name='flat_swissmetro', dataframe=flat_dataframe)

SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)


# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
SIGMA_CAR = Beta('SIGMA_CAR', 3.818728, None, None, 0)
SIGMA_SM = Beta('SIGMA_SM', 0.946202, None, None, 0)
SIGMA_TRAIN = Beta('SIGMA_TRAIN', 2.874614, None, None, 0)

EC_CAR = SIGMA_CAR * Draws('EC_CAR', 'NORMAL')
EC_SM = SIGMA_SM * Draws('EC_SM', 'NORMAL')
EC_TRAIN = SIGMA_TRAIN * Draws('EC_TRAIN', 'NORMAL')

ASC_CAR = Beta('ASC_CAR', -0.035345, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', -0.747857, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', -6.136095, None, 0, 0)
B_COST = Beta('B_COST', -2.955199, None, 0, 0)


# For latent class 1, where the time coefficient is zero
V11 = [
    ASC_TRAIN + B_COST * Variable(f'TRAIN_COST_SCALED__panel__{t:02d}') + EC_TRAIN
    for t in range(1, largest_group + 1)
]
V12 = [
    ASC_SM + B_COST * Variable(f'SM_COST_SCALED__panel__{t:02d}') + EC_SM
    for t in range(1, largest_group + 1)
]
V13 = [
    ASC_CAR + B_COST * Variable(f'CAR_CO_SCALED__panel__{t:02d}') + EC_CAR
    for t in range(1, largest_group + 1)
]

V1 = [{1: V11[t], 2: V12[t], 3: V13[t]} for t in range(largest_group)]

# For latent class 2, where the time coefficient is estimated
V21 = [
    ASC_TRAIN
    + B_TIME * Variable(f'TRAIN_TT_SCALED__panel__{t:02d}')
    + B_COST * Variable(f'TRAIN_COST_SCALED__panel__{t:02d}')
    + EC_TRAIN
    for t in range(1, largest_group + 1)
]
V22 = [
    ASC_SM
    + B_TIME * Variable(f'SM_TT_SCALED__panel__{t:02d}')
    + B_COST * Variable(f'SM_COST_SCALED__panel__{t:02d}')
    + EC_SM
    for t in range(1, largest_group + 1)
]
V23 = [
    ASC_CAR
    + B_TIME * Variable(f'CAR_TT_SCALED__panel__{t:02d}')
    + B_COST * Variable(f'CAR_CO_SCALED__panel__{t:02d}')
    + EC_CAR
    for t in range(1, largest_group + 1)
]

V2 = [{1: V21[t], 2: V22[t], 3: V23[t]} for t in range(largest_group)]

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}


# Class membership model
CLASS_CTE = Beta('CLASS_CTE', -0.792233, None, None, 0)
CLASS_INC = Beta('CLASS_INC', -0.251995, None, None, 0)
W1 = CLASS_CTE + CLASS_INC * INCOME
prob_class1 = models.logit({1: W1, 2: 0}, None, 1)
prob_class2 = models.logit({1: W1, 2: 0}, None, 2)

# The choice model is a discrete mixture of logit, with availability conditions
# Conditional to the random variables, likelihood if the individual is
# in class 1
obs_prob_1 = [
    models.loglogit(V1[t], av, Variable(f'CHOICE__panel__{t + 1:02d}'))
    for t in range(largest_group)
]
prob1 = exp(MultipleSum(obs_prob_1))

# The choice model is a discrete mixture of logit, with availability conditions
# Conditional to the random variables, likelihood if the individual is
# in class 2
obs_prob_2 = [
    models.loglogit(V2[t], av, Variable(f'CHOICE__panel__{t + 1:02d}'))
    for t in range(largest_group)
]
prob2 = exp(MultipleSum(obs_prob_2))

# Conditional to the random variables, likelihood for the individual.
prob_indiv = prob_class1 * prob1 + prob_class2 * prob2

# We integrate over the random variables using Monte-Carlo
logprob = log(MonteCarlo(prob_indiv))


class Test16(unittest.TestCase):
    def testEstimation(self):
        biogeme = bio.BIOGEME(
            flat_database,
            logprob,
            number_of_draws=100,
            seed=1111,
            save_iterations=False,
            generate_html=False,
            generate_yaml=False,
        )
        results = biogeme.estimate()
        self.assertAlmostEqual(results.final_log_likelihood, -3637.7836216145975, 2)


if __name__ == '__main__':
    unittest.main()
