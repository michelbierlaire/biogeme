"""

Calculation of individual level parameters
==========================================

Calculation of the individual level parameters for the model defined
in :ref:`plot_b05normal_mixture`.

:author: Michel Bierlaire, EPFL
:date: Mon Apr 10 12:17:12 2023

"""

import os
import pickle
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, MonteCarlo

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    database,
    CHOICE,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
)

# %%
# Parameters. The initial value is irrelevant.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

# %%
# Define values for these parameters
beta_values = {
    'ASC_CAR': 0.137,
    'ASC_TRAIN': -0.402,
    'B_COST': -1.28,
    'B_TIME': -2.26,
    'B_TIME_S': 1.65,
}

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional on B_TIME_RND, we have a logit model (called the kernel).
prob_chosen = models.logit(V, av, CHOICE)

# %%
# Numerator and denominator of the formula for individual parameters.
numerator = MonteCarlo(B_TIME_RND * prob_chosen)
denominator = MonteCarlo(prob_chosen)

# %%
simulate = {
    'Numerator': numerator,
    'Denominator': denominator,
    'Choice': CHOICE,
}

# %%
# The results are saved in a picke file. The next time the script is
# run, if the file exists, the results are simply loaded instead of
# being re-calcuated.
PICKLE_FILE = 'b19individual_level_parameters.pickle'
if os.path.isfile(PICKLE_FILE):
    with open(PICKLE_FILE, 'rb') as f:
        sim = pickle.load(f)
else:
    biosim = bio.BIOGEME(database, simulate)
    sim = biosim.simulate(beta_values)
    sim['Individual-level parameters'] = sim['Numerator'] / sim['Denominator']
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(sim, f)

sim
