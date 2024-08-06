"""

Simulation of a logit model
===========================

Example of simulation with a logit model


:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:13:23 2023

"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Derive

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    database,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
)

# %%
# Parameters.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Choice probability.
#
prob1 = models.logit(V, av, 1)
prob2 = models.logit(V, av, 2)
prob3 = models.logit(V, av, 3)

# %%
# Elasticities.
#
# Elasticities can be computed. We illustrate below two
# formulas. Check in the output file that they produce the same
# result.

# %%
# First, the general definition of elasticities. This illustrates the
# use of the Derive expression, and can be used with any model,
# however complicated it is. Note the quotes in the Derive opertor.

genelas1 = Derive(prob1, 'TRAIN_TT') * TRAIN_TT / prob1
genelas2 = Derive(prob2, 'SM_TT') * SM_TT / prob2
genelas3 = Derive(prob3, 'CAR_TT') * CAR_TT / prob3

# %%
# Second, the elasticity of logit models. See Ben-Akiva and Lerman for
# the formula

logitelas1 = TRAIN_AV_SP * (1.0 - prob1) * TRAIN_TT_SCALED * B_TIME
logitelas2 = SM_AV * (1.0 - prob2) * SM_TT_SCALED * B_TIME
logitelas3 = CAR_AV_SP * (1.0 - prob3) * CAR_TT_SCALED * B_TIME

# %%
# Quantities to be simulated.
#
simulate = {
    'Prob. train': prob1,
    'Prob. Swissmetro': prob2,
    'Prob. car': prob3,
    'logit elas. 1': logitelas1,
    'generic elas. 1': genelas1,
    'logit elas. 2': logitelas2,
    'generic elas. 2': genelas2,
    'logit elas. 3': logitelas3,
    'generic elas. 3': genelas3,
}


# %%
# Create the Biogeme object.
#
# As we simulate the probability for all aternatives, even when one of
# them is not available, Biogeme may trigger some warnings.
biosim = bio.BIOGEME(database, simulate)
biosim.modelName = 'b01logit_simul'

# %%
# Values of the parameters.
betas = {
    'ASC_TRAIN': -0.701188,
    'B_TIME': -1.27786,
    'B_COST': -1.08379,
    'ASC_CAR': -0.154633,
}


# %%
# Simulation
#
results = biosim.simulate(theBetaValues=betas)
results.describe()
