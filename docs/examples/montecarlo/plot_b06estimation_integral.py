"""

Estimation of mixtures of logit
===============================

Estimation of a mixtures of logit models where the integral is
calculated using numerical integration.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 21:03:03 2023
"""

import biogeme.biogeme as bio
import biogeme.distributions as dist
from biogeme import models
from biogeme.expressions import Beta, RandomVariable, Integrate, log

from swissmetro import (
    database,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    SM_AV,
    CAR_AV_SP,
    CHOICE,
)

# %%
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distirbuted, designed to be used
# for Monte-Carlo simulation
omega = RandomVariable('omega')
density = dist.normalpdf(omega)
B_TIME_RND = B_TIME + B_TIME_S * omega

# %%
# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# The choice model is a logit, with availability conditions
condprob = models.logit(V, av, CHOICE)
prob = Integrate(condprob * density, 'omega')
logprob = log(prob)

# %%
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = '06estimationIntegral'

# %%
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
