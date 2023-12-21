"""

Numerical integration
=====================

Calculation of a mixtures of logit models where the integral is
calculated using numerical integration.

:author: Michel Bierlaire, EPFL
:date: Thu Apr 13 20:51:32 2023
"""

import biogeme.biogeme as bio
import biogeme.distributions as dist
from biogeme.expressions import RandomVariable, Integrate
from biogeme import models

from swissmetro_one import (
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
# Parameters
ASC_CAR = 0.137
ASC_TRAIN = -0.402
ASC_SM = 0
B_TIME = -2.26
B_TIME_S = 1.66
B_COST = -1.29

# %%
# Define a random parameter, normally distributed,
# designed to be used for integration
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
integrand = models.logit(V, av, CHOICE)
numericalI = Integrate(integrand * density, 'omega')

# %%
simulate = {'Numerical': numericalI}

# %%
biosim = bio.BIOGEME(database, simulate)

# %%
results = biosim.simulate(theBetaValues={})
results

# %%
print('Mixture of logit - numerical integration: ', results.iloc[0]['Numerical'])
