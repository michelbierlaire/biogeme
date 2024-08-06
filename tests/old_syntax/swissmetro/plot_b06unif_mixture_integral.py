"""

Mixture of logit models
=======================

Example of a mixture of logit models, using numerical integration.
The mixing distribution is uniform.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:52:52 2023

"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta,
    Integrate,
    RandomVariable,
    exp,
    log,
)

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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b06unif_mixture_integral.py')

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for numerical integration
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
omega = RandomVariable('omega')

# %%
# .. |infinity| unicode:: U+221E
#    :trim:
#
# As the numerical integration ranges from -|infinity| \  to + |infinity| ,
# we need to perform a change of variable in order to integrate
# between -1 and 1.
LOWER_BND = -1
UPPER_BND = 1
x = LOWER_BND + (UPPER_BND - LOWER_BND) / (1 + exp(-omega))
dx = (UPPER_BND - LOWER_BND) * exp(-omega) * (1 + exp(-omega)) ** (-2)
B_TIME_RND = B_TIME + B_TIME_S * x

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional on omega, we have a logit model (called the kernel).
condprob = models.logit(V, av, CHOICE)

# %%
# We integrate over omega using numerical integration.
logprob = log(Integrate(condprob * dx / (UPPER_BND - LOWER_BND), 'omega'))

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = '06unif_mixture_integral'

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
