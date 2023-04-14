"""File b08boxcox.py

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 16:23:42 2023

oExample of a logit model, with a Box-Cox transform of variables, in
moneytric utilities (WTP space).

"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta

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

# Parameters to be estimated.
# In WTP space, the cost coefficient is normalized to -1, and the
# scale parameter MU is estimated
value_of_car = Beta('value_of_car', 0, None, None, 0)
value_of_train = Beta('value_of_train', 0, None, None, 0)
value_of_time = Beta('value_of_time', 0, None, None, 0)
mu = Beta('mu', 1, None, None, 0)
ell = Beta('ell', 0, None, None, 0)

# Definition of the cost functions
cost_1 = (
    TRAIN_COST_SCALED
    + value_of_train
    + value_of_time * models.boxcox(TRAIN_TT_SCALED, ell)
)
cost_2 = SM_COST_SCALED + value_of_time * models.boxcox(SM_TT_SCALED, ell)
cost_3 = (
    CAR_CO_SCALED + value_of_car + value_of_time * models.boxcox(CAR_TT_SCALED, ell)
)

# To obtain the utilities, we change the sign of the cost
# functions. We also scale them with the parameter mu

scaled_V1 = -mu * cost_1
scaled_V2 = -mu * cost_2
scaled_V3 = -mu * cost_3

# Associate utility functions with the numbering of alternatives
V = {1: scaled_V1, 2: scaled_V2, 3: scaled_V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)


# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b08boxcox'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.shortSummary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
