"""File b11cnl_wtp.py

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 16:48:29 2023

 Example of a cross-nested logit model with moneymetric utilities (WTP space)

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

value_of_car = Beta('value_of_car', 0, None, None, 0)
value_of_train = Beta('value_of_train', 0, None, None, 0)
value_of_time = Beta('value_of_time', 0, None, None, 0)

mu = Beta('mu', 1, 0, None, 0)

mu_existing = Beta('mu_existing', 1, 1, None, 0)
mu_public = Beta('mu_public', 1, 1, None, 0)
alpha_existing = Beta('alpha_existing', 0.5, 0, 1, 0)
alpha_public = 1 - alpha_existing

# Definition of the cost functions
cost_1 = TRAIN_COST_SCALED + value_of_train + value_of_time * TRAIN_TT_SCALED
cost_2 = SM_COST_SCALED + value_of_time * SM_TT_SCALED
cost_3 = CAR_CO_SCALED + value_of_car + value_of_time * CAR_TT_SCALED

# To obtain the utilities, we change the sign of the cost
# functions. Note that we do not scale the utilities here, as the
# scale parameter is involved in the MEV function of the nested logit
# model

# To obtain the utilities, we change the sign of the cost
# functions. We also scale them with the parameter mu

scaled_V1 = -mu * cost_1
scaled_V2 = -mu * cost_2
scaled_V3 = -mu * cost_3

# Associate utility functions with the numbering of alternatives
V = {1: scaled_V1, 2: scaled_V2, 3: scaled_V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests
# Nest membership parameters
alpha_existing_assign = {1: alpha_existing, 2: 0.0, 3: 1.0}

alpha_public_assign = {1: alpha_public, 2: 1.0, 3: 0.0}

nest_existing = mu_existing, alpha_existing_assign
nest_public = mu_public, alpha_public_assign
nests = nest_existing, nest_public

# The choice model is a cross-nested logit, with availability conditions
logprob = models.logcnl_avail(V, av, nests, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b11cnl_wpt'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.shortSummary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
