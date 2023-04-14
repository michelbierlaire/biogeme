"""File b01logit_wtp.py

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 15:23:14 2023

 Example of a logit model, where the utilities are expressed in the WTP space
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


# Definition of the cost functions
cost_1 = TRAIN_COST_SCALED + value_of_train + value_of_time * TRAIN_TT_SCALED
cost_2 = SM_COST_SCALED + value_of_time * SM_TT_SCALED
cost_3 = CAR_CO_SCALED + value_of_car + value_of_time * CAR_TT_SCALED

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
the_biogeme.modelName = 'b01logit_wtp'

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood(av)

# Estimate the parameters
results = the_biogeme.estimate()
print(results.shortSummary())
# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
print(pandas_results)
