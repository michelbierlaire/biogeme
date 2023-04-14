"""File b09nested_wtp.py

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 16:28:29 2023

 Example of a nested logit model, with moneymetrix utilities (WTP space).
"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.results import calculate_correlation
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

mu_nest = Beta('mu_nest', 1, 0, None, 0)
mu = Beta('mu', 1, 0, 1, 0)

# Definition of the cost functions
cost_1 = TRAIN_COST_SCALED + value_of_train + value_of_time * TRAIN_TT_SCALED
cost_2 = SM_COST_SCALED + value_of_time * SM_TT_SCALED
cost_3 = CAR_CO_SCALED + value_of_car + value_of_time * CAR_TT_SCALED

# To obtain the utilities, we change the sign of the cost
# functions. Note that we do not scale the utilities here, as the
# scale parameter is involved in the MEV function of the nested logit
# model

scaled_V1 = -mu * cost_1
scaled_V2 = -mu * cost_2
scaled_V3 = -mu * cost_3


# Associate utility functions with the numbering of alternatives
V = {1: scaled_V1, 2: scaled_V2, 3: scaled_V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests:
# 1: nests parameter
# 2: list of alternatives
existing = mu_nest, [1, 3]
future = 1.0, [2]
nests = existing, future

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
# The choice model is a nested logit, with availability conditions
logprob = models.lognested(V, av, nests, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = "b09nested_wtp"

# Calculate the null log likelihood for reporting.
the_biogeme.calculateNullLoglikelihood(av)

# Estimate the parameters
results = the_biogeme.estimate()
print(results.shortSummary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)

corr = calculate_correlation(
    nests,
    results,
    alternative_names={1: 'Train', 2: 'Swissmetro', 3: 'Car'},
)
print(corr)
