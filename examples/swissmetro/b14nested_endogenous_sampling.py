"""File b14nested_endogenous_sampling.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 18:25:03 2023

 Example of a nested logit model, with the corrections for endogenous sampling.
 Three alternatives: Train, Car and Swissmetro
 Train and car are in the same nest.

"""

import numpy as np
import biogeme.logging as blog
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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b14nested_endogenous_sampling.py')

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
MU = Beta('MU', 1, 1, 10, 0)

# In this example, we assume that the three modes exist, and that the
# sampling protocal is choice-based. The probability that a respondent
# belongs to the sample is R_i.
R_TRAIN = 4.42e-2
R_SM = 3.36e-3
R_CAR = 7.5e-3

# The correction terms are the log of these quantities
correction = {1: np.log(R_TRAIN), 2: np.log(R_SM), 3: np.log(R_CAR)}

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of nests:
# 1: nests parameter
# 2: list of alternatives
existing = MU, [1, 3]
future = 1.0, [2]
nests = existing, future

# The choice model is a nested logit, with corrections for endogenous sampling
# We first obtain the expression of the Gi function for nested logit.
Gi = models.getMevForNested(V, av, nests)

# Then we calculate the MEV log probability, accounting for the correction.
logprob = models.logmev_endogenousSampling(V, Gi, av, correction, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b14nested_endogenous_eampling'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.short_summary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
