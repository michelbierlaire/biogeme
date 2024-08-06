"""

Box-Cox transforms
==================

Example of a logit model, with a Box-Cox transform of variables.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:58:15 2023
"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta

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
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
LAMBDA = Beta('LAMBDA', 0, None, None, 0)

# %%
# Definition of the utility functions.
V1 = (
    ASC_TRAIN
    + B_TIME * models.boxcox(TRAIN_TT_SCALED, LAMBDA)
    + B_COST * TRAIN_COST_SCALED
)
V2 = ASC_SM + B_TIME * models.boxcox(SM_TT_SCALED, LAMBDA) + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * models.boxcox(CAR_TT_SCALED, LAMBDA) + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b08boxcox'

# %%
# Check the derivatives of the log liklelihood function around 0.
the_biogeme.checkDerivatives(beta=[0, 0, 0, 0, 0], verbose=True)

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
