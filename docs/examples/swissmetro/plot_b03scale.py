"""

Heteroscedastic specification
=============================

 Illustrates a heteroscedastic specification. A different scale is
 associated with different segments of the sample.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:23:03 2023


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
    GROUP,
)


# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
Scale_group3 = Beta('Scale_group3', 1, 0.001, None, 0)

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Scale associated with group 3 is estimated.
scale = (GROUP != 3) + (GROUP == 3) * Scale_group3

# %%
# Scale the utility functions, and associate them with the numbering
# of alternatives.
V = {1: scale * V1, 2: scale * V2, 3: scale * V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# %%
# These notes will be included as such in the report file.
USER_NOTES = (
    'Illustrates heteroscedastic specification. A different scale is'
    ' associated with different segments of the sample.'
)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob, user_notes=USER_NOTES)
the_biogeme.modelName = 'b03scale'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
pandas_results
