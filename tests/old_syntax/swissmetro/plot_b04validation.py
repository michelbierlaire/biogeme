"""

Out-of-sample validation
========================

Example of the out-of-sample validation of a logit model.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:24:32 2023

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

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

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
the_biogeme.modelName = 'b04validation'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
# The validation consists in organizing the data into several slices
# of about the same size, randomly defined. Each slice is considered
# as a validation dataset. The model is then re-estimated using all
# the data except the slice, and the estimated model is applied on the
# validation set (i.e. the slice). The value of the log likelihood for
# each observation in the validation set is reported in a
# dataframe. As this is done for each slice, the output is a list of
# dataframes, each corresponding to one of these exercises.

validation_data = database.split(slices=5)

validation_results = the_biogeme.validate(results, validation_data)

for slide in validation_results:
    print(
        f'Log likelihood for {slide.shape[0]} validation data: '
        f'{slide["Loglikelihood"].sum()}'
    )
