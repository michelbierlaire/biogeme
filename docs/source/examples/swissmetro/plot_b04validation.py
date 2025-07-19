"""

Out-of-sample validation
========================

Example of the out-of-sample validation of a logit model.

Michel Bierlaire, EPFL
Wed Jun 18 2025, 11:27:07
"""

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.validation import ValidationResult

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, logprob)
the_biogeme.model_name = 'b04validation'

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
validation_results: list[ValidationResult] = the_biogeme.validate(results, slices=5)

for slide in validation_results:
    print(
        f'Log likelihood for {slide.simulated_values.shape[0]} validation data: '
        f'{slide.simulated_values.iloc[:, 0].sum()}'
    )
