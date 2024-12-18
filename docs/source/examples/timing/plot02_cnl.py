"""

Timing of a cross-nested logit model
====================================

Michel Bierlaire
Tue Jul 2 14:49:25 2024
"""

# %%
# See the data processing script: :ref:`swissmetro_data`.
from biogeme.data.swissmetro import (
    read_data,
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
from biogeme.expressions import Beta, Expression
from biogeme.models import logcnl
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit
from biogeme.tools.time import Timing

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %% Nest parameters.
MU_EXISTING = Beta('MU_EXISTING', 1.1, 1, 10, 0)
MU_PUBLIC = Beta('MU_PUBLIC', 1.1, 1, 10, 0)

# %%
# Nest membership parameters.
ALPHA_EXISTING = Beta('ALPHA_EXISTING', 0.5, 0, 1, 0)
ALPHA_PUBLIC = 1 - ALPHA_EXISTING

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
# Definition of nests.

nest_existing = OneNestForCrossNestedLogit(
    nest_param=MU_EXISTING,
    dict_of_alpha={1: ALPHA_EXISTING, 2: 0.0, 3: 1.0},
    name='existing',
)

nest_public = OneNestForCrossNestedLogit(
    nest_param=MU_PUBLIC, dict_of_alpha={1: ALPHA_PUBLIC, 2: 1.0, 3: 0.0}, name='public'
)

nests = NestsForCrossNestedLogit(
    choice_set=[1, 2, 3], tuple_of_nests=(nest_existing, nest_public)
)

# %%
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
logprob: Expression = logcnl(V, av, nests, CHOICE)

# %%
database = read_data()

# %%
# Create a Timing object
the_timing = Timing(warm_up_runs=10, num_runs=100)

# %%
# Timing when only the log likelihood is needed
average_time_function_only = the_timing.time_function(
    logprob.get_value_and_derivatives,
    kwargs={
        'gradient': False,
        'hessian': False,
        'bhhh': False,
        'prepare_ids': True,
        'database': database,
    },
)

print(
    f'Cross-nested logit model, log likelihood for one iteration: {average_time_function_only:.3g} seconds'
)

# %%
# Timing when the log likelihood and the gradient
average_time_function_gradient = the_timing.time_function(
    logprob.get_value_and_derivatives,
    kwargs={
        'gradient': True,
        'hessian': False,
        'bhhh': False,
        'prepare_ids': True,
        'database': database,
    },
)
print(
    f'Cross-nested logit model, log likelihood and gradient for one iteration: '
    f'{average_time_function_gradient:.3g} seconds'
)
relative_gradient = (
    average_time_function_gradient - average_time_function_only
) / average_time_function_only
print(
    f'Calculating the gradient means a {100*relative_gradient:.2f}% increase of the calculation time.'
)

# %%
# Timing when the log likelihood, the gradient and hessian are needed
average_time_function_gradient_hessian = the_timing.time_function(
    logprob.get_value_and_derivatives,
    kwargs={
        'gradient': True,
        'hessian': True,
        'bhhh': False,
        'prepare_ids': True,
        'database': database,
    },
)
print(
    f'Cross-nested logit model, log likelihood, gradient and hessian for one iteration: '
    f'{average_time_function_gradient_hessian:.3g} seconds'
)
relative_to_function = (
    average_time_function_gradient_hessian - average_time_function_only
) / average_time_function_only
relative_to_gradient = (
    average_time_function_gradient_hessian - average_time_function_gradient
) / average_time_function_gradient
print(
    f'Calculating the hessian means a {100*relative_to_function:.2f}% increase of the calculation time compared to '
    f'calculating the function only.'
)
print(
    f'Calculating the hessian means a {100*relative_to_gradient:.2f}% increase of the calculation time compared to '
    f'calculating the function and the gradient only.'
)
