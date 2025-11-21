"""
.. _plot_b15_panel_discrete:

15. Discrete mixture with panel data
====================================

Bayesian estimation of a discrete mixture of logit models, also called latent
class model.  The datafile is organized as panel data.


Michel Bierlaire, EPFL
Sat Nov 15 2025, 17:39:13

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import BayesianResults, get_pandas_estimated_parameters
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Beta,
    DistributedParameter,
    Draws,
    exp,
    log,
)
from biogeme.models import logit
# %%
# See the data processing script: :ref:`swissmetro_panel`.
from swissmetro_panel import (
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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b15_panel_discrete.py')

# %%
# Parameters to be estimated. One version for each latent_old class.
NUMBER_OF_CLASSES = 2
b_cost = [Beta(f'b_cost_class{i}', 0, None, None, 0) for i in range(NUMBER_OF_CLASSES)]

# %%
# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation
b_time = [Beta(f'b_time_class{i}', 0, None, None, 0) for i in range(NUMBER_OF_CLASSES)]

# %%
# It is advised not to use 0 as starting value for the following parameter.
b_time_s = [
    Beta(f'b_time_s_class{i}', 1, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
b_time_rnd = [
    DistributedParameter(
        f'b_time_rnd_class{i}',
        b_time[i] + b_time_s[i] * Draws(f'b_time_eps_class{i}', 'NORMAL'),
    )
    for i in range(NUMBER_OF_CLASSES)
]

# %%
# We do the same for the constants, to address serial correlation.
asc_car = [
    Beta(f'asc_car_class{i}', 0, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_car_s = [
    Beta(f'asc_car_s_class{i}', 1, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_car_rnd = [
    DistributedParameter(
        f'asc_car_rnd_class{i}',
        asc_car[i] + asc_car_s[i] * Draws(f'asc_car_eps_class{i}', 'NORMAL'),
    )
    for i in range(NUMBER_OF_CLASSES)
]

asc_train = [
    Beta(f'asc_train_class{i}', 0, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_train_s = [
    Beta(f'asc_train_s_class{i}', 1, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_train_rnd = [
    DistributedParameter(
        f'asc_train_rnd_class{i}',
        asc_train[i] + asc_train_s[i] * Draws(f'asc_train_eps_class{i}', 'NORMAL'),
    )
    for i in range(NUMBER_OF_CLASSES)
]

asc_sm = [Beta(f'asc_sm_class{i}', 0, None, None, 1) for i in range(NUMBER_OF_CLASSES)]
asc_sm_s = [
    Beta(f'asc_sm_s_class{i}', 1, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_sm_rnd = [
    DistributedParameter(
        f'asc_sm_rnd_class{i}',
        asc_sm[i] + asc_sm_s[i] * Draws(f'asc_sm_eps_class{i}', 'NORMAL'),
    )
    for i in range(NUMBER_OF_CLASSES)
]

# %%
# Class membership probability. Note: for Bayesian estimation, this should not call the logit model.
score_class_0 = Beta('score_class_0', -1.7, None, None, 0)
probability_class_1 = 1 / (1 + exp(score_class_0))
probability_class_0 = 1 - probability_class_1

# %%
# In class 0, it is assumed that the time coefficient is zero.
b_time_rnd[0] = 0

# %%
# Utility functions.
v_train_per_class = [
    asc_train_rnd[i] + b_time_rnd[i] * TRAIN_TT_SCALED + b_cost[i] * TRAIN_COST_SCALED
    for i in range(NUMBER_OF_CLASSES)
]
v_swissmetro_per_class = [
    asc_sm_rnd[i] + b_time_rnd[i] * SM_TT_SCALED + b_cost[i] * SM_COST_SCALED
    for i in range(NUMBER_OF_CLASSES)
]
v_car_per_class = [
    asc_car_rnd[i] + b_time_rnd[i] * CAR_TT_SCALED + b_cost[i] * CAR_CO_SCALED
    for i in range(NUMBER_OF_CLASSES)
]
v_per_class = [
    {1: v_train_per_class[i], 2: v_swissmetro_per_class[i], 3: v_car_per_class[i]}
    for i in range(NUMBER_OF_CLASSES)
]

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# The choice model is a discrete mixture of logit, with availability conditions
# We calculate the conditional probability for each class.
conditional_probability_per_class = [
    logit(v_per_class[i], av, CHOICE) for i in range(NUMBER_OF_CLASSES)
]

# %%
# Conditional to the random variables, likelihood for the individual.
conditional_choice_probability = (
    probability_class_0 * conditional_probability_per_class[0]
    + probability_class_1 * conditional_probability_per_class[1]
)

# %%
# We need the log probability per observation
conditional_log_probability = log(conditional_choice_probability)

# %%
the_biogeme = BIOGEME(
    database,
    conditional_log_probability,
    warmup=4000,
    bayesian_draws=4000,
    chains=8,
)
the_biogeme.model_name = 'b15_panel_discrete'

# %%
# Estimate the parameters.
try:
    results = BayesianResults.from_netcdf(
        filename=f'saved_results/{the_biogeme.model_name}.nc'
    )
except FileNotFoundError:
    results = the_biogeme.bayesian_estimation()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
