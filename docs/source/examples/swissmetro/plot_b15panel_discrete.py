"""
.. _plot_b15panel_discrete:

Discrete mixture with panel data
================================

Example of a discrete mixture of logit models, also called latent_old
 class model.  The datafile is organized as panel data.

Michel Bierlaire, EPFL
Sat Jun 21 2025, 17:16:14

"""

from IPython.core.display_functions import display

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

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Beta,
    Draws,
    ExpressionOrNumeric,
    MonteCarlo,
    PanelLikelihoodTrajectory,
    log,
)
from biogeme.models import logit
from biogeme.results_processing import get_pandas_estimated_parameters

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b15panel_discrete.py')

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
b_time_rnd: list[ExpressionOrNumeric] = [
    b_time[i] + b_time_s[i] * Draws(f'b_time_rnd_class{i}', 'NORMAL_ANTI')
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
    asc_car[i] + asc_car_s[i] * Draws(f'asc_car_rnd_class{i}', 'NORMAL_ANTI')
    for i in range(NUMBER_OF_CLASSES)
]

asc_train = [
    Beta(f'asc_train_class{i}', 0, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_train_s = [
    Beta(f'asc_train_s_class{i}', 1, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_train_rnd = [
    asc_train[i] + asc_train_s[i] * Draws(f'asc_train_rnd_class{i}', 'NORMAL_ANTI')
    for i in range(NUMBER_OF_CLASSES)
]

asc_sm = [Beta(f'asc_sm_class{i}', 0, None, None, 1) for i in range(NUMBER_OF_CLASSES)]
asc_sm_s = [
    Beta(f'asc_sm_s_class{i}', 1, None, None, 0) for i in range(NUMBER_OF_CLASSES)
]
asc_sm_rnd = [
    asc_sm[i] + asc_sm_s[i] * Draws(f'asc_sm_rnd_class{i}', 'NORMAL_ANTI')
    for i in range(NUMBER_OF_CLASSES)
]

# %%
# Class membership probability.
score_class_0 = Beta('score_class_0', -1.7, None, None, 0)
probability_class_0 = logit({0: score_class_0, 1: 0}, None, 0)
probability_class_1 = logit({0: score_class_0, 1: 0}, None, 1)

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
choice_probability_per_class = [
    PanelLikelihoodTrajectory(logit(v_per_class[i], av, CHOICE))
    for i in range(NUMBER_OF_CLASSES)
]

# %%
# Conditional to the random variables, likelihood for the individual.
choice_probability = (
    probability_class_0 * choice_probability_per_class[0]
    + probability_class_1 * choice_probability_per_class[1]
)

# %%
# We integrate over the random variables using Monte-Carlo.
log_probability = log(MonteCarlo(choice_probability))

# %%
# The model is complex, and there are numerical issues when calculating the second derivatives. Therefore,
# we instruct Biogeme not to evaluate the second derivatives. As a consequence, the statistics reported after
# estimation are based on the BHHH matrix instead of the Rao-Cramer bound.
the_biogeme = BIOGEME(
    database,
    log_probability,
    number_of_draws=10_000,
    calculating_second_derivatives='never',
    seed=1223,
)
the_biogeme.model_name = 'b15panel_discrete'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
