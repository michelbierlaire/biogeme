"""
.. _plot_b05normal_mixture:

Mixture of logit models
=======================

Example of a normal mixture of logit models, using Monte-Carlo integration.

Michel Bierlaire, EPFL
Wed Jun 18 2025, 11:28:46
"""

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, DistributedParameter, Draws
from biogeme.models import loglogit

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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b05normal_mixtures.py')

# %%
# Parameters to be estimated
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_cost = Beta('b_cost', 0, None, 0, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
b_time = Beta('b_time', 0, None, 0, 0)

# %%
# It is advised *not* to use 0 as starting value for the following parameter.
b_time_s = Beta('b_time_s', 10, 0.000001, None, 0)
b_time_eps = Draws('b_time_eps', 'NORMAL')
b_time_rnd = DistributedParameter('b_time_rnd', b_time + b_time_s * b_time_eps)

# %%
# Definition of the utility functions.
v_train = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# In order to obtain the loglikelihood, we would first calculate the kernek conditional on b_time_rnd,
# and the integrate over b_time_rnd using Monte-Carlo.
# prob = logit(v, av, CHOICE)
# log_likelihood = log(MonteCarlo(prob))

# %%
# However, when performing Bayesian estimation, the random parameters will be explicitly simulated. Therefore,
# what the algorithm needs is the *conditional* log likelihood, which is simply a (log) logit here.
conditional_log_likelihood = loglogit(v, av, CHOICE)

# %%
# These notes will be included as such in the report file.
USER_NOTES = 'Example of a mixture of logit models with three alternatives'

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(
    database,
    conditional_log_likelihood,
    user_notes=USER_NOTES,
)
the_biogeme.model_name = 'b05normal_mixture'

# %%
# Estimate the parameters
bayesian_results = the_biogeme.bayesian_estimation()
