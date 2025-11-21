"""
.. _plot_b05_normal_mixture:

5. Mixture of logit models: normal distribution
===============================================

Example of a normal mixture of logit models, using Bayesian inference.

Michel Bierlaire, EPFL
Thu Nov 20 2025, 11:26:01
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import get_pandas_estimated_parameters
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
logger.info('Example b05_normal_mixtures.py')

# %%
# The scale parameters must stay away from zero. We define a small but positive lower bound
POSITIVE_LOWER_BOUND = 1.0e-5

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
b_time_s = Beta('b_time_s', 10, POSITIVE_LOWER_BOUND, None, 0)
b_time_eps = Draws('b_time_eps', 'NORMAL')

# %%
# The purpose of the `DistributedParameter` operator is to explicitly store the simulated individual-level parameters
# in the output file.
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
# When performing maximum likelihood estimation, in order to obtain the loglikelihood, we would first calculate the
# kernel conditional on b_time_rnd, and then integrate over b_time_rnd using Monte-Carlo.
# However, when performing Bayesian estimation, the random parameters will be explicitly simulated. Therefore,
# what the algorithm needs is the *conditional* log likelihood, which is simply a (log) logit here.
# This is one of the most important advantage of this estimation method: it does not require to calculate the
# complicated integrals.
conditional_log_likelihood = loglogit(v, av, CHOICE)

# %%
# These notes will be included as such in the report file.
USER_NOTES = 'Example of Bayesian estimation of a mixture of logit models with three alternatives'

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(
    database,
    conditional_log_likelihood,
    user_notes=USER_NOTES,
)
the_biogeme.model_name = 'b05_normal_mixture'

# %%
# Estimate the parameters
bayesian_results = the_biogeme.bayesian_estimation()

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=bayesian_results,
)
display(pandas_results)
