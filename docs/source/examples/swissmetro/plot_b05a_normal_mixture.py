"""
.. _plot_b05normal_mixture:

5a. Mixture of logit models with Monte-Carlo integration
========================================================

Example of a normal mixture of logit models, using Monte-Carlo integration.

Michel Bierlaire, EPFL
Wed Jun 18 2025, 11:28:46
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Draws, MonteCarlo, log
from biogeme.models import logit
from biogeme.results_processing import get_pandas_estimated_parameters

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
logger.info('Example b05a_normal_mixture.py')

# %%
# Parameters to be estimated
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
b_time = Beta('b_time', 0, None, None, 0)

# %%
# It is advised *not* to use 0 as starting value for the following parameter.
b_time_s = Beta('b_time_s', 1, None, None, 0)
b_time_rnd = b_time + b_time_s * Draws('b_time_rnd', 'NORMAL')

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
# Conditional to b_time_rnd, we have a logit model (called the kernel).
prob = logit(v, av, CHOICE)

# %%
# We integrate over b_time_rnd using Monte-Carlo.
logprob = log(MonteCarlo(prob))

# %%
# These notes will be included as such in the report file.
USER_NOTES = (
    'Example of a mixture of logit models with three alternatives, '
    'approximated using Monte-Carlo integration.'
)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(
    database, logprob, user_notes=USER_NOTES, number_of_draws=10_000, seed=1223
)
the_biogeme.model_name = 'b05a_normal_mixture'

# %%
print(f'Number of draws: {the_biogeme.number_of_draws:_}')

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
