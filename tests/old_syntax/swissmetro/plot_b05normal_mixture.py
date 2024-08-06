"""
.. _plot_b05normal_mixture:

Mixture of logit models
=======================

Example of a normal mixture of logit models, using Monte-Carlo integration.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:30:14 2023
"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, log, MonteCarlo

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

logger = blog.get_screen_logger(level=blog.DEBUG)
logger.info('Example b05normal_mixtures.py')

# %%
# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
B_TIME = Beta('B_TIME', 0, None, None, 0)

# %%
# It is advised *not* to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional to B_TIME_RND, we have a logit model (called the kernel).
prob = models.logit(V, av, CHOICE)

# %%
# We integrate over B_TIME_RND using Monte-Carlo.
logprob = log(MonteCarlo(prob))

# %%
# These notes will be included as such in the report file.
USER_NOTES = (
    'Example of a mixture of logit models with three alternatives, '
    'approximated using Monte-Carlo integration.'
)

# %%
# Create the Biogeme object. As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws. To achieve that, we provide a parameter file
# different from the default one.
the_biogeme = bio.BIOGEME(
    database, logprob, userNotes=USER_NOTES, parameter_file='few_draws.toml'
)
the_biogeme.modelName = 'b05normal_mixture'

# %%
print(f'Number of draws: {the_biogeme.number_of_draws}')

# %%
# Estimate the parameters
results = the_biogeme.estimate()
# %%
print(results.short_summary())
# %%
pandas_results = results.getEstimatedParameters()
pandas_results
