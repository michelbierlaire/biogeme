"""
.. _plot_b17lognormal_mixture:

Mixture with lognormal distribution
===================================

Example of a mixture of logit models, using Monte-Carlo integration.
The mixing distribution is distributed as a log normal.

:author: Michel Bierlaire, EPFL
:date: Mon Apr 10 12:11:53 2023

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Beta,
    exp,
    log,
    MonteCarlo,
    bioDraws,
)
from biogeme.models import logit
from biogeme.results_processing import get_pandas_estimated_parameters

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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b17lognormal_mixture.py')

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
B_TIME = Beta('B_TIME', 0, None, None, 0)

# %%
# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, -2, 2, 0)

# %%
# Define a random parameter, log normally distributed, designed to be used
# for Monte-Carlo simulation.
B_TIME_RND = -exp(B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL'))

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
# Conditional to b_time_rnd, we have a logit model (called the kernel).
prob = logit(V, av, CHOICE)

# %%
# We integrate over b_time_rnd using Monte-Carlo.
logprob = log(MonteCarlo(prob))

# %%
# As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws.
the_biogeme = BIOGEME(database, logprob, number_of_draws=100, seed=1223)
the_biogeme.modelName = '17lognormal_mixture'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
