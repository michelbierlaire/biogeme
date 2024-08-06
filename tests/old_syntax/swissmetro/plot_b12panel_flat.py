"""

Mixture of logit with panel data
================================

Example of a mixture of logit models, using Monte-Carlo integration.
 The datafile is organized as panel data, but a flat version is
 generated.  It means that each row corresponds to one individuals,
 and contains all observations associated with this individual.


:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 18:14:16 2023

"""

import numpy as np
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta,
    Variable,
    bioDraws,
    MonteCarlo,
    log,
    exp,
    bioMultSum,
)

# %%
# See the data processing script: :ref:`swissmetro_panel`.
from swissmetro_panel import (
    flat_database,
    SM_AV,
    CAR_AV_SP,
    TRAIN_AV_SP,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b12panel_flat.py')

# %%
# We set the seed so that the results are reproducible. This is not necessary in general.
np.random.seed(seed=90267)

# %%
# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
# print(database.data.describe())

# %%
# Parameters to be estimated.
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation.
B_TIME = Beta('B_TIME', 0, None, None, 0)

# %%
# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL_ANTI')

# %%
# We do the same for the constants, to address serial correlation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR_S = Beta('ASC_CAR_S', 1, None, None, 0)
ASC_CAR_RND = ASC_CAR + ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL_ANTI')

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_TRAIN_S = Beta('ASC_TRAIN_S', 1, None, None, 0)
ASC_TRAIN_RND = ASC_TRAIN + ASC_TRAIN_S * bioDraws('ASC_TRAIN_RND', 'NORMAL_ANTI')

ASC_SM = Beta('ASC_SM', 0, None, None, 1)
ASC_SM_S = Beta('ASC_SM_S', 1, None, None, 0)
ASC_SM_RND = ASC_SM + ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL_ANTI')

# %%
# In a flatten database, the names of the variables include the time
# or, here, the number of the question, as a prefix

# %%
# Definition of the utility functions
V1 = [
    ASC_TRAIN_RND
    + B_TIME_RND * Variable(f'{t}_TRAIN_TT_SCALED')
    + B_COST * Variable(f'{t}_TRAIN_COST_SCALED')
    for t in range(1, 10)
]

V2 = [
    ASC_SM_RND
    + B_TIME_RND * Variable(f'{t}_SM_TT_SCALED')
    + B_COST * Variable(f'{t}_SM_COST_SCALED')
    for t in range(1, 10)
]

V3 = [
    ASC_CAR_RND
    + B_TIME_RND * Variable(f'{t}_CAR_TT_SCALED')
    + B_COST * Variable(f'{t}_CAR_CO_SCALED')
    for t in range(1, 10)
]

# %%
# Associate utility functions with the numbering of alternatives.
V = [{1: V1[t], 2: V2[t], 3: V3[t]} for t in range(9)]

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional on the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel). The likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
obsprob = [models.loglogit(V[t], av, Variable(f'{t+1}_CHOICE')) for t in range(9)]
condprobIndiv = exp(bioMultSum(obsprob))

# %%
# We integrate over the random parameters using Monte-Carlo.
logprob = log(MonteCarlo(condprobIndiv))

# %%
# Create the Biogeme object. As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws. To achieve that, we provide a parameter file
# different from the default one: `<few_draws.toml>`_
the_biogeme = bio.BIOGEME(flat_database, logprob, parameter_file='few_draws.toml')
the_biogeme.modelName = 'b12panel_flat'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
