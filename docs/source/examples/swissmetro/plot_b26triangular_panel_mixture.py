"""

Triangular mixture with panel data
==================================

 Example of a mixture of logit models, using Monte-Carlo integration.
 The mixing distribution is user-defined (triangular, here).
 The datafile is organized as panel data.

:author: Michel Bierlaire, EPFL
:date: Tue Dec  6 18:30:44 2022

"""

import numpy as np
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Beta,
    Draws,
    MonteCarlo,
    PanelLikelihoodTrajectory,
    log,
)
from biogeme.models import logit
from biogeme.draws import RandomNumberGeneratorTuple
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# See the data processing script: :ref:`swissmetro_panel`.
from swissmetro_panel import (
    database,
    CHOICE,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    SM_AV,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b26triangular_panel_mixture.py')


# %%
# Function generating the draws.
def the_triangular_generator(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Provide my own random number generator to the database.
    See the `numpy.random` documentation to obtain a list of other distributions.
    """
    return np.random.triangular(-1, 0, 1, (sample_size, number_of_draws))


# %%
# Associate the function with a name.
my_random_number_generators = {
    'TRIANGULAR': RandomNumberGeneratorTuple(
        the_triangular_generator,
        'Draws from a triangular distribution',
    )
}

# %%
# Parameters to be estimated.
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation.

# %%
# Mean of the distribution.
B_TIME = Beta('B_TIME', 0, None, None, 0)

# %%
# Scale of the distribution. It is advised not to use 0 as starting
# value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * Draws('b_time_rnd', 'TRIANGULAR')

# %%
# We do the same for the constants, to address serial correlation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR_S = Beta('ASC_CAR_S', 1, None, None, 0)
ASC_CAR_RND = ASC_CAR + ASC_CAR_S * Draws('ASC_CAR_RND', 'TRIANGULAR')

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_TRAIN_S = Beta('ASC_TRAIN_S', 1, None, None, 0)
ASC_TRAIN_RND = ASC_TRAIN + ASC_TRAIN_S * Draws('ASC_TRAIN_RND', 'TRIANGULAR')

ASC_SM = Beta('ASC_SM', 0, None, None, 1)
ASC_SM_S = Beta('ASC_SM_S', 1, None, None, 0)
ASC_SM_RND = ASC_SM + ASC_SM_S * Draws('ASC_SM_RND', 'TRIANGULAR')

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN_RND + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM_RND + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR_RND + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional to the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel).
obs_prob = logit(V, av, CHOICE)

# %%
# Conditional on the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
condprob_indiv = PanelLikelihoodTrajectory(obs_prob)

# %%
# We integrate over the random parameters using Monte-Carlo
logprob = log(MonteCarlo(condprob_indiv))

# %%
# As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws.
the_biogeme = BIOGEME(
    database,
    logprob,
    random_number_generators=my_random_number_generators,
    number_of_draws=1000,
    seed=1223,
)
the_biogeme.model_name = 'b26triangular_panel_mixture'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
