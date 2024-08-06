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
import biogeme.biogeme as bio
from biogeme import models
import biogeme.biogeme_logging as blog
from biogeme.expressions import (
    Beta,
    bioDraws,
    MonteCarlo,
    PanelLikelihoodTrajectory,
    log,
)

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
    See the numpy.random documentation to obtain a list of other distributions.
    """
    return np.random.triangular(-1, 0, 1, (sample_size, number_of_draws))


# %%
# Associate the function with a name.
myRandomNumberGenerators = {
    'TRIANGULAR': (
        the_triangular_generator,
        'Draws from a triangular distribution',
    )
}

# %%
# Submit the generator to the database.
database.setRandomNumberGenerators(myRandomNumberGenerators)

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
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'TRIANGULAR')

# %%
# We do the same for the constants, to address serial correlation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR_S = Beta('ASC_CAR_S', 1, None, None, 0)
ASC_CAR_RND = ASC_CAR + ASC_CAR_S * bioDraws('ASC_CAR_RND', 'TRIANGULAR')

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_TRAIN_S = Beta('ASC_TRAIN_S', 1, None, None, 0)
ASC_TRAIN_RND = ASC_TRAIN + ASC_TRAIN_S * bioDraws('ASC_TRAIN_RND', 'TRIANGULAR')

ASC_SM = Beta('ASC_SM', 0, None, None, 1)
ASC_SM_S = Beta('ASC_SM_S', 1, None, None, 0)
ASC_SM_RND = ASC_SM + ASC_SM_S * bioDraws('ASC_SM_RND', 'TRIANGULAR')

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
obsprob = models.logit(V, av, CHOICE)

# %%
# Conditional on the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
condprobIndiv = PanelLikelihoodTrajectory(obsprob)

# %%
# We integrate over the random parameters using Monte-Carlo
logprob = log(MonteCarlo(condprobIndiv))

# %%
# Create the Biogeme object. As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws. To achieve that, we provide a parameter file
# different from the default one.
the_biogeme = bio.BIOGEME(database, logprob, parameter_file='few_draws.toml')
the_biogeme.modelName = 'b26triangular_panel_mixture'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
