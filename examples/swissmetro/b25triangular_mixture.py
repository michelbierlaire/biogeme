"""File b25triangular_mixture.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 18:24:18 2023

 Example of a mixture of logit models, using Monte-Carlo integration.
 The mixing distirbution is specified by the user. Here, a triangular
 distribution.

"""

import numpy as np
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, log, MonteCarlo

from swissmetro_data import (
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
logger.info('Example b25triangular_mixture.py')

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_COST = Beta('B_COST', 0, None, None, 0)

# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
B_TIME = Beta('B_TIME', 0, None, None, 0)

# It is advised not to use 0 as starting value for the following parameter.
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)


def the_triangular_generator(sample_size, number_of_draws):
    """
    User-defined random number generator to the database.
    See the numpy.random documentation to obtain a list of other distributions.
    """
    return np.random.triangular(-1, 0, 1, (sample_size, number_of_draws))


myRandomNumberGenerators = {
    'TRIANGULAR': (
        the_triangular_generator,
        'Draws from a triangular distribution',
    )
}

database.setRandomNumberGenerators(myRandomNumberGenerators)

# Define a random parameter with a triangular distribution, designed to be used
# for Monte-Carlo simulation
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'TRIANGULAR')

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Conditional to B_TIME_RND, we have a logit model (called the kernel)
prob = models.logit(V, av, CHOICE)

# We integrate over B_TIME_RND using Monte-Carlo
logprob = log(MonteCarlo(prob))

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b25triangular_mixture'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.short_summary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
