"""File 12panel_p.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 15 11:26:49 2020

 Example of a mixture of logit models, using Monte-Carlo integration.
 The datafile is organized as panel data.
 Three alternatives: Train, Car and Swissmetro
 SP data

The Swissmetro data is organized such that each row contains all the
responses of one individual.

"""
import biogeme.logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta,
    Variable,
    bioMultSum,
    bioDraws,
    MonteCarlo,
    log,
    exp,
)

from swissmetro import (
    database,
    NBR_QUESTIONS,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    TRAIN_AV_SP,
    SM_AV,
    CAR_AV_SP,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b12panel_p.py')

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, 0, None, 0)

# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND', 'NORMAL')

# Definition of the utility functions
V1 = [
    ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED[q] + B_COST * TRAIN_COST_SCALED[q]
    for q in range(NBR_QUESTIONS)
]
V2 = [
    ASC_SM + B_TIME_RND * SM_TT_SCALED[q] + B_COST * SM_COST_SCALED[q]
    for q in range(NBR_QUESTIONS)
]
V3 = [
    ASC_CAR + B_TIME_RND * CAR_TT_SCALED[q] + B_COST * CAR_CO_SCALED[q]
    for q in range(NBR_QUESTIONS)
]

# Associate utility functions with the numbering of alternatives
V = [{1: V1[q], 2: V2[q], 3: V3[q]} for q in range(NBR_QUESTIONS)]

av = [{1: TRAIN_AV_SP[q], 2: SM_AV[q], 3: CAR_AV_SP[q]} for q in range(NBR_QUESTIONS)]


# Conditional to B_TIME_RND, the likelihood of one observation is
# given by the logit model (called the kernel)
obslogprob = [
    models.loglogit(V[q], av[q], Variable(f'CHOICE_{q}')) for q in range(NBR_QUESTIONS)
]

# Conditional to B_TIME_RND, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
condprobIndiv = exp(bioMultSum(obslogprob))

# We integrate over B_TIME_RND using Monte-Carlo
logprob = log(MonteCarlo(condprobIndiv))

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '12panel_p'

# Estimate the parameters.
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)
