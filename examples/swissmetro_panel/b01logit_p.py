"""File b01logit_p.py

:author: Michel Bierlaire, EPFL
:date: Fri Apr 14 14:17:32 2023

 Example of a logit model.
The Swissmetro data is organized such that each row contains all the
responses of one individual.

"""
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioMultSum, Variable

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

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)


# Definition of the utility functions
V1 = [
    ASC_TRAIN + B_TIME * TRAIN_TT_SCALED[q] + B_COST * TRAIN_COST_SCALED[q]
    for q in range(NBR_QUESTIONS)
]
V2 = [
    ASC_SM + B_TIME * SM_TT_SCALED[q] + B_COST * SM_COST_SCALED[q]
    for q in range(NBR_QUESTIONS)
]
V3 = [
    ASC_CAR + B_TIME * CAR_TT_SCALED[q] + B_COST * CAR_CO_SCALED[q]
    for q in range(NBR_QUESTIONS)
]

# Associate utility functions with the numbering of alternatives
V = [{1: V1[q], 2: V2[q], 3: V3[q]} for q in range(NBR_QUESTIONS)]

# Associate the availability conditions with the alternatives
av = [{1: TRAIN_AV_SP[q], 2: SM_AV[q], 3: CAR_AV_SP[q]} for q in range(NBR_QUESTIONS)]

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = [
    models.loglogit(V[q], av[q], Variable(f'CHOICE_{q}')) for q in range(NBR_QUESTIONS)
]

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, bioMultSum(logprob))
the_biogeme.modelName = '01logit_p'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.shortSummary())
# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
print(pandas_results)
