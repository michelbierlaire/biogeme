"""File 18ordinalLogit.py

:author: Michel Bierlaire, EPFL
:date: Mon Sep  9 08:08:40 2019

 Example of an ordinal logit model.
 This is just to illustrate the syntax, as the data are not ordered.
 But the example assume, for the sake of it, that they are 1->2->3
 Three alternatives: Train, Car and Swissmetro
 SP data
"""

import biogeme.biogeme as bio
import biogeme.distributions as dist
import biogeme.messaging as msg
from biogeme.expressions import Beta, log, Elem
from swissmetro import (
    database,
    CHOICE,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
)


# Parameters to be estimated
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Parameters for the ordered logit.
# tau1 <= 0
tau1 = Beta('tau1', -1, None, 0, 0)
# delta2 >= 0
delta2 = Beta('delta2', 2, 0, None, 0)
tau2 = tau1 + delta2

#  Utility
U = B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED

# Associate each discrete indicator with an interval.
#   1: -infinity -> tau1
#   2: tau1 -> tau2
#   3: tau2 -> +infinity

ChoiceProba = {
    1: 1 - dist.logisticcdf(U - tau1),
    2: dist.logisticcdf(U - tau1) - dist.logisticcdf(U - tau2),
    3: dist.logisticcdf(U - tau2),
}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = log(Elem(ChoiceProba, CHOICE))

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '18ordinalLogit'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)
