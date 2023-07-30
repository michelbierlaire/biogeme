"""File b23binary_probit.py

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:58:18 2023

 Example of a binary probit model.
 Two alternatives: Train and Car
"""

import biogeme.biogeme as bio
from biogeme.expressions import Beta, bioNormalCdf, Elem, log
from swissmetro_binary import (
    database,
    CHOICE,
    TRAIN_AV_SP,
    CAR_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
B_TIME_CAR = Beta('B_TIME_CAR', 0, None, None, 0)
B_TIME_TRAIN = Beta('B_TIME_TRAIN', 0, None, None, 0)
B_COST_CAR = Beta('B_COST_CAR', 0, None, None, 0)
B_COST_TRAIN = Beta('B_COST_TRAIN', 0, None, None, 0)

# Definition of the utility functions
# We estimate a binary probit model. There are only two alternatives.
V1 = B_TIME_TRAIN * TRAIN_TT_SCALED + B_COST_TRAIN * TRAIN_COST_SCALED
V3 = ASC_CAR + B_TIME_CAR * CAR_TT_SCALED + B_COST_CAR * CAR_CO_SCALED

# Associate choice probability with the numbering of alternatives
# If one alternative is not available, the choice probability of the other one is 1.
logP = {
    1: TRAIN_AV_SP * (CAR_AV_SP * log(bioNormalCdf(V1 - V3) + 1 - CAR_AV_SP)),
    3: CAR_AV_SP * (TRAIN_AV_SP * log(bioNormalCdf(V3 - V1) + 1 - TRAIN_AV_SP)),
}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = Elem(logP, CHOICE)

# Create the Biogeme object
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b23probit'

# Estimate the parameters
results = the_biogeme.estimate()
print(results.short_summary())
pandas_results = results.getEstimatedParameters()
print(pandas_results)
