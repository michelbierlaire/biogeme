"""File b02weight.py

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 17:02:59 2023

 Example of a logit model with Weighted Exogenous Sample Maximum
 Likelihood (WESML).
"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta

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
    GROUP,
)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Definition of the weight
weight = 8.890991e-01 * (1.0 * (GROUP == 2) + 1.2 * (GROUP == 3))

# These notes will be included as such in the report file.
USER_NOTES = (
    'Example of a logit model with three alternatives: '
    'Train, Car and Swissmetro.'
    ' Weighted Exogenous Sample Maximum Likelihood estimator (WESML)'
)


# Create the Biogeme object
formulas = {'loglike': logprob, 'weight': weight}
the_biogeme = bio.BIOGEME(database, formulas, userNotes=USER_NOTES)
the_biogeme.modelName = 'b02weight'

# It is possible to control the generation of the HTML and the pickle
# files Note that these parameters can also be modified in the .TOML
# configuration file
the_biogeme.generate_html = True
the_biogeme.generate_pickle = False
results = the_biogeme.estimate()
print(results.short_summary())

# Get the results in a pandas table
pandas_results = results.getEstimatedParameters()
print(pandas_results)
