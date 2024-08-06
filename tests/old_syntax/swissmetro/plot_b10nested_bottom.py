"""

Nested logit model normalized from bottom
=========================================

Example of a nested logit model where the normalization is done at the
 bottom level.

:author: Michel Bierlaire, EPFL
:date: Tue Oct 24 13:40:46 2023

"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit

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
logger.info('Example b10nested_bottom.py')

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# This is the scale parameter of the choice model. It is usually
# normalized to one. In this example, we normalize the nest parameter
# instead, and estimate the scale parameter for the model.  If the
# lower bound is set to zero, the model cannot be evaluated.
# Therefore, we set the lower bound to a small number, strictly larger
# than zero.
MU = Beta('MU', 0.5, 0.000001, 1.0, 0)

# %%
# Definition of the utility functions
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of nests. Only the non trival nests must be defined. A
# trivial nest is a nest containing exactly one alternative.

existing = OneNestForNestedLogit(
    nest_param=1.0, list_of_alternatives=[1, 3], name='existing'
)

nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(existing,))

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
# The choice model is a nested logit, with availability conditions,
# where the scale parameter mu is explicitly involved.
logprob = models.lognestedMevMu(V, av, nests, CHOICE, MU)

# %%
# Create the Biogeme object.
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'b10nested_bottom'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = results.getEstimatedParameters()
pandas_results
