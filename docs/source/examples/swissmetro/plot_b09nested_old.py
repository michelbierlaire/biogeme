"""

Nested logit model
==================

Example of a nested logit model, using the original syntax for nests.
Since biogeme 3.13, a new syntax, more explicit, has been adopted.

:author: Michel Bierlaire, EPFL
:date: Tue Oct 24 13:37:27 2023

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import lognested
from biogeme.results_processing import get_pandas_estimated_parameters

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
logger.info('Example b09nested')

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
MU = Beta('MU', 1, 1, 10, 0)

# %%
# Definition of the utility functions.
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
# Definition of nests. In this example, we create a nest for the
# existing modes, that is train (1) and car (3).  Each nest is
# associated with a tuple containing (i) the nest parameter and (ii)
# the list of alternatives.
existing = MU, [1, 3]
future = 1.0, [2]
nests = existing, future

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
# The choice model is a nested logit, with availability conditions.
logprob = lognested(V, av, nests, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, logprob)
the_biogeme.modelName = "b09nested_old"

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculate_null_loglikelihood(av)

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
