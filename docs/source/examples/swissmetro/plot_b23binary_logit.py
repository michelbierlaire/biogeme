"""

Binary logit model
==================

Example of a binary logit model.
Two alternatives: Train and Car.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:58:18 2023

"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# See the data processing script: :ref:`swissmetro_binary`.
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

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
B_TIME_CAR = Beta('B_TIME_CAR', 0, None, None, 0)
B_TIME_TRAIN = Beta('B_TIME_TRAIN', 0, None, None, 0)
B_COST_CAR = Beta('B_COST_CAR', 0, None, None, 0)
B_COST_TRAIN = Beta('B_COST_TRAIN', 0, None, None, 0)

# %%
# Definition of the utility functions.
# We estimate a binary logit model. There are only two alternatives.
V1 = B_TIME_TRAIN * TRAIN_TT_SCALED + B_COST_TRAIN * TRAIN_COST_SCALED
V3 = ASC_CAR + B_TIME_CAR * CAR_TT_SCALED + B_COST_CAR * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = loglogit(V, av, CHOICE)

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, logprob)
the_biogeme.modelName = 'b23logit'

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
