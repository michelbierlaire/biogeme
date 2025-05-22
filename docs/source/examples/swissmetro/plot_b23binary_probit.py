"""

Binary probit model
===================

Example of a binary probit model.
Two alternatives: Train and Car.

:author: Michel Bierlaire, EPFL
:date: Wed Apr 12 17:58:18 2023

"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Elem, NormalCdf, log
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# See the data processing script: :ref:`swissmetro_binary`.
from swissmetro_binary import (
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
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
# We estimate a binary probit model. There are only two alternatives.
V1 = B_TIME_TRAIN * TRAIN_TT_SCALED + B_COST_TRAIN * TRAIN_COST_SCALED
V3 = ASC_CAR + B_TIME_CAR * CAR_TT_SCALED + B_COST_CAR * CAR_CO_SCALED

# %%
# Associate choice probability with the numbering of alternatives.
logP = {
    1: log(NormalCdf(V1 - V3)),
    3: log(NormalCdf(V3 - V1)),
}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = Elem(logP, CHOICE)
# logprob = (CHOICE == 1) * logP[1] + (CHOICE == 3) * logP[3]

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, logprob, save_iterations=False)
the_biogeme.model_name = 'b23probit'

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
