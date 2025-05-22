"""

Base model
==========

Logit model.

:Michel Bierlaire, EPFL
Sun Apr 27 2025, 15:44:48
"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.data.swissmetro import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    SM_AV,
    SM_COST_SCALED,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    read_data,
)
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = B_TIME * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR + B_TIME * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = loglogit(V, av, CHOICE)

# %%
# Read the data
database = read_data()

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, logprob, generate_html=False, generate_yaml=False)
the_biogeme.model_name = 'b00logit'

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculate_null_loglikelihood(av)

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
