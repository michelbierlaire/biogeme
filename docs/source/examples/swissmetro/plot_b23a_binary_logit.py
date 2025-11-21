"""

23a. Binary logit model
=======================

Example of a binary logit model.
Two alternatives: Train and Car.
All observations such that the Swissmetro was chosen haven been removed from the sample.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 12:42:27

"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

# %%
# See the data processing script: :ref:`swissmetro_binary`.
from swissmetro_binary import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
b_time_car = Beta('b_time_car', 0, None, None, 0)
b_time_train = Beta('b_time_train', 0, None, None, 0)
b_cost_car = Beta('b_cost_car', 0, None, None, 0)
b_cost_train = Beta('b_cost_train', 0, None, None, 0)

# %%
# Definition of the utility functions.
# We estimate a binary logit model. There are only two alternatives.
v_train = b_time_train * TRAIN_TT_SCALED + b_cost_train * TRAIN_COST_SCALED
v_car = asc_car + b_time_car * CAR_TT_SCALED + b_cost_car * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b23a_logit'

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
