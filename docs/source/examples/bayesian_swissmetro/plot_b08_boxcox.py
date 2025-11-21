"""

8. Box-Cox transforms
=====================

Bayesian estimation of a logit model, with a Box-Cox transform of variables.

Michel Bierlaire, EPFL
Mon Nov 03 2025, 13:41:40
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import get_pandas_estimated_parameters
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import boxcox, loglogit
# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
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
    database,
)

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, 0, 0)
b_cost = Beta('b_cost', 0, None, 0, 0)
boxcox_parameter = Beta('boxcox_parameter', 1, -10, 10, 0)

# %%
# Definition of the utility functions.
v_train = (
    asc_train
    + b_time * boxcox(TRAIN_TT_SCALED, boxcox_parameter)
    + b_cost * TRAIN_COST_SCALED
)
v_swissmetro = (
    asc_sm + b_time * boxcox(SM_TT_SCALED, boxcox_parameter) + b_cost * SM_COST_SCALED
)
v_car = (
    asc_car + b_time * boxcox(CAR_TT_SCALED, boxcox_parameter) + b_cost * CAR_CO_SCALED
)

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability, bayesian_draws=10000, warmup=10000)
the_biogeme.model_name = 'b08_boxcox'

# %%
# Estimate the parameters
results = the_biogeme.bayesian_estimation()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
