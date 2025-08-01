"""

Estimation of a logit model
===========================

 Three alternatives:

   - train,
   - car and,
   - Swissmetro.

 Stated preferences data.

Michel Bierlaire, EPFL
Wed Jun 18 2025, 09:56:59
"""

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

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

# %%
# The logger sets the verbosity of Biogeme. By default, Biogem eis quite silent and generates only warnings.
# To have more information about what it happening behind the scene, the level should be set to `blog.INFO`.
logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01logit_bis.py')


# %%
# Parameters to be estimated: alternative specific constants
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)

# %%
# The constant associated with Swissmetro is normalized to zero. It does not need to be defined at all.
# Here, we illustrate the fact that setting the last argument of the `Beta` function to 1 fixes the parameter
# to its default value (here, 0).
asc_sm = Beta('asc_sm', 0, None, None, 1)

# %%
# Coefficients of the attributes
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)


# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_sm = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_sm, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b01logit'

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculate_null_loglikelihood(av)

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
