"""

Illustration the quick_estimate of Biogeme
==========================================

Same model as b01logit, estimated using the quick_estimate, that skips the calculation of the second orders statistics.

Michel Bierlaire, EPFL
Wed Jun 18 2025, 11:19:12
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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01logit_ter.py')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)


# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
logprob = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, logprob)
the_biogeme.model_name = 'b01logit_ter'

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculate_null_loglikelihood(av)

# %%
# Estimate the parameters.
results = the_biogeme.quick_estimate()

# %%
print(results.short_summary())

# %%
# Where quick_estimate is called, the initial log likelihood is not calculated. The derivatives of the loglikelihood
# function are not calculated either. It means that several statistics are missing in the report.
# This function is convenient when only the estimated values of the parameters are needed.

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)

# %%
# Get general statistics.
#
print('General statistics')
print('------------------')
stats = results.get_general_statistics()
for description, value in stats.items():
    print(f'{description}: {value}')

# %%
# The YAML file is not automatically generated when quick_estimate is used. It can be done manually if needed.
results.dump_yaml_file(filename=f'{the_biogeme.model_name}.yaml')
