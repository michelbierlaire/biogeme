"""

Illustration the quick_estimate of Biogeme
==========================================

Same model as b01logit, estimated using the quick_estimate, that skips the calculation of the second orders statistics.

Michel Bierlaire, EPFL
Mon Apr 21 2025, 10:43:38
"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, LinearTermTuple, LinearUtility
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters
from biogeme.segmentation import Segmentation
import biogeme.biogeme_logging as blog

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    database,
    CHOICE,
    GA,
    CAR_AV_SP,
    TRAIN_AV_SP,
    TRAIN_TT_SCALED,
    TRAIN_COST_SCALED,
    SM_TT_SCALED,
    SM_COST_SCALED,
    CAR_TT_SCALED,
    CAR_CO_SCALED,
    MALE,
    SM_AV,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b01logit_ter.py')

# %%
# Parameters to be estimated.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)


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
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
logprob = loglogit(V, av, CHOICE)

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
# The YAML file is not automatically generated when quick_estimate is used.
results.dump_yaml_file(filename=f'{the_biogeme.model_name}.yaml')
