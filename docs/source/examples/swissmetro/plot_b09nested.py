"""

Nested logit model
==================

Example of a nested logit model.

Michel Bierlaire, EPFL
Sat Jun 21 2025, 15:33:00
"""

from IPython.core.display_functions import display
from biogeme import biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import lognested
from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
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
logger.info('Example b09nested')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)
nest_parameter = Beta('nest_parameter', 1, 1, 3, 0)

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
# Definition of nests. Only the non-trivial nests must be defined. A
# trivial nest is a nest containing exactly one alternative.  In this
# example, we create a nest for the existing modes, that is train (1)
# and car (3).

existing = OneNestForNestedLogit(
    nest_param=nest_parameter, list_of_alternatives=[1, 3], name='existing'
)

nests = NestsForNestedLogit(choice_set=list(v), tuple_of_nests=(existing,))

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
# The choice model is a nested logit, with availability conditions.
log_probability = lognested(v, av, nests, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(
    database, log_probability, optimization_algorithm='simple_bounds_BFGS'
)
the_biogeme.modelName = "b09nested"

# %%
# Calculate the null log likelihood for reporting.
the_biogeme.calculate_null_loglikelihood(av)

# %%
# Estimate the parameters
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)

# %%
# We calculate the correlation between the error terms of the
# alternatives.
corr = nests.correlation(
    parameters=results.get_beta_values(),
    alternatives_names={1: 'Train', 2: 'Swissmetro', 3: 'Car'},
)
print(corr)
