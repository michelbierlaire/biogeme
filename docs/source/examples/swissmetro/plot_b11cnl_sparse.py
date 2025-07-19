"""

Cross-nested logit
==================

 Example of a cross-nested logit model with two nests:

 - one with existing alternatives (car and train),
 - one with public transportation alternatives (train and Swissmetro)

This illustrates the possibility to ignore all membership parameters that are 0.

Michel Bierlaire, EPFL
Sat Jun 21 2025, 16:50:19

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import logcnl
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit
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
logger.info('Example b11cnl.py')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %% Nest parameters.
existing_nest_parameter = Beta('existing_nest_parameter', 1, 1, 5, 0)
public_nest_parameter = Beta('public_nest_parameter', 1, 1, 5, 0)

# %%
# Nest membership parameters.
alpha_existing = Beta('alpha_existing', 0.5, 0, 1, 0)
alpha_public = 1 - alpha_existing


# %%
# Definition of the utility functions
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of nests.

# %%
# The parameter for alternative 2 is omitted, which is equivalent to sez it to zero.
nest_existing = OneNestForCrossNestedLogit(
    nest_param=existing_nest_parameter,
    dict_of_alpha={1: alpha_existing, 3: 1.0},
    name='existing',
)

# %%
# The parameter for alternative 3 is omitted, which is equivalent to sez it to zero.
nest_public = OneNestForCrossNestedLogit(
    nest_param=public_nest_parameter,
    dict_of_alpha={1: alpha_public, 2: 1.0},
    name='public',
)

nests = NestsForCrossNestedLogit(
    choice_set=[1, 2, 3], tuple_of_nests=(nest_existing, nest_public)
)

# %%
# The choice model is a cross-nested logit, with availability conditions.
log_probability = logcnl(v, av, nests, CHOICE)

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b11cnl_sparse'

# %%
# Estimate the parameters.
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
