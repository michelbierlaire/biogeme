"""

11. Cross-nested logit
======================

 Bayesian estimation of a cross-nested logit model with two nests:

 - one with existing alternatives (car and train),
 - one with public transportation alternatives (train and Swissmetro)

Michel Bierlaire, EPFL
Mon Nov 03 2025, 20:14:23
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import BayesianResults, get_pandas_estimated_parameters
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import logcnl
from biogeme.nests import NestsForCrossNestedLogit, OneNestForCrossNestedLogit
# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    GA,
    SM_AV,
    SM_COST_SCALED,
    SM_HE,
    SM_TT_SCALED,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_HE,
    TRAIN_TT_SCALED,
    database,
)

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b11_cnl.py')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time_swissmetro = Beta('b_time_swissmetro', 0, None, 0, 0)
b_time_train = Beta('b_time_train', 0, None, 0, 0)
b_time_car = Beta('b_time_car', 0, None, 0, 0)
b_cost = Beta('b_cost', 0, None, 0, 0)
b_headway_swissmetro = Beta('b_headway_swissmetro', 0, None, 0, 0)
b_headway_train = Beta('b_headway_train', 0, None, 0, 0)
ga_train = Beta('ga_train', 0, None, None, 0)
ga_swissmetro = Beta('ga_swissmetro', 0, None, None, 0)

# %% Nest parameters.
existing_nest_parameter = Beta('existing_nest_parameter', 1.05, 1, 3, 0)
public_nest_parameter = Beta('public_nest_parameter', 1.05, 1, 3, 0)

# %%
# Nest membership parameters.
alpha_existing = Beta('alpha_existing', 0.5, 0, 1, 0)
alpha_public = 1 - alpha_existing

# %%
# Definition of the utility functions
v_train = (
    asc_train
    + b_time_train * TRAIN_TT_SCALED
    + b_cost * TRAIN_COST_SCALED
    + b_headway_train * TRAIN_HE
    + ga_train * GA
)
v_swissmetro = (
    asc_sm
    + b_time_swissmetro * SM_TT_SCALED
    + b_cost * SM_COST_SCALED
    + b_headway_swissmetro * SM_HE
    + ga_swissmetro * GA
)
v_car = asc_car + b_time_car * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of nests.

nest_existing = OneNestForCrossNestedLogit(
    nest_param=existing_nest_parameter,
    dict_of_alpha={1: alpha_existing, 2: 0.0, 3: 1.0},
    name='existing',
)

nest_public = OneNestForCrossNestedLogit(
    nest_param=public_nest_parameter,
    dict_of_alpha={1: alpha_public, 2: 1.0, 3: 0.0},
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
the_biogeme = BIOGEME(
    database,
    log_probability,
    chains=4,
    bayesian_draws=40_000,
    warmup=40_000,
)
the_biogeme.model_name = 'b11_cnl'

# %%
already_saved_results = f'saved_results/{the_biogeme.model_name}.nc'
# %%
# Estimate the parameters.
try:
    results = BayesianResults.from_netcdf(filename=already_saved_results)
except FileNotFoundError:
    results = the_biogeme.bayesian_estimation()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
