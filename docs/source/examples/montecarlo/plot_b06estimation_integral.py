"""

Estimation of mixtures of logit
===============================

Estimation of a mixtures of logit models where the integral is
calculated using numerical integration.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 21:12:42
"""

from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, IntegrateNormal, RandomVariable, log
from biogeme.models import logit
from biogeme.results_processing import get_pandas_estimated_parameters

from swissmetro import (
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
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_time_s = Beta('b_time_s', 1, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
omega = RandomVariable('omega')
b_time_rnd = b_time + b_time_s * omega

# %%
# Definition of the utility functions
v_train = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
util = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# The choice model is a logit, with availability conditions
cond_prob = logit(util, av, CHOICE)
prob = IntegrateNormal(cond_prob, 'omega')
log_prob = log(prob)

# %%
the_biogeme = BIOGEME(database, log_prob)
the_biogeme.model_name = '06estimation_integral'

# %%
results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
