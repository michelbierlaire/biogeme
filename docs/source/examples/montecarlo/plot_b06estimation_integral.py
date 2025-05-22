"""

Estimation of mixtures of logit
===============================

Estimation of a mixtures of logit models where the integral is
calculated using numerical integration.

Michel Bierlaire, EPFL
Tue Apr 29 2025, 12:11:20
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
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation
omega = RandomVariable('omega')
b_time_rnd = B_TIME + B_TIME_S * omega

# %%
# Definition of the utility functions
v1 = ASC_TRAIN + b_time_rnd * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
v2 = ASC_SM + b_time_rnd * SM_TT_SCALED + B_COST * SM_COST_SCALED
v3 = ASC_CAR + b_time_rnd * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives
util = {1: v1, 2: v2, 3: v3}

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
