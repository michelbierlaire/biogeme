"""

1c. Simulation of a logit model (traditional and Bayesian)
==========================================================

Example of simulation with a logit model

Michel Bierlaire, EPFL
Thu Oct 30 2025, 14:03:15
"""

import pandas as pd
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import BayesianResults
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Derive
from biogeme.models import logit

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT,
    SM_AV,
    SM_COST_SCALED,
    SM_TT,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT,
    database,
)

logger = blog.get_screen_logger(level=blog.INFO)

# %%
# Parameters.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Definition of the utility functions. As we will calculate the derivative with respect to TRAIN_TT, SM_TT and CAR_TT,
# they must explicitly appear in the model. If not, the derivative will be zero. Therefore, we do not use the
# `_SCALED` version of the attributes. We explicitly include their definition.
v_train = asc_train + b_time * TRAIN_TT / 100 + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT / 100 + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT / 100 + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Choice probability.
#
prob_train = logit(v, av, 1)

# %%
# Elasticity.
#
time_elasticity_train = Derive(prob_train, 'TRAIN_TT') * TRAIN_TT / prob_train

# %%
# Quantities to be simulated.
#
simulate = {
    'Prob. train': prob_train,
    'train time elasticity': time_elasticity_train,
    'Value of time': b_time / b_cost,
}


# %%
# Create the Biogeme object.
#
# As we simulate the probability for all alternatives, even when one of
# them is not available, Biogeme may trigger some warnings.
biosim = BIOGEME(database, simulate)
biosim.model_name = 'b01c_logit_simul'

# %%
# Retrieve the estimated values of the parameters.
RESULTS_FILE_NAME = 'saved_results/b01a_logit.nc'
estimation_results = BayesianResults.from_netcdf(filename=RESULTS_FILE_NAME)
betas = estimation_results.get_beta_values()


# %%
# Simulation using the posterior mean of each parameter

print('Simulation using the posterior mean of each parameter')
results = biosim.simulate(the_beta_values=betas)
display(results)

# %%
# Bayesian simulation using the posterior draws
#
print('Bayesian simulation')
bayesian_results = biosim.simulate_bayesian(
    bayesian_estimation_results=estimation_results, percentage_of_draws_to_use=3
)
with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
    display(bayesian_results)
