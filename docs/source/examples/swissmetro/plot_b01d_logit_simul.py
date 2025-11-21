"""

1d. Simulation of a logit model
===============================

Example of simulation with a logit model


Michel Bierlaire, EPFL
Wed Jun 18 2025, 11:05:50
"""

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Derive
from biogeme.models import logit
from biogeme.results_processing import EstimationResults

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
prob_swissmetro = logit(v, av, 2)
prob_car = logit(v, av, 3)

# %%
# Elasticities.
#
# Elasticities can be computed. We illustrate below two
# formulas. Check in the output file that they produce the same
# results.

# %%
# First, the general definition of elasticities. This illustrates the
# use of the Derive expression, and can be used with any model,
# however complicated it is. Note the quotes in the Derive operator.

general_time_elasticity_train = Derive(prob_train, 'TRAIN_TT') * TRAIN_TT / prob_train
general_time_elasticity_swissmetro = (
    Derive(prob_swissmetro, 'SM_TT') * SM_TT / prob_swissmetro
)
general_time_elasticity_car = Derive(prob_car, 'CAR_TT') * CAR_TT / prob_car

# %%
# Second, the elasticity of logit models. See Ben-Akiva and Lerman for
# the formula

logit_time_elasticity_train = TRAIN_AV_SP * (1.0 - prob_train) * TRAIN_TT * b_time / 100
logit_time_elasticity_swissmetro = (
    SM_AV * (1.0 - prob_swissmetro) * SM_TT * b_time / 100
)
logit_time_elasticity_car = CAR_AV_SP * (1.0 - prob_car) * CAR_TT * b_time / 100

# %%
# Quantities to be simulated.
#
simulate = {
    'Prob. train': prob_train,
    'Prob. Swissmetro': prob_swissmetro,
    'Prob. car': prob_car,
    'logit elas. 1': logit_time_elasticity_train,
    'generic elas. 1': general_time_elasticity_train,
    'logit elas. 2': logit_time_elasticity_swissmetro,
    'generic elas. 2': general_time_elasticity_swissmetro,
    'logit elas. 3': logit_time_elasticity_car,
    'generic elas. 3': general_time_elasticity_car,
}


# %%
# Create the Biogeme object.
#
# As we simulate the probability for all alternatives, even when one of
# them is not available, Biogeme may trigger some warnings.
biosim = BIOGEME(database, simulate)
biosim.model_name = 'b01d_logit_simul'

# %%
# Retrieve the estimated values of the parameters.
RESULTS_FILE_NAME = 'saved_results/b01logit.yaml'
estimation_results = EstimationResults.from_yaml_file(filename=RESULTS_FILE_NAME)
betas = estimation_results.get_beta_values()


# %%
# Simulation
#
results = biosim.simulate(the_beta_values=betas)
display(results.describe())
