"""

1d. Simulation of a logit model
===============================

This example illustrates how to simulate choice probabilities and
point elasticities for a logit model using previously estimated
parameters.

Two formulations of direct elasticities are compared:

- the general definition based on symbolic derivatives using
  ``Derive``;
- the closed-form expression available for the logit model.

The example verifies that both formulations produce identical
results.

The script has been tested with Biogeme 3.3.3.

Michel Bierlaire, EPFL
Tue Jun 09 2026
"""

from IPython.core.display_functions import display

# %%
# Load the Swissmetro data set and the variables required for the simulation.
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

from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Derive
from biogeme.models import logit
from biogeme.results_processing import EstimationResults

# %%
# Define the model parameters.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_time = Beta('b_time', 0, None, None, 0)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define the utility functions.
#
# The elasticities are calculated with respect to TRAIN_TT, SM_TT,
# and CAR_TT. Therefore, these variables must appear explicitly in
# the utility functions. Using the scaled travel-time variables would
# hide the original variables from the symbolic differentiation and
# would lead to zero derivatives.
v_train = asc_train + b_time * TRAIN_TT / 100 + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time * SM_TT / 100 + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT / 100 + b_cost * CAR_CO_SCALED

# %%
# Associate each utility function with its alternative identifier.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Define the choice probabilities for each alternative.
#
prob_train = logit(v, av, 1)
prob_swissmetro = logit(v, av, 2)
prob_car = logit(v, av, 3)

# %%
# Define direct elasticities.
#
# Two formulations are implemented and compared in order to verify
# that they produce identical results.

# %%
# General definition based on symbolic differentiation.
#
# This formulation can be applied to any choice model, regardless of
# its complexity. The variable name must be provided as a string in
# the Derive expression.

general_time_elasticity_train = Derive(prob_train, 'TRAIN_TT') * TRAIN_TT / prob_train
general_time_elasticity_swissmetro = (
    Derive(prob_swissmetro, 'SM_TT') * SM_TT / prob_swissmetro
)
general_time_elasticity_car = Derive(prob_car, 'CAR_TT') * CAR_TT / prob_car

# %%
# Closed-form elasticity formula for the logit model.
#
# The expression follows the standard result presented in
# Ben-Akiva and Lerman.

logit_time_elasticity_train = TRAIN_AV_SP * (1.0 - prob_train) * TRAIN_TT * b_time / 100
logit_time_elasticity_swissmetro = (
    SM_AV * (1.0 - prob_swissmetro) * SM_TT * b_time / 100
)
logit_time_elasticity_car = CAR_AV_SP * (1.0 - prob_car) * CAR_TT * b_time / 100

# %%
# Define the quantities to be simulated.
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
# Create the Biogeme object used for simulation.
#
# Because probabilities are simulated for all alternatives, including
# observations where an alternative may be unavailable, Biogeme may
# generate warning messages.
biosim = BIOGEME(database, simulate)
biosim.model_name = 'b01d_logit_simul'

# %%
# Load the parameter estimates obtained from a previous estimation.
RESULTS_FILE_NAME = 'saved_results/b01a_logit.yaml'
estimation_results = EstimationResults.from_yaml_file(filename=RESULTS_FILE_NAME)
betas = estimation_results.get_beta_values()


# %%
# Run the simulation and display descriptive statistics.
#
results = biosim.simulate(the_beta_values=betas)
display(results.describe())
