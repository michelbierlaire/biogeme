"""

Simulation of panel model
=========================

Calculates each contribution to the log likelihood function using
simulation. We also calculate the individual parameters.

Michel Bierlaire, EPFL
Sat Jun 21 2025, 17:06:31

"""

import sys

import biogeme.biogeme_logging as blog
from IPython.core.display_functions import display
from biogeme.biogeme import BIOGEME
from biogeme.calculator.single_formula import calculate_single_formula_from_expression
from biogeme.expressions import Beta, Draws, MonteCarlo, PanelLikelihoodTrajectory, log
from biogeme.models import logit
from biogeme.results_processing import EstimationResults

# %%
# See the data processing script: :ref:`swissmetro_panel`.
from swissmetro_panel import (
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
# Define the number of draws to be used for Monte-Carlo integration.
NUMBER_OF_DRAWS = 100_000

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b13panel_simul.py')

# %%
# Parameters to be estimated.
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation.
b_time = Beta('b_time', 0, None, None, 0)

# %%
# It is advised not to use 0 as starting value for the following parameter.
b_time_s = Beta('b_time_s', 1, None, None, 0)
b_time_rnd = b_time + b_time_s * Draws('b_time_rnd', 'NORMAL_ANTI')

# %%
# We do the same for the constants, to address serial correlation.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_car_s = Beta('asc_car_s', 1, None, None, 0)
asc_car_rnd = asc_car + asc_car_s * Draws('asc_car_rnd', 'NORMAL_ANTI')

asc_train = Beta('asc_train', 0, None, None, 0)
asc_train_s = Beta('asc_train_s', 1, None, None, 0)
asc_train_rnd = asc_train + asc_train_s * Draws('asc_train_rnd', 'NORMAL_ANTI')

asc_sm = Beta('asc_sm', 0, None, None, 0)
asc_sm_s = Beta('asc_sm_s', 1, None, None, 0)
asc_sm_rnd = asc_sm + asc_sm_s * Draws('asc_sm_rnd', 'NORMAL_ANTI')

# %%
# Definition of the utility functions.
v_train = asc_train_rnd + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm_rnd + b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car_rnd + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional on the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel).
choice_probability_one_observation = logit(v, av, CHOICE)

# %%
# Conditional on the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
conditional_trajectory_probability = PanelLikelihoodTrajectory(
    choice_probability_one_observation
)

# %%
# We integrate over the random parameters using Monte-Carlo
log_probability = log(MonteCarlo(conditional_trajectory_probability))

# %%
# We retrieve the parameters estimates.
try:
    results = EstimationResults.from_yaml_file(filename='saved_results/b12panel.yaml')
except FileNotFoundError:
    sys.exit(
        'Run first the script b12panel.py '
        'in order to generate the '
        'file b12panel.yaml and move it to the directory saved_results.'
    )

# %%
# Simulate to recalculate the log likelihood directly from the
# formula, without the Biogeme object
simulated_loglike = calculate_single_formula_from_expression(
    expression=log_probability,
    database=database,
    number_of_draws=NUMBER_OF_DRAWS,
    the_betas=results.get_beta_values(),
    avoid_analytical_second_derivatives=False,
    numerically_safe=False,
)

# %%
print(f'Simulated log likelihood: {simulated_loglike}')

# %%
# We also calculate the individual parameters for the time coefficient.
numerator = MonteCarlo(b_time_rnd * conditional_trajectory_probability)
denominator = MonteCarlo(conditional_trajectory_probability)

simulate = {
    'Numerator': numerator,
    'Denominator': denominator,
}

# %%
# Creation of the Biogeme object.
biosim = BIOGEME(database, simulate, number_of_draws=NUMBER_OF_DRAWS, seed=1223)

# %%
# Simulation.
sim = biosim.simulate(results.get_beta_values())

# %%
sim['Individual-level parameters'] = sim['Numerator'] / sim['Denominator']

# %%
print(f'{sim.shape=}')

# %%
display(sim)
