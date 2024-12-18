"""

Simulation of panel model
=========================

Calculates each contribution to the log likelihood function using
simulation. We also calculate the individual parameters.

:author: Michel Bierlaire, EPFL
:date: Sun Apr  9 18:16:29 2023

"""

import sys

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import (
    Beta,
    bioDraws,
    PanelLikelihoodTrajectory,
    MonteCarlo,
    log,
)
from biogeme.models import logit
from biogeme.results_processing import EstimationResults

# %%
# See the data processing script: :ref:`swissmetro_panel`.
from swissmetro_panel import (
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

print(f'Samples size = {database.get_sample_size()}')

# %%
# We use a low number of draws, as the objective is to illustrate the
# syntax. In practice, this value is insufficient to have a good
# approximation of the integral.
NUMBER_OF_DRAWS = 50

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b13panel_simul.py')

# %%
# Parameters.
B_COST = Beta('B_COST', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation.
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL_ANTI')

# %%
# We do the same for the constants, to address serial correlation.
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_CAR_S = Beta('ASC_CAR_S', 1, None, None, 0)
ASC_CAR_RND = ASC_CAR + ASC_CAR_S * bioDraws('ASC_CAR_RND', 'NORMAL_ANTI')

ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_TRAIN_S = Beta('ASC_TRAIN_S', 1, None, None, 0)
ASC_TRAIN_RND = ASC_TRAIN + ASC_TRAIN_S * bioDraws('ASC_TRAIN_RND', 'NORMAL_ANTI')

ASC_SM = Beta('ASC_SM', 0, None, None, 1)
ASC_SM_S = Beta('ASC_SM_S', 1, None, None, 0)
ASC_SM_RND = ASC_SM + ASC_SM_S * bioDraws('ASC_SM_RND', 'NORMAL_ANTI')

# %%
# Definition of the utility functions.
V1 = ASC_TRAIN_RND + B_TIME_RND * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
V2 = ASC_SM_RND + B_TIME_RND * SM_TT_SCALED + B_COST * SM_COST_SCALED
V3 = ASC_CAR_RND + B_TIME_RND * CAR_TT_SCALED + B_COST * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
V = {1: V1, 2: V2, 3: V3}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional on the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel).
obsprob = logit(V, av, CHOICE)

# %%
# Conditional on the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
condprobIndiv = PanelLikelihoodTrajectory(obsprob)

# %%
# We integrate over the random parameters using Monte-Carlo
logprob = log(MonteCarlo(condprobIndiv))

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
simulated_loglike = logprob.get_value_c(
    database=database,
    betas=results.get_beta_values(),
    number_of_draws=NUMBER_OF_DRAWS,
    aggregation=True,
    prepare_ids=True,
)

# %%
print(f'Simulated log likelihood: {simulated_loglike}')

# %%
# We also calculate the individual parameters for the time coefficient.
numerator = MonteCarlo(B_TIME_RND * condprobIndiv)
denominator = MonteCarlo(condprobIndiv)

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
