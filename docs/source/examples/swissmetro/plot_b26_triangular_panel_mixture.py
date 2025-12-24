"""

26. Triangular mixture with panel data
======================================

 Example of a mixture of logit models, using Monte-Carlo integration.
 The mixing distribution is user-defined (triangular, here).
 The datafile is organized as panel data.

Michel Bierlaire, EPFL

"""

import numpy as np
from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.draws import RandomNumberGeneratorTuple
from biogeme.expressions import (
    Beta,
    Draws,
    MonteCarlo,
    PanelLikelihoodTrajectory,
    log,
)
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

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

logger = blog.get_screen_logger(level=blog.INFO)
logger.info('Example b26_triangular_panel_mixture.py')


# %%
# Function generating the draws.
def the_triangular_generator(sample_size: int, number_of_draws: int) -> np.ndarray:
    """
    Provide my own random number generator to the database.
    See the `numpy.random` documentation to obtain a list of other distributions.
    """
    return np.random.triangular(-1, 0, 1, (sample_size, number_of_draws))


# %%
# Associate the function with a name.
my_random_number_generators = {
    'TRIANGULAR': RandomNumberGeneratorTuple(
        the_triangular_generator,
        'Draws from a triangular distribution',
    )
}

# %%
# Parameters to be estimated.
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation.

# %%
# Mean of the distribution.
b_time = Beta('b_time', 0, None, None, 0)

# %%
# Scale of the distribution. It is advised not to use 0 as starting
# value for the following parameter.
b_time_s = Beta('b_time_s', 1, None, None, 0)
b_time_rnd = b_time + b_time_s * Draws('b_time_rnd', 'TRIANGULAR')

# %%
# We do the same for the constants, to address serial correlation.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_car_s = Beta('asc_car_s', 1, None, None, 0)
asc_car_rnd = asc_car + asc_car_s * Draws('asc_car_rnd', 'TRIANGULAR')

asc_train = Beta('asc_train', 0, None, None, 0)
asc_train_s = Beta('asc_train_s', 1, None, None, 0)
asc_train_rnd = asc_train + asc_train_s * Draws('asc_train_rnd', 'TRIANGULAR')

asc_sm = Beta('asc_sm', 0, None, None, 1)
asc_sm_s = Beta('asc_sm_s', 1, None, None, 0)
asc_sm_rnd = asc_sm + asc_sm_s * Draws('asc_sm_rnd', 'TRIANGULAR')

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
# Conditional to the random parameters, the likelihood of one observation is
# given by the logit model (called the kernel).
one_observation_conditional_probability = logit(v, av, CHOICE)

# %%
# Conditional on the random parameters, the likelihood of all observations for
# one individual (the trajectory) is the product of the likelihood of
# each observation.
trajectory_conditional_probability = PanelLikelihoodTrajectory(
    one_observation_conditional_probability
)

# %%
# We integrate over the random parameters using Monte-Carlo
log_probability = log(MonteCarlo(trajectory_conditional_probability))

# %%
the_biogeme = BIOGEME(
    database,
    log_probability,
    random_number_generators=my_random_number_generators,
    number_of_draws=10_000,
    seed=1223,
)
the_biogeme.model_name = 'b26_triangular_panel_mixture'

# %%
# Estimate the parameters.
try:
    results = EstimationResults.from_yaml_file(
        filename=f'saved_results/{the_biogeme.model_name}.yaml'
    )
except FileNotFoundError:
    results = the_biogeme.estimate()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)
