"""

24. Mixture of logit with Halton draws
======================================

Example of a mixture of logit models, using quasi Monte-Carlo integration with
Halton draws (base 5). The mixing distribution is normal.

Michel Bierlaire, EPFL
Sat Jun 28 2025, 12:45:21


"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Draws, MonteCarlo, log
from biogeme.models import logit
from biogeme.results_processing import (
    EstimationResults,
    get_pandas_estimated_parameters,
)

# %%
# See the data processing script: :ref:`swissmetro_data`.
from swissmetro_data import (
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
logger.info('Example b24_halton_mixture.py')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for Monte-Carlo simulation.
b_time = Beta('b_time', 0, None, None, 0)

# %%
# It is advised not to use 0 as starting value for the following parameter.
b_time_s = Beta('b_time_s', 1, None, None, 0)
# %%
# Define a random parameter with a normal distribution, designed to be used
# for quasi Monte-Carlo simulation with Halton draws (base 5).
b_time_rnd = b_time + b_time_s * Draws('b_time_rnd', 'NORMAL_HALTON5')

# %%
# Definition of the utility functions.
v_train = asc_train + b_time_rnd * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_swissmetro = asc_sm + b_time_rnd * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time_rnd * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_swissmetro, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Conditional on b_time_rnd, we have a logit model (called the kernel)
conditional_probability = logit(v, av, CHOICE)

# %%
# We integrate over b_time_rnd using Monte-Carlo.
log_probability = log(MonteCarlo(conditional_probability))

# %%
# These notes will be included as such in the report file.
USER_NOTES = (
    'Example of a mixture of logit models with three alternatives, '
    'approximated using Monte-Carlo integration with Halton draws.'
)

# %%
# As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws.
the_biogeme = BIOGEME(
    database, log_probability, user_notes=USER_NOTES, number_of_draws=10_000, seed=1223
)
the_biogeme.model_name = 'b24_halton_mixture'

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
