"""

12. Mixture of logit with panel data
====================================

Bayesian estimation of a mixture of logit models.
The datafile is organized as panel data.
Note that, with Bayesian estimation, there is no need to calculate a Monte-Carlo integration.

Michel Bierlaire, EPFL
Thu Nov 20 2025, 14:50:04
"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import (
    BayesianResults,
    FigureSize,
    generate_html_file as generate_bayesian_html_file,
    get_pandas_estimated_parameters,
)
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, DistributedParameter, Draws
from biogeme.filenames import get_new_file_name
from biogeme.models import loglogit

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
logger.info('Example b12_panel.py')

# %%
# The scale parameters must stay away from zero. We define a small but positive lower bound
POSITIVE_LOWER_BOUND = 1.0e-5

# %%
# Parameters to be estimated.
b_cost = Beta('b_cost', 0, None, 0, 0)

# %%
# Define a random parameter, normally distributed across individuals,
# designed to be used for Monte-Carlo simulation.
b_time = Beta('b_time', 0, None, 0, 0)
b_time_s = Beta('b_time_s', 1, POSITIVE_LOWER_BOUND, None, 0)
b_time_eps = Draws('b_time_eps', 'NORMAL')
b_time_eps.set_draw_per_individual()
b_time_rnd = DistributedParameter('b_time_rnd', b_time + b_time_s * b_time_eps)

# %%
# We do the same for the constants, to address serial correlation.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_car_s = Beta('asc_car_s', 1, POSITIVE_LOWER_BOUND, None, 0)
asc_car_eps = Draws('asc_car_eps', 'NORMAL')
asc_car_eps.set_draw_per_individual()
asc_car_rnd = DistributedParameter('asc_car_rnd', asc_car + asc_car_s * asc_car_eps)

asc_train = Beta('asc_train', 0, None, None, 0)
asc_train_s = Beta('asc_train_s', 1, POSITIVE_LOWER_BOUND, None, 0)
asc_train_eps = Draws('asc_train_eps', 'NORMAL')
asc_car_eps.set_draw_per_individual()
asc_train_rnd = DistributedParameter(
    'asc_train_rnd', asc_train + asc_train_s * asc_train_eps
)

asc_sm = Beta('asc_sm', 0, None, None, 0)
asc_sm_s = Beta('asc_sm_s', 1, POSITIVE_LOWER_BOUND, None, 0)
asc_sm_eps = Draws('asc_sm_eps', 'NORMAL')
asc_sm_eps.set_draw_per_individual()
asc_sm_rnd = DistributedParameter('asc_sm_rnd', asc_sm + asc_sm_s * asc_sm_eps)

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
log_probability_one_observation = loglogit(v, av, CHOICE)

# %%
# As the objective is to illustrate the
# syntax, we calculate the Monte-Carlo approximation with a small
# number of draws.
the_biogeme = BIOGEME(
    database,
    log_probability_one_observation,
    warmup=5000,
    bayesian_draws=5000,
    chains=4,
)
the_biogeme.model_name = 'b12_panel'

# %%
# Estimate the parameters.
try:
    results = BayesianResults.from_netcdf(
        filename=f'saved_results/{the_biogeme.model_name}.nc'
    )
    html_filename = get_new_file_name(the_biogeme.model_name, "html")
    generate_bayesian_html_file(
        filename=html_filename,
        estimation_results=results,
        figure_size=FigureSize.LARGE,
    )
    print(f'{html_filename} generated')

except FileNotFoundError:
    results = the_biogeme.bayesian_estimation()

# %%
print(results.short_summary())

# %%
pandas_results = get_pandas_estimated_parameters(estimation_results=results)
display(pandas_results)


print(results.idata.posterior.dims)
print(results.idata.posterior)
