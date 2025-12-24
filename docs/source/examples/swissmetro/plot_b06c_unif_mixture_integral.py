""" "

6c. Mixture of logit models with uniform distribution and numerical integration
===============================================================================

Example of a mixture of logit models, using numerical integration.
The mixing distribution is uniform.

Michel Bierlaire, EPFL
Fri Jun 20 2025, 10:47:24

"""

from IPython.core.display_functions import display

import biogeme.biogeme_logging as blog
from biogeme.biogeme import BIOGEME
from biogeme.distributions import normalpdf
from biogeme.expressions import (
    Beta,
    IntegrateNormal,
    RandomVariable,
    exp,
    log,
)
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
logger.info('Example b06unif_mixture_integral.py')

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)
asc_sm = Beta('asc_sm', 0, None, None, 1)
b_cost = Beta('b_cost', 0, None, None, 0)

# %%
# Define a random parameter, normally distributed, designed to be used
# for numerical integration
b_time = Beta('b_time', 0, None, None, 0)
b_time_s = Beta('b_time_s', 1, None, None, 0)
omega = RandomVariable('omega')

# %%
# .. |infinity| unicode:: U+221E
#    :trim:
#
# As the numerical integration ranges from -|infinity| \  to + |infinity| ,
# we need to perform a change of variable in order to integrate
# between -1 and 1.
LOWER_BND = -1
UPPER_BND = 1
x = LOWER_BND + (UPPER_BND - LOWER_BND) / (1 + exp(-omega))
dx = (UPPER_BND - LOWER_BND) * exp(-omega) / ((1 + exp(-omega)) ** 2)
b_time_rnd = b_time + b_time_s * x

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
# Conditional on omega, we have a logit model (called the kernel).
conditional_probability = logit(v, av, CHOICE)

# %%
# pdf of the uniform distribution
pdf_uniform = 1 / (UPPER_BND - LOWER_BND)

# %%
# As the `IntegrateNormal` expression is designed for a normal distribution, we need to divide by the pdf of
# the normal distribution, and multiply by the pdf of the uniform distribution, after applying the change of variable.
new_integrand = conditional_probability * dx * pdf_uniform / normalpdf(omega)


# %%
# We integrate over omega using numerical integration. To illustrate the syntax, we specific the number of quadrature
# points to be used.
log_probability = log(
    IntegrateNormal(
        new_integrand,
        'omega',
        number_of_quadrature_points=60,
    )
)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b06c_unif_mixture_integral'

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
