"""

1b. Estimation of a logit model (Bayesian)
==========================================

 This example illustrates how to change the prior distribution of the parameters.

Michel Bierlaire, EPFL
Thu Nov 20 2025, 08:58:43
"""

import pymc as pm
from IPython.core.display_functions import display
from pytensor.tensor.var import TensorVariable

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import BayesianResults, get_pandas_estimated_parameters
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit

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

# %%
# See the data processing script: :ref:`swissmetro_data`.

# %%
# The logger sets the verbosity of Biogeme. By default, Biogeme is quite silent and generates only warnings.
# To have more information about what it happening behind the scene, the level should be set to `blog.INFO`.
logger = blog.get_screen_logger(level=blog.DEBUG)
logger.info('Example b01b_logit.py')


# %%
# Parameters to be estimated: alternative specific constants.
# By default, the prior distribution is normal, possibly truncated if bounds are defined, with the mean
# defined by the user, and the scale parameter explicitly defined. Here, we decide to replace the default sigma
# by `sigma_prior=30`.
asc_car = Beta('asc_car', 0, None, None, 0, sigma_prior=30)
asc_train = Beta('asc_train', 0, None, None, 0, sigma_prior=30)


# For the other parameters, we use a student distribution truncated at zero. We need to explicitly implement this prior,
# as illustrated below. Consult the PyMc documentation for the catalog of available distributions.
def negative_student_prior(
    name: str,
    initial_value: float,
    lower_bound: float | None,
    upper_bound: float | None,
) -> TensorVariable:
    """
    Example of a Student-t prior.
    """

    if lower_bound is None and upper_bound is None:
        return pm.StudentT(name=name, mu=initial_value, sigma=10.0, nu=5.0)

    rv = pm.StudentT.dist(mu=initial_value, sigma=10.0, nu=5.0)
    return pm.Truncated(
        name, rv, lower=lower_bound, upper=upper_bound, initval=initial_value
    )


# %%
# Coefficients of the attributes. It is useful to set the upper bound to 0 to reflect the prior assumption about
# the sign of those parameters.
b_time = Beta('b_time', -1, None, 0, 0, prior=negative_student_prior)
b_cost = Beta('b_cost', -1, None, 0, 0, prior=negative_student_prior)


# %%
# Definition of the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_sm = b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 2: v_sm, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Definition of the model.
# This is the contribution of each observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object. We illustrate the use of a specific sampler. Here, the default PyMC sampler.
# See the PyMC documentation for details.
the_biogeme = BIOGEME(database, log_probability, mcmc_sampling_strategy="pymc")
the_biogeme.model_name = 'b01b_logit'

# %%
# Estimate the parameters.
try:
    results = BayesianResults.from_netcdf(
        filename=f'saved_results/{the_biogeme.model_name}.nc'
    )
except FileNotFoundError:
    results = the_biogeme.bayesian_estimation()

# %%
print(results.short_summary())

# %%
# Get the results in a pandas table
pandas_results = get_pandas_estimated_parameters(
    estimation_results=results,
)
display(pandas_results)
