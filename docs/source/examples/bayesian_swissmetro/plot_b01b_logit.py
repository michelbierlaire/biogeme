"""

1b. Estimation of a logit model with custom priors (Bayesian)
=============================================================

This example estimates the same Bayesian logit model as in Example 1a,
but illustrates how to specify custom prior distributions for the model
parameters.

The example demonstrates:

- the use of non-default prior distributions,
- the specification of custom PyMC prior functions,
- the use of truncated prior distributions to enforce sign constraints,
- the estimation of a Bayesian logit model,
- the extraction and display of the estimation results.

The `# %%` markers are used to separate the script into notebook cells
when the example gallery is converted into Jupyter notebooks.

Tested with Biogeme 3.3.3.

Michel Bierlaire, EPFL
Tue Jun 09 2026, 15:10:00
"""

import pymc as pm
from IPython.core.display_functions import display
from pytensor.tensor.variable import TensorVariable

# %%
# Import the variables and the database prepared in the Swissmetro data-processing example.
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

import biogeme.biogeme_logging as blog
from biogeme.bayesian_estimation import (
    BayesianResultsSummary,
    get_pandas_estimated_parameters,
)
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit

# %%
# Configure the Biogeme logger. Increasing the verbosity level provides
# additional information about the estimation process.
logger = blog.get_screen_logger(level=blog.DEBUG)
logger.info('Example b01b_logit.py')


# %%
# Define the alternative-specific constants. By default, Biogeme uses a
# normal prior distribution, possibly truncated if bounds are specified.
# Here, we increase the prior standard deviation by setting
# `sigma_prior=30`.
asc_car = Beta('asc_car', 0, None, None, 0, sigma_prior=30)
asc_train = Beta('asc_train', 0, None, None, 0, sigma_prior=30)


# For the remaining parameters, we define a custom Student-t prior.
# When bounds are specified, the prior is truncated accordingly.
# Consult the PyMC documentation for the available probability
# distributions.
def negative_student_prior(
    name: str,
    initial_value: float,
    lower_bound: float | None,
    upper_bound: float | None,
) -> TensorVariable:
    """Generate a Student-t prior distribution."""

    if lower_bound is None and upper_bound is None:
        return pm.StudentT(name=name, mu=initial_value, sigma=10.0, nu=5.0)

    rv = pm.StudentT.dist(mu=initial_value, sigma=10.0, nu=5.0)
    return pm.Truncated(
        name, rv, lower=lower_bound, upper=upper_bound, initval=initial_value
    )


# %%
# Define the coefficients associated with travel time and travel cost.
# The upper bound is fixed to zero in order to reflect the prior
# assumption that increases in time and cost reduce utility.
b_time = Beta('b_time', -1, None, 0, 0, prior=negative_student_prior)
b_cost = Beta('b_cost', -1, None, 0, 0, prior=negative_student_prior)


# %%
# Define the utility functions.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_sm = b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Associate each utility function with the corresponding alternative identifier.
v = {1: v_train, 2: v_sm, 3: v_car}

# %%
# Associate the availability conditions with each alternative.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Define the log-likelihood contribution of each observation.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object. We explicitly request the standard PyMC
# sampler. See the PyMC documentation for details about the available
# sampling algorithms.
the_biogeme = BIOGEME(database, log_probability, mcmc_sampling_strategy='pymc')
the_biogeme.model_name = 'b01b_logit'

# %%
# Estimate the parameters. The estimation code is placed inside a
# function protected by the standard Python main guard. This is required
# because PyMC may use multiprocessing, and worker processes re-import
# the script.


def main() -> None:
    """Estimate the Bayesian logit model and display the results."""

    try:
        results = BayesianResultsSummary.from_yaml_file(
            filename=f'saved_results/{the_biogeme.model_name}.yaml'
        )
    except FileNotFoundError:
        results = the_biogeme.bayesian_estimation().to_summary()

    # %%
    # Display a short summary of the estimation results.
    print(results.short_summary())

    # %%
    # Convert the estimated parameters into a pandas DataFrame.
    pandas_results = get_pandas_estimated_parameters(
        estimation_results=results,
    )
    display(pandas_results)


if __name__ == '__main__':
    main()
