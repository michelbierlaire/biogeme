"""

1a. Estimation of a logit model (Bayesian)
==========================================

This example illustrates the Bayesian estimation of a logit model using
Biogeme and the Swissmetro stated-preference dataset.

The choice situation involves three transportation alternatives:

- train,
- car,
- Swissmetro.

The script is organized into sections separated by ``# %%`` markers.
These markers are important because the examples are automatically
converted into Jupyter notebooks for the documentation. Each ``# %%``
marker defines a new notebook cell.

Tested with Biogeme 3.3.3.

Michel Bierlaire, EPFL
Tue Jun 09 2026, 15:30:00
"""

from pathlib import Path

from IPython.core.display_functions import display

# %%
# Import the processed Swissmetro dataset and all variables used in the specification.
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
    BayesianResults,
    BayesianResultsSummary,
    get_pandas_estimated_parameters,
)
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit

# %%
# Configure the logger. DEBUG provides detailed information about the execution of the example.
logger = blog.get_screen_logger(level=blog.DEBUG)
logger.info('Example b01a_logit.py')


# %%
# Alternative-specific constants.
# By default, Biogeme assigns a normal prior distribution centered on the
# starting value. Bounds, when specified, are also used to truncate the prior.
asc_car = Beta('asc_car', 0, None, None, 0)
asc_train = Beta('asc_train', 0, None, None, 0)

# %%
# The Swissmetro constant is normalized to zero for identification purposes.
# Setting the last argument of Beta to 1 fixes the parameter at its default
# value and removes it from the estimation.
asc_sm = Beta('asc_sm', 0, None, None, 1)

# %%
# Coefficients associated with travel time and travel cost.
# The upper bound is set to zero to enforce a non-positive marginal utility.
b_time = Beta('b_time', 0, None, 0, 0)
b_cost = Beta('b_cost', 0, None, 0, 0)


# %%
# Utility functions for the three alternatives.
v_train = asc_train + b_time * TRAIN_TT_SCALED + b_cost * TRAIN_COST_SCALED
v_sm = asc_sm + b_time * SM_TT_SCALED + b_cost * SM_COST_SCALED
v_car = asc_car + b_time * CAR_TT_SCALED + b_cost * CAR_CO_SCALED

# %%
# Mapping between alternative identifiers and utility functions.
v = {1: v_train, 2: v_sm, 3: v_car}

# %%
# Availability conditions associated with each alternative.
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

# %%
# Log of the choice probability for the logit model.
# This expression defines the contribution of one observation to the log likelihood.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object.
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b01a_logit'

# %%
# Estimate the posterior distribution of the parameters, or read the results if
# already available.
yaml_file = Path('saved_results') / f'{the_biogeme.model_name}.yaml'
try:
    summary_results = BayesianResultsSummary.from_yaml_file(filename=yaml_file)
except FileNotFoundError:
    results: BayesianResults = the_biogeme.bayesian_estimation()
    summary_results = results.to_summary()

# %%
print(summary_results.short_summary())

# %%
# Present the parameter estimates in a pandas table.
pandas_results = get_pandas_estimated_parameters(
    estimation_results=summary_results,
)
display(pandas_results)

# %%
# Report the variables stored in the Bayesian estimation results.
display(summary_results.report_stored_variables())
