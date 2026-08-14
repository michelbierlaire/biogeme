"""

23a. Binary logit model
=======================

Bayesian estimation of a binary logit model.
Two alternatives: Train and Car.

Michel Bierlaire, EPFL
Tue Nov 18 2025, 18:42:42

"""

from pathlib import Path

from IPython.core.display_functions import display

# %%
# See the data processing script ``swissmetro_binary.py``.
from swissmetro_binary import (
    CAR_AV_SP,
    CAR_CO_SCALED,
    CAR_TT_SCALED,
    CHOICE,
    TRAIN_AV_SP,
    TRAIN_COST_SCALED,
    TRAIN_TT_SCALED,
    database,
)

from biogeme.bayesian_estimation import (
    BayesianResults,
    BayesianResultsSummary,
    get_pandas_estimated_parameters,
)
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta
from biogeme.models import loglogit

# %%
# Parameters to be estimated.
asc_car = Beta('asc_car', 0, None, None, 0)
b_time_car = Beta('b_time_car', 0, None, 0, 0)
b_time_train = Beta('b_time_train', 0, None, 0, 0)
b_cost_car = Beta('b_cost_car', 0, None, 0, 0)
b_cost_train = Beta('b_cost_train', 0, None, 0, 0)

# %%
# Definition of the utility functions.
# We estimate a binary logit model. There are only two alternatives.
v_train = b_time_train * TRAIN_TT_SCALED + b_cost_train * TRAIN_COST_SCALED
v_car = asc_car + b_time_car * CAR_TT_SCALED + b_cost_car * CAR_CO_SCALED

# %%
# Associate utility functions with the numbering of alternatives.
v = {1: v_train, 3: v_car}

# %%
# Associate the availability conditions with the alternatives.
av = {1: TRAIN_AV_SP, 3: CAR_AV_SP}

# %%
# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
log_probability = loglogit(v, av, CHOICE)

# %%
# Create the Biogeme object
the_biogeme = BIOGEME(database, log_probability)
the_biogeme.model_name = 'b23a_binary_logit'

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
